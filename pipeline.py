"""
pipeline.py

Caminho dos dados: Sample (bruto) -> alinhamento de eixos -> remoção do bias do
giroscópio -> FILTRO -> PoseState.

O filtro em si fica em filters/ (e madgwick.py). Aqui não há matemática de
orientação, só a "cola".
"""

import os
import threading
from collections import deque
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

import filter_config
import geometry
import imu_axes
import position
import sphere_color
import quaternion as Q
from filters import FILTERS, FilterInput, make_filter
from sample import Sample
from sensor_io import Frames, load_config

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_NAME = "axes_config.json"


def find_file(name):
    for folder in (os.getcwd(), APP_DIR):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    return None


@dataclass
class PoseState:
    t_s: float
    q: np.ndarray          # [w, x, y, z]
    euler_deg: np.ndarray  # roll, pitch, yaw (só para exibição)
    tilt_deg: float        # ângulo entre o eixo Z do sensor e a vertical
    accel_norm: float

    def as_row(self):
        return [*self.euler_deg, self.tilt_deg, *self.q, self.accel_norm]


@dataclass
class GeometryState:
    """Resultado dos sensores de distância (h1 = esquerda, h2 = direita)."""
    t_s: float
    h1: float
    h2: float
    valid: bool
    reason: str = ""
    result: Optional[geometry.GeometryResult] = None   # pontos para desenhar

    @property
    def alpha_deg(self):
        return self.result.alpha_deg if self.result else float("nan")

    @property
    def H(self):
        return self.result.H if self.result else float("nan")

    @property
    def Y(self):
        return self.result.Y if self.result else float("nan")

    @property
    def warning(self):
        return self.result.warning if self.result else ""

    def as_row(self):
        return [self.h1, self.h2, self.alpha_deg, self.H, self.Y]


TOF_CAL_N = 50   # amostras (~1 s a 50 Hz) usadas na calibração do ToF
TOF_AVG_N = 5   # amostras da média móvel dos sensores de distância
CONS_MIN_RATE = np.radians(8.0)   # só avalia giro × acelerômetro acima de 8 °/s
CONS_DECAY = 0.995                # janela ≈ 200 amostras
CONS_MIN_ENERGY = 5e-4            # movimento acumulado mínimo para opinar


class Pipeline:
    def __init__(self):
        self.lock = threading.RLock()
        self.force_default_frames = False   # usado pela fonte Demo
        self.config_path = None
        self._frames_cfg = Frames(None)
        self.reload_alignment()

        # parâmetros fixos do filtro: filter_config.json
        self.use_mag = True
        self.filter_name = ""
        self.filter = None
        self.filter_message = ""
        self.color_cfg = sphere_color.ColorConfig()
        self._fusion = position.LongitudeFusion()   # longitude: orientação + cor
        self.imu_axes = imu_axes.load()            # troca de eixos / sentido (imu_axes.json)
        self._cons = np.zeros(3)                   # giro × acelerômetro: Σ d·p, Σ p·p, Σ d·d
        self._cons_prev = None
        self.reload_configs()

        self.gyro_bias = np.zeros(3)
        self._calib_target = 0
        self._calib_buf = []
        self._prev_t_us: Optional[int] = None
        self._t0_us: Optional[int] = None

        # sensores de distância (geometry.py)
        self.geometry_cfg = geometry.load_config()
        self._geo_t0_us: Optional[int] = None
        self.tof_avg_n = TOF_AVG_N
        self._tof_win = deque(maxlen=self.tof_avg_n)   # (dist1, dist2)
        self._tof_cal = None       # calibração em andamento: {"H": altura, "buf": [(d1, d2)]}
        self._tof_cal_text = ""

    # -------------------------------------------------------- alinhamento
    def reload_alignment(self):
        with self.lock:
            self.config_path = find_file(CONFIG_NAME)
            cfg = load_config(self.config_path) if self.config_path else None
            self._frames_cfg = Frames(cfg)
            if hasattr(self, "gyro_bias"):
                self.gyro_bias[:] = 0.0

    @property
    def alignment_loaded(self) -> bool:
        return self._frames_cfg.from_file

    @property
    def frames(self) -> Frames:
        if not self.force_default_frames:
            return self._frames_cfg
        return Frames(None)

    # ------------------------------------------------------------- filtro
    def reload_configs(self):
        """(Re)lê filter_config.json (recria o filtro) e sphere_config.json (cor da esfera)."""
        cfg = filter_config.load()
        with self.lock:
            self.color_cfg = sphere_color.load()
            self._fusion.rate = self.color_cfg.fusion_rate
            self._fusion.gate = self.color_cfg.fusion_gate_deg
            self.filter_name = cfg.filter
            self.filter = make_filter(cfg.filter, **cfg.params)
            self.use_mag = cfg.use_mag
            self.filter_message = "; ".join(m for m in (cfg.message, self.color_cfg.message) if m)
            self._prev_t_us = None

    def set_filter(self, name: str):
        """Troca o método de orientação ao vivo (parâmetros do filter_config.json só se forem desse filtro)."""
        if name not in FILTERS:
            return
        cfg = filter_config.load()
        params = cfg.all_params.get(name, {})
        with self.lock:
            self.filter_name = name
            self.filter = make_filter(name, **params)
            self._prev_t_us = None
            self._fusion.reset()

    # ------------------------------------------ parâmetros do algoritmo
    def filter_params(self) -> dict:
        with self.lock:
            return dict(self.filter.values)

    def set_filter_param(self, key: str, value: float):
        with self.lock:
            self.filter.set_param(key, value)

    def reset_filter_params(self):
        with self.lock:
            for p in self.filter.params:
                self.filter.set_param(p.key, p.default)

    def set_use_mag(self, use: bool):
        with self.lock:
            self.use_mag = bool(use)

    def save_filter_config(self) -> str:
        """Grava filtro atual + parâmetros + use_mag em filter_config.json."""
        with self.lock:
            return filter_config.save(self.filter_name, self.use_mag, self.filter.values)

    # ----------------------------------------------- eixos e sentido do IMU
    def set_axes(self, names, invert_rotation: bool, save: bool = True):
        """Troca de eixos (vale para acel, giro e mag) e inversão do giro. Levanta ValueError se inválida."""
        ax = imu_axes.ImuAxes(tuple(names), bool(invert_rotation))
        ax.matrix()   # valida
        with self.lock:
            self.imu_axes = ax
            self.gyro_bias[:] = 0.0          # o bias estava no referencial anterior
            self._cons[:] = 0.0
            self._cons_prev = None
            self.filter.reset()
            self._prev_t_us = None
            self._fusion.reset()
        if save:
            imu_axes.save(ax)
            ax.saved = True

    def _axes(self):
        """(matriz, sinal do giro); a fonte Demo já vem alinhada."""
        if self.force_default_frames:
            return np.eye(3), 1.0
        return self.imu_axes.matrix(), self.imu_axes.gyro_sign

    def _track_consistency(self, a, w, dt):
        """Giro e acelerômetro precisam concordar: da/dt = -ω × a. Acumula a correlação."""
        prev, self._cons_prev = self._cons_prev, a.copy()
        if prev is None or dt <= 0 or np.linalg.norm(w) < CONS_MIN_RATE:
            return
        pred = -np.cross(w, 0.5 * (a + prev)) * dt
        d = a - prev
        self._cons = CONS_DECAY * self._cons + (d @ pred, pred @ pred, d @ d)

    def gyro_consistency(self):
        """Correlação (-1..1) entre o giro e a variação do acelerômetro: + concordam,
        - o giro está invertido em relação ao acelerômetro. None = ainda sem movimento suficiente."""
        with self.lock:
            c0, c1, c2 = self._cons
        if c1 < CONS_MIN_ENERGY or c2 <= 0:
            return None
        return float(c0 / np.sqrt(c1 * c2))

    # ------------------------------------------------------- calibração
    def start_gyro_calibration(self, n: int = 200):
        with self.lock:
            self._calib_target = n
            self._calib_buf = []

    def calibration_text(self) -> str:
        with self.lock:
            if self._calib_target:
                return f"Calibrando giroscópio… {len(self._calib_buf)}/{self._calib_target}"
            if np.any(self.gyro_bias):
                b = np.degrees(self.gyro_bias)
                return f"Bias gyro (°/s): {b[0]:+.2f} {b[1]:+.2f} {b[2]:+.2f}"
            return "Bias do gyro: não calibrado"

    # ---------------------------------------------------------- controle
    def reset(self):
        with self.lock:
            self.filter.reset()
            self._prev_t_us = None
            self._t0_us = None
            self._geo_t0_us = None
            self._tof_win.clear()
            self._tof_cal = None
            self._fusion.reset()
            self._cons[:] = 0.0
            self._cons_prev = None

    # ------------------------------------------------- sensores de distância
    def set_geometry(self, L: float, R: float, save: bool = True,
                     bar_axis: Optional[str] = None, toward_axis: Optional[str] = None) -> str:
        """Define L (distância entre sensores) e R (raio), na unidade das leituras, e
        (opcional) os eixos da placa. Sem eixos = mantém os atuais."""
        if not (L > 0 and R > 0):
            raise ValueError("L e R precisam ser positivos")
        old = self.geometry_cfg
        ba = bar_axis or old.bar_axis
        ta = toward_axis or old.toward_axis
        if not geometry.axes_ok(ba, ta):
            raise ValueError("o eixo da barra e o eixo dos sensores precisam ser de eixos diferentes")
        with self.lock:
            # a calibração do ToF foi calculada com o L e o R antigos: mudou, apaga
            same = float(L) == old.L and float(R) == old.R
            self.geometry_cfg = replace(old, L=float(L), R=float(R), saved=save, bar_axis=ba,
                                        toward_axis=ta,
                                        tof_offset1=old.tof_offset1 if same else 0.0,
                                        tof_offset2=old.tof_offset2 if same else 0.0)
            if not same and (old.tof_offset1 or old.tof_offset2):
                self._tof_cal_text = "L ou R mudou: calibração do ToF apagada"
        return geometry.save_config(self.geometry_cfg) if save else ""

    def process_geometry(self, s: Sample) -> GeometryState:
        with self.lock:
            if self._geo_t0_us is None:
                self._geo_t0_us = s.t_us
            t = (s.t_us - self._geo_t0_us) * 1e-6
            if not s.tof_ok:
                self._tof_win.clear()   # não mistura leituras antigas com as novas
            else:
                if self._tof_cal is not None:   # calibração usa as leituras sem média nem offset
                    self._tof_cal["buf"].append((float(s.dist1), float(s.dist2)))
                    if len(self._tof_cal["buf"]) >= TOF_CAL_N:
                        self._finish_tof_calibration()
                self._tof_win.append((float(s.dist1), float(s.dist2)))
            cfg = self.geometry_cfg
            if s.tof_ok:   # média móvel de N amostras + offset da calibração
                m1, m2 = np.mean(self._tof_win, axis=0)
                h1, h2 = float(m1) + cfg.tof_offset1, float(m2) + cfg.tof_offset2
        if not s.tof_ok:
            return GeometryState(t, float(s.dist1), float(s.dist2), False, "ToF inválido (tof_ok = 0)")
        r = geometry.compute(h1, h2, cfg.L, cfg.R)
        return GeometryState(t, h1, h2, r.valid, r.reason, r)

    # ------------------------------------------------- posição na esfera
    def process_position(self, pose: PoseState, geom: Optional[GeometryState],
                         s: Optional[Sample] = None) -> position.Position:
        """Orientação (pose) × α (geom) dão a distância ao polo; a cor (s) dá a longitude.
        Tudo da MESMA amostra."""
        with self.lock:
            cfg, ccfg = self.geometry_cfg, self.color_cfg
        alpha = geom.alpha_deg if geom is not None and geom.valid else None
        pos = position.compute(pose.t_s, pose.q, alpha, cfg.bar_axis, cfg.toward_axis)

        lon_c = None
        if s is not None and s.color_ok:
            lon_c, _ = sphere_color.read_longitude(s.red, s.green, s.blue, ccfg)
        # na calota vermelha do polo (e muito perto dele) a longitude não existe: cor não conta
        min_polar = max(5.0, sphere_color.pole_cap_deg(cfg.R, ccfg))
        with self.lock:
            lon, src, _ = self._fusion.update(pose.t_s, pos.azimuth_deg, pos.polar_deg, lon_c, min_polar)
        pos = position.with_longitude(pos, lon, src, lon_c)
        pos.radius_mm = float(cfg.R)
        return pos

    # ------------------------------------------------- calibração do ToF
    def start_tof_calibration(self, H_known: float) -> str:
        """Com o objeto parado numa altura H conhecida (mesma unidade de L e R), mede
        TOF_CAL_N amostras e calcula o offset que faz o H calculado bater com H_known.
        Retorna "" se começou, ou o motivo de não poder."""
        with self.lock:
            if not self.geometry_cfg.saved:
                return "salve L e R antes (botão Geometria…)"
            if not H_known > 0:
                return "a altura precisa ser positiva"
            self._tof_cal = {"H": float(H_known), "buf": []}
            self._tof_cal_text = ""
        return ""

    def clear_tof_calibration(self):
        with self.lock:
            self._tof_cal = None
            cfg = self.geometry_cfg
            self.geometry_cfg = replace(cfg, tof_offset1=0.0, tof_offset2=0.0)
            self._tof_cal_text = "calibração do ToF removida"
            if cfg.saved:
                geometry.save_config(self.geometry_cfg)

    def _finish_tof_calibration(self):   # chamado com o lock tomado
        cal, self._tof_cal = self._tof_cal, None
        h1, h2 = np.mean(cal["buf"], axis=0)
        cfg = self.geometry_cfg
        # os dois sensores devem ler o mesmo valor (α = 0): h verdadeiro a partir de H
        h, why = geometry.true_distance(cal["H"], cfg.L, cfg.R)
        if h is None:
            self._tof_cal_text = f"calibração falhou: {why}"
            return
        off1, off2 = h - float(h1), h - float(h2)
        self.geometry_cfg = replace(cfg, tof_offset1=off1, tof_offset2=off2)
        try:
            geometry.save_config(self.geometry_cfg)
            self._tof_cal_text = (f"ToF calibrado em H = {cal['H']:g} (h = {h:.1f}): "
                                  f"offset h1 {off1:+.2f}, h2 {off2:+.2f}")
        except OSError as e:
            self._tof_cal_text = f"offsets aplicados, mas não salvou: {e}"

    def tof_calibration_text(self) -> str:
        with self.lock:
            if self._tof_cal is not None:
                return f"Calibrando ToF… {len(self._tof_cal['buf'])}/{TOF_CAL_N} (parado, sensores lendo o mesmo valor)"
            if self._tof_cal_text:
                return self._tof_cal_text
            c = self.geometry_cfg
            if c.tof_offset1 or c.tof_offset2:
                return f"ToF: offset h1 {c.tof_offset1:+.2f}, h2 {c.tof_offset2:+.2f}"
            return "ToF: sem calibração"

    # ------------------------------------------------------------ passo
    def process(self, s: Sample) -> Optional[PoseState]:
        with self.lock:
            if not s.imu_ok:
                self._prev_t_us = None
                return None
            if self._t0_us is None:
                self._t0_us = s.t_us

            frames = self.frames
            P, sign = self._axes()
            accel = P @ s.accel
            gyro = sign * (P @ frames.gyro(s.gyro))

            if self._calib_target:
                self._calib_buf.append(gyro.copy())
                if len(self._calib_buf) >= self._calib_target:
                    self.gyro_bias = np.mean(self._calib_buf, axis=0)
                    self._calib_target = 0
                    self._calib_buf = []

            if self._prev_t_us is None:
                dt = 0.0
            else:
                dt = (s.t_us - self._prev_t_us) * 1e-6
                if dt < 0 or dt > 0.5:
                    dt = 0.0
            self._prev_t_us = s.t_us

            w = gyro - self.gyro_bias
            self._track_consistency(accel, w, dt)
            mag = P @ frames.mag(s.mag) if self.use_mag else np.zeros(3)

            q = self.filter.update(FilterInput(accel=accel, gyro=w, mag=mag, dt=dt))

            q = np.asarray(q, dtype=float)
            tilt = np.degrees(np.arccos(np.clip(Q.to_matrix(q)[2, 2], -1.0, 1.0)))
            return PoseState(
                t_s=(s.t_us - self._t0_us) * 1e-6,
                q=q,
                euler_deg=Q.to_euler_deg(q),
                tilt_deg=float(tilt),
                accel_norm=float(np.linalg.norm(s.accel)),
            )
