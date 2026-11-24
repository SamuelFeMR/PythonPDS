"""
pipeline.py

Caminho dos dados: Sample (bruto) -> alinhamento de eixos -> remoção do bias do
giroscópio -> FILTRO -> PoseState.

O filtro em si fica em filters/ (e madgwick.py). Aqui não há matemática de
orientação, só a "cola".
"""

import os
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

import geometry
import quaternion as Q
from filters import DEFAULT_FILTER, FILTERS, FilterInput, make_filter
from sample import Sample
from sensor_io import Frames, load_config

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_NAME = "axes_config.json"


def find_config():
    for folder in (os.getcwd(), APP_DIR):
        path = os.path.join(folder, CONFIG_NAME)
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


class Pipeline:
    def __init__(self):
        self.lock = threading.RLock()
        self.use_mag = True
        self.align_enabled = True
        self.force_default_frames = False   # usado pela fonte Demo
        self.config_path = None
        self._frames_cfg = Frames(None)
        self.reload_alignment()

        self._saved_params = {}
        self.filter_name = DEFAULT_FILTER
        self.filter = make_filter(DEFAULT_FILTER)

        self.gyro_bias = np.zeros(3)
        self._calib_target = 0
        self._calib_buf = []
        self._prev_t_us: Optional[int] = None
        self._t0_us: Optional[int] = None

        # sensores de distância (geometry.py)
        self.geometry_cfg = geometry.load_config()
        self._geo_t0_us: Optional[int] = None

    # -------------------------------------------------------- alinhamento
    def reload_alignment(self):
        with self.lock:
            self.config_path = find_config()
            cfg = load_config(self.config_path) if self.config_path else None
            self._frames_cfg = Frames(cfg)
            if hasattr(self, "gyro_bias"):
                self.gyro_bias[:] = 0.0

    @property
    def alignment_loaded(self) -> bool:
        return self._frames_cfg.from_file

    @property
    def frames(self) -> Frames:
        if self.align_enabled and not self.force_default_frames:
            return self._frames_cfg
        return Frames(None)

    def set_align_enabled(self, enabled: bool):
        with self.lock:
            self.align_enabled = bool(enabled)
            self.gyro_bias[:] = 0.0   # o bias estava no referencial anterior

    # ------------------------------------------------------------- filtro
    def set_filter(self, name: str):
        with self.lock:
            self._saved_params[self.filter_name] = dict(self.filter.values)
            self.filter_name = name
            self.filter = make_filter(name, **self._saved_params.get(name, {}))
            self._prev_t_us = None

    def set_filter_param(self, key: str, value: float):
        with self.lock:
            self.filter.set_param(key, value)

    def filter_names(self):
        return list(FILTERS)

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

    # ------------------------------------------------- sensores de distância
    def set_geometry(self, L: float, R: float, save: bool = True) -> str:
        """Define L (distância entre sensores) e R (raio do cilindro), na unidade das leituras."""
        if not (L > 0 and R > 0):
            raise ValueError("L e R precisam ser positivos")
        with self.lock:
            self.geometry_cfg = geometry.GeometryConfig(L=float(L), R=float(R), saved=save)
        return geometry.save_config(self.geometry_cfg) if save else ""

    def process_geometry(self, s: Sample) -> GeometryState:
        with self.lock:
            if self._geo_t0_us is None:
                self._geo_t0_us = s.t_us
            t = (s.t_us - self._geo_t0_us) * 1e-6
            cfg = self.geometry_cfg
        if not s.tof_ok:
            return GeometryState(t, float(s.dist1), float(s.dist2), False, "ToF inválido (tof_ok = 0)")
        r = geometry.compute(s.dist1, s.dist2, cfg.L, cfg.R)
        return GeometryState(t, float(s.dist1), float(s.dist2), r.valid, r.reason, r)

    # ------------------------------------------------------------ passo
    def process(self, s: Sample) -> Optional[PoseState]:
        with self.lock:
            if not s.imu_ok:
                self._prev_t_us = None
                return None
            if self._t0_us is None:
                self._t0_us = s.t_us

            frames = self.frames
            gyro = frames.gyro(s.gyro)

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

            mag = frames.mag(s.mag) if self.use_mag else np.zeros(3)

            q = self.filter.update(FilterInput(
                accel=s.accel.copy(), gyro=gyro - self.gyro_bias, mag=mag, dt=dt))

            q = np.asarray(q, dtype=float)
            tilt = np.degrees(np.arccos(np.clip(Q.to_matrix(q)[2, 2], -1.0, 1.0)))
            return PoseState(
                t_s=(s.t_us - self._t0_us) * 1e-6,
                q=q,
                euler_deg=Q.to_euler_deg(q),
                tilt_deg=float(tilt),
                accel_norm=float(np.linalg.norm(s.accel)),
            )
