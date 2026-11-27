"""
imu_calib.py

Calibrações básicas (mínimo/máximo por eixo) do acelerômetro e do magnetômetro,
salvas em imu_calibration.json e aplicadas às leituras brutas:

    leitura_corrigida = (leitura - offset) * escala        (por eixo)

Magnetômetro (girar o sensor em todas as direções):
    offset_i = (max_i + min_i) / 2                 hard iron
    escala_i = média(amplitudes) / amplitude_i      soft iron simplificado (só os eixos)
    amplitude_i = (max_i - min_i) / 2

Acelerômetro (apoiar o sensor parado com cada face para cima e para baixo):
    offset_i = (max_i + min_i) / 2                 erro de zero
    escala_i = 1 / amplitude_i                      cada eixo passa a ler ±1 g
    Só entram amostras com o sensor quase parado (|giro| < 10 °/s), para que a
    leitura seja só a gravidade.
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_NAME = "imu_calibration.json"
SENSORS = ("accel", "mag")
STILL_DPS = 10.0         # acelerômetro: só amostras com |giro| abaixo disso (°/s)
MIN_POINTS = 50


@dataclass
class AxisCal:
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))

    def apply(self, v) -> np.ndarray:
        return (np.asarray(v, float) - self.offset) * self.scale


def compute(sensor: str, points) -> AxisCal:
    """Calibração mín/máx a partir das leituras brutas capturadas. ValueError se faltar dado."""
    p = np.asarray(points, float).reshape(-1, 3)
    if len(p) < MIN_POINTS:
        raise ValueError(f"poucos pontos ({len(p)}; mínimo {MIN_POINTS})")
    lo, hi = p.min(axis=0), p.max(axis=0)
    amp = (hi - lo) / 2.0
    if np.any(amp <= 1e-9):
        raise ValueError("algum eixo não variou: gire o sensor em todas as direções")
    offset = (hi + lo) / 2.0
    if sensor == "mag":
        if amp.min() < 0.5 * amp.max():
            raise ValueError("um eixo variou bem menos que os outros: gire em todas as direções")
        return AxisCal(offset, amp.mean() / amp)
    if sensor == "accel":
        if amp.min() < 0.5 * amp.max():
            raise ValueError("falta apoiar alguma face: cada eixo precisa apontar para cima e para baixo")
        return AxisCal(offset, 1.0 / amp)
    raise ValueError(f"sensor '{sensor}'")


# ----------------------------------------------------------------- arquivo
def config_path() -> str:
    for folder in (os.getcwd(), APP_DIR):
        path = os.path.join(folder, CONFIG_NAME)
        if os.path.exists(path):
            return path
    return os.path.join(APP_DIR, CONFIG_NAME)


def load():
    """({"accel": AxisCal|None, "mag": AxisCal|None}, mensagem de erro ou "")."""
    out = {k: None for k in SENSORS}
    path = config_path()
    if not os.path.exists(path):
        return out, ""
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        for k in SENSORS:
            if d.get(k):
                o, s = np.array(d[k]["offset"], float), np.array(d[k]["scale"], float)
                if o.shape != (3,) or s.shape != (3,) or not np.all(np.isfinite(s)):
                    raise ValueError(k)
                out[k] = AxisCal(o, s)
        return out, ""
    except (OSError, ValueError, KeyError, TypeError) as e:
        return {k: None for k in SENSORS}, f"{CONFIG_NAME} com erro ({e}); sem calibração de acel/mag"


def save(cals: dict) -> str:
    path = config_path()
    data = {k: ({"offset": c.offset.tolist(), "scale": c.scale.tolist()} if c is not None else None)
            for k, c in cals.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path
