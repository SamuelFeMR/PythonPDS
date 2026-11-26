"""
sample.py

Uma linha do ESP32 já interpretada.

Formato (18 campos):
  t_us, ax,ay,az, gx,gy,gz, mx,my,mz, dist1,dist2, red,green,blue,
  imu_ok,tof_ok,color_ok
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

N_FIELDS = 18


@dataclass
class Sample:
    t_us: int
    accel: np.ndarray      # como veio do firmware
    gyro: np.ndarray       # como veio do firmware (graus/s)
    mag: np.ndarray        # como veio do firmware
    dist1: int
    dist2: int
    red: int
    green: int
    blue: int
    imu_ok: bool
    tof_ok: bool
    color_ok: bool


def parse_sample(line: str) -> Optional[Sample]:
    """Converte uma linha CSV em Sample. None se a linha for inválida
    (logs do ESP-IDF, cabeçalhos, linhas cortadas...)."""
    p = line.strip().split(",")
    if len(p) != N_FIELDS:
        return None
    try:
        v = [float(x) for x in p[1:10]]
        return Sample(
            t_us=int(p[0]),
            accel=np.array(v[0:3]), gyro=np.array(v[3:6]), mag=np.array(v[6:9]),
            dist1=int(p[10]), dist2=int(p[11]),
            red=int(p[12]), green=int(p[13]), blue=int(p[14]),
            imu_ok=int(p[15]) != 0, tof_ok=int(p[16]) != 0,
            color_ok=int(p[17]) != 0,
        )
    except ValueError:
        return None
