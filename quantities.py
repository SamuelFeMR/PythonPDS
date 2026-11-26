"""
quantities.py

Catálogo das grandezas que podem ser plotadas/salvas e a linha de histórico de cada amostra.
Para acrescentar uma grandeza: uma linha em QUANTITIES e o valor correspondente em row().
Valores que não existem naquele instante (ex.: sem ToF válido) entram como NaN (o gráfico
fica com um buraco, como no MATLAB).
"""

from dataclasses import dataclass

import numpy as np

NAN = float("nan")


@dataclass(frozen=True)
class Quantity:
    key: str
    label: str
    unit: str
    group: str


QUANTITIES = (
    Quantity("roll", "roll", "°", "Orientação"),
    Quantity("pitch", "pitch", "°", "Orientação"),
    Quantity("yaw", "yaw", "°", "Orientação"),
    Quantity("tilt", "inclinação (tilt)", "°", "Orientação"),
    Quantity("polar", "ângulo do polo", "°", "Posição na esfera"),
    Quantity("lon", "longitude (fundida)", "°", "Posição na esfera"),
    Quantity("lon_cor", "longitude (só cor)", "°", "Posição na esfera"),
    Quantity("delta", "Δ mundo→esfera", "°", "Posição na esfera"),
    Quantity("px", "x (centro da esfera)", "mm", "Posição na esfera"),
    Quantity("py", "y (centro da esfera)", "mm", "Posição na esfera"),
    Quantity("pz", "z (centro da esfera)", "mm", "Posição na esfera"),
    Quantity("alpha", "α (ângulo de ataque)", "°", "Sensores de distância"),
    Quantity("H", "H", "mm", "Sensores de distância"),
    Quantity("Y", "Y (corda)", "mm", "Sensores de distância"),
    Quantity("h1", "h1 (calibrado)", "mm", "Sensores de distância"),
    Quantity("h2", "h2 (calibrado)", "mm", "Sensores de distância"),
    Quantity("dist1", "dist1 (bruto)", "mm", "Sensores de distância"),
    Quantity("dist2", "dist2 (bruto)", "mm", "Sensores de distância"),
    Quantity("ax", "acc x", "g", "Acelerômetro"),
    Quantity("ay", "acc y", "g", "Acelerômetro"),
    Quantity("az", "acc z", "g", "Acelerômetro"),
    Quantity("anorm", "|acc|", "g", "Acelerômetro"),
    Quantity("gx", "gyro x", "°/s", "Giroscópio"),
    Quantity("gy", "gyro y", "°/s", "Giroscópio"),
    Quantity("gz", "gyro z", "°/s", "Giroscópio"),
    Quantity("mx", "mag x", "", "Magnetômetro"),
    Quantity("my", "mag y", "", "Magnetômetro"),
    Quantity("mz", "mag z", "", "Magnetômetro"),
    Quantity("red", "R", "", "Cor"),
    Quantity("green", "G", "", "Cor"),
    Quantity("blue", "B", "", "Cor"),
)

KEYS = tuple(q.key for q in QUANTITIES)
BY_KEY = {q.key: q for q in QUANTITIES}
GROUPS = tuple(dict.fromkeys(q.group for q in QUANTITIES))
N = len(QUANTITIES)


def row(s, pose, geom, pos) -> np.ndarray:
    """Uma linha do histórico, na ordem de QUANTITIES. Tudo da MESMA amostra `s`;
    pose, geom e pos podem ser None."""
    v = dict.fromkeys(KEYS, NAN)
    v.update(dist1=s.dist1, dist2=s.dist2, red=s.red, green=s.green, blue=s.blue)
    if s.imu_ok:
        v.update(ax=s.accel[0], ay=s.accel[1], az=s.accel[2],
                 gx=s.gyro[0], gy=s.gyro[1], gz=s.gyro[2],
                 mx=s.mag[0], my=s.mag[1], mz=s.mag[2])
    if pose is not None:
        v.update(roll=pose.euler_deg[0], pitch=pose.euler_deg[1], yaw=pose.euler_deg[2],
                 tilt=pose.tilt_deg, anorm=pose.accel_norm)
    if geom is not None and s.tof_ok:
        v.update(h1=geom.h1, h2=geom.h2)
        if geom.valid:
            v.update(alpha=geom.alpha_deg, H=geom.H, Y=geom.Y)
    if pos is not None:
        v.update(polar=pos.polar_deg, lon=pos.azimuth_deg, delta=pos.yaw_shift_deg)
        if np.isfinite(pos.radius_mm):
            v.update(zip(('px', 'py', 'pz'), pos.cartesian()))
        if pos.lon_color_deg is not None:
            v.update(lon_cor=pos.lon_color_deg)
    return np.array([float(v[k]) for k in KEYS])
