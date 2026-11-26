"""
position.py

Cruza a ORIENTAÇÃO da placa (IMU) com o ÂNGULO DE ATAQUE α (ToF) para dizer em que ponto
da esfera o sensor está.

No referencial da placa, a normal da esfera no ponto medido (apontando da esfera para o
sensor) fica no plano da barra e dos sensores:

        normal_placa = sen(α)·barra − cos(α)·direção_dos_sensores

(α = 0: a superfície é paralela à barra, a normal é contrária à direção dos sensores.)
A orientação R (placa -> mundo, Z para cima) leva essa normal para o mundo:

        normal_mundo = R · normal_placa

Com a placa deitada (Z para cima), sensores olhando para baixo e α = 0 isso dá (0, 0, 1):
o polo. α ≠ 0, ou uma placa inclinada, desloca o ponto para o lado.

O ponto na esfera é o ângulo do polo (polar) e o azimute da normal. Nada é travado: se o
polar passar de 90° (abaixo do equador) é porque a orientação diz que os sensores olham para
cima, e isso é mostrado como aviso (normalmente é o eixo dos sensores errado na configuração).

LONGITUDE (fusão): o polar vem da orientação × α. O azimute da orientação responde na hora ao
LADO para onde a placa inclina, mas está no referencial do MUNDO (e deriva com o yaw). A COR da
esfera (sphere_color.py, gradiente do gomos_r9.py) dá a longitude no referencial da ESFERA,
sem deriva, mas só enquanto o sensor enxerga a cor. A diferença entre as duas é um giro em torno
da vertical, Δ (yaw_shift_deg), que LongitudeFusion estima devagar (filtro complementar):

        longitude = azimute_do_IMU + Δ        Δ ← Δ + taxa·dt·(cor − (azimute_do_IMU + Δ))

Assim a orientação manda no curto prazo (inverter a inclinação leva o ponto para o outro lado
da esfera) e a cor corrige a deriva. Cor muito diferente da prevista (mais que `gate`) é
descartada como discordante. Δ gira a placa inteira junto com o ponto, então a placa nunca
fica olhando para outro lado que não o ponto.

POSE COMPLETA (scene_points): o ponto medido h fica na superfície; a barra fica a H dele,
no lado dos sensores; os raios dos dois sensores terminam exatamente em c e d, na superfície.

Sem lógica de interface: testado por test_position.py.
"""

import math
from dataclasses import dataclass

import numpy as np

import quaternion as Q

DEFAULT_BAR = "+X"
DEFAULT_TOWARD = "-Z"


def axis_vector(name: str) -> np.ndarray:
    """'+X', '-Z', ... -> vetor unitário no referencial da placa."""
    v = np.zeros(3)
    v["XYZ".index(name.strip("+-").upper())] = -1.0 if name.startswith("-") else 1.0
    return v


def board_normal(alpha_deg: float, bar: np.ndarray, toward: np.ndarray) -> np.ndarray:
    """Normal da esfera no ponto medido, no referencial da PLACA."""
    a = math.radians(alpha_deg)
    return math.sin(a) * bar - math.cos(a) * toward


def world_normal(R: np.ndarray, alpha_deg: float, bar: np.ndarray, toward: np.ndarray) -> np.ndarray:
    """Normal da esfera no ponto medido, no MUNDO (com yaw)."""
    return R @ board_normal(alpha_deg, bar, toward)


def rotate_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class Position:
    t_s: float
    normal: np.ndarray            # vetor unitário do centro da esfera ao ponto (Z para cima)
    polar_deg: float              # ângulo a partir do polo (0 = topo, 90 = equador, 180 = baixo)
    azimuth_deg: float            # direção do ponto (0 = +X/norte, 90 = +Y); 0 no polo
    alpha_deg: float              # α usado
    alpha_assumed: bool = False   # True: sem ToF válido, usou α = 0
    lon_source: str = "imu"       # origem do azimute: "fusão", "discordante" (cor descartada) ou "imu"
    yaw_shift_deg: float = 0.0    # giro (em torno de Z) que leva o mundo ao referencial da esfera
    lon_color_deg: object = None  # longitude lida na cor neste instante (None = sem cor válida)
    radius_mm: float = float('nan')   # raio da esfera (distância do centro ao ponto)

    @property
    def below_equator(self) -> bool:
        return self.polar_deg > 90.0

    def cartesian(self):
        """(x, y, z) em mm com origem no centro da esfera; Z para cima."""
        th, lam = math.radians(self.polar_deg), math.radians(self.azimuth_deg)
        r = self.radius_mm
        return (r * math.sin(th) * math.cos(lam), r * math.sin(th) * math.sin(lam), r * math.cos(th))

    @property
    def height_ratio(self) -> float:
        """Altura do ponto como fração do raio (cos do ângulo polar)."""
        return math.cos(math.radians(self.polar_deg))

    def display_rotation(self, R: np.ndarray) -> np.ndarray:
        """Orientação da placa no referencial da esfera (a real, girada de yaw_shift)."""
        return rotate_z(self.yaw_shift_deg) @ R


def compute(t_s: float, q, alpha_deg, bar_axis: str = DEFAULT_BAR,
            toward_axis: str = DEFAULT_TOWARD) -> Position:
    """q = quaternion [w,x,y,z] placa -> mundo; alpha_deg = None se o ToF não é válido."""
    bar, toward = axis_vector(bar_axis), axis_vector(toward_axis)
    assumed = alpha_deg is None or not math.isfinite(alpha_deg)
    alpha = 0.0 if assumed else float(alpha_deg)
    n = world_normal(Q.to_matrix(np.asarray(q, float)), alpha, bar, toward)
    n = n / np.linalg.norm(n)
    polar = math.degrees(math.acos(max(-1.0, min(1.0, n[2]))))
    azim = math.degrees(math.atan2(n[1], n[0])) if polar > 1e-6 and polar < 180 - 1e-6 else 0.0
    return Position(t_s, n, polar, azim, alpha, assumed)


def wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def with_longitude(pos: Position, lon_deg, source: str, lon_color=None) -> Position:
    """Põe o azimute na longitude da ESFERA (None = mantém o do IMU), sem mexer no polar.
    O giro necessário (yaw_shift_deg) fica guardado para girar a placa junto."""
    if lon_deg is None:
        return pos
    th, lam = math.radians(pos.polar_deg), math.radians(lon_deg)
    n = np.array([math.sin(th) * math.cos(lam), math.sin(th) * math.sin(lam), math.cos(th)])
    shift = 0.0 if 1e-6 > pos.polar_deg or pos.polar_deg > 180 - 1e-6 else (lon_deg - pos.azimuth_deg)
    return Position(pos.t_s, n, pos.polar_deg, float(lon_deg) % 360.0, pos.alpha_deg,
                    pos.alpha_assumed, source, wrap180(shift), lon_color)


class LongitudeFusion:
    """Filtro complementar da longitude: azimute do IMU + Δ, com Δ corrigido devagar pela cor."""

    def __init__(self, rate: float = 0.5, gate_deg: float = 60.0, min_polar_deg: float = 5.0):
        self.rate, self.gate, self.min_polar = rate, gate_deg, min_polar_deg
        self.reset()

    def reset(self):
        self.delta = None       # Δ ainda não conhecido: usa o azimute do mundo
        self._t = None

    def update(self, t_s: float, az_imu: float, polar: float, lon_color, min_polar=None):
        """-> (longitude da esfera em graus, origem, Δ). Origem: "fusão" (cor aceita),
        "discordante" (cor descartada), "imu" (sem cor válida; Δ vem da última fusão)."""
        dt = 0.0 if self._t is None else t_s - self._t
        dt = dt if 0.0 < dt < 0.5 else (0.0 if self._t is None else 0.02)
        self._t = t_s
        usable = lon_color is not None and polar >= (self.min_polar if min_polar is None else min_polar)
        if self.delta is None:
            if not usable:
                return az_imu % 360.0, "imu", 0.0
            self.delta = wrap180(lon_color - az_imu)         # primeira leitura de cor: alinha
            return lon_color % 360.0, "fusão", self.delta
        if not usable:
            return (az_imu + self.delta) % 360.0, "imu", self.delta
        err = wrap180(lon_color - (az_imu + self.delta))
        if abs(err) > self.gate:
            return (az_imu + self.delta) % 360.0, "discordante", self.delta
        self.delta = wrap180(self.delta + min(1.0, self.rate * dt) * err)
        return (az_imu + self.delta) % 360.0, "fusão", self.delta


def scene_points(res, L: float, R_mm: float, Rd: np.ndarray, bar: np.ndarray, toward: np.ndarray,
                 normal: np.ndarray, radius: float) -> dict:
    """
    Pose completa na cena: esfera de raio `radius` no centro (0,0,0).

    res = geometry.GeometryResult válido; Rd = orientação da placa no referencial da esfera
    (Position.display_rotation); normal = Position.normal. As distâncias do ToF (mm) são
    convertidas para a cena por k = radius / R_mm.

    Devolve (todos em coordenadas da cena): h (ponto medido, na superfície), c e d (onde
    os raios tocam a esfera), s1 e s2 (sensores), m (meio da barra), foot (pé da perpendicular
    de h sobre a barra; foot→h é o H).
    """
    k = radius / R_mm
    bw, tw = Rd @ bar, Rd @ toward
    h = normal * radius
    lateral = bw * (res.h[0] - L / 2.0) * k
    m = h - lateral - tw * res.h[1] * k
    s1 = m - bw * (L / 2.0) * k
    return dict(h=h, m=m, foot=m + lateral, s1=s1, s2=m + bw * (L / 2.0) * k,
                c=s1 + (bw * res.c[0] + tw * res.c[1]) * k,
                d=s1 + (bw * res.d[0] + tw * res.d[1]) * k)
