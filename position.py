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

O ponto na esfera é o ângulo do polo (polar) e o azimute da normal. O giro em torno da
vertical (yaw) não muda onde o sensor está na esfera, só para onde a placa aponta, e vem do
magnetômetro (que deriva); por isso ele é retirado (yaw=False o mantém). A esfera é só o
hemisfério de cima: se a normal apontar para baixo do equador, trava no equador.

LONGITUDE: orientação × α só dão a distância ao polo (polar); o azimute que sai deles vale só
para o lado em que a placa está inclinada. A longitude de verdade vem da COR da esfera
(sphere_color.py, gradiente do gomos_r9.py). with_longitude() junta as duas: polar da
orientação × α, longitude da cor.

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


def remove_yaw(R: np.ndarray) -> np.ndarray:
    """Rz(−yaw)·R: a mesma inclinação (roll/pitch), com a placa apontando para o norte."""
    yaw = math.atan2(R[1, 0], R[0, 0])
    c, s = math.cos(-yaw), math.sin(-yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ R


@dataclass
class Position:
    t_s: float
    normal: np.ndarray            # vetor unitário do centro da esfera ao ponto (mundo, Z para cima)
    polar_deg: float              # ângulo a partir do polo (0 = topo, 90 = equador)
    azimuth_deg: float            # direção do ponto (0 = +X/norte, 90 = +Y); 0 no polo
    alpha_deg: float              # α usado
    alpha_assumed: bool = False   # True: sem ToF válido, usou α = 0
    clamped: bool = False         # True: estava abaixo do equador
    lon_source: str = "imu"       # de onde veio o azimute: "cor", "cor retida" ou "imu"

    @property
    def height_ratio(self) -> float:
        """Altura do ponto como fração do raio (cos do ângulo polar)."""
        return math.cos(math.radians(self.polar_deg))


def compute(t_s: float, q, alpha_deg, bar_axis: str = DEFAULT_BAR,
            toward_axis: str = DEFAULT_TOWARD, yaw: bool = False) -> Position:
    """q = quaternion [w,x,y,z] placa -> mundo; alpha_deg = None se o ToF não é válido."""
    bar, toward = axis_vector(bar_axis), axis_vector(toward_axis)
    assumed = alpha_deg is None or not math.isfinite(alpha_deg)
    alpha = 0.0 if assumed else float(alpha_deg)
    R = Q.to_matrix(np.asarray(q, float))
    if not yaw:
        R = remove_yaw(R)
    n = world_normal(R, alpha, bar, toward)
    clamped = bool(n[2] < 0)
    if clamped:
        n = np.array([n[0], n[1], 0.0])
        norm = np.linalg.norm(n)
        n = n / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])
    polar = math.degrees(math.acos(max(-1.0, min(1.0, n[2]))))
    azim = math.degrees(math.atan2(n[1], n[0])) if polar > 1e-6 else 0.0
    return Position(t_s, n, polar, azim, alpha, assumed, clamped)


def with_longitude(pos: Position, lon_deg, source: str) -> Position:
    """Troca o azimute pela longitude lida na COR (None = mantém o do IMU), sem mexer no polar."""
    if lon_deg is None:
        return pos
    th, lam = math.radians(pos.polar_deg), math.radians(lon_deg)
    n = np.array([math.sin(th) * math.cos(lam), math.sin(th) * math.sin(lam), math.cos(th)])
    return Position(pos.t_s, n, pos.polar_deg, float(lon_deg) % 360.0, pos.alpha_deg,
                    pos.alpha_assumed, pos.clamped, source)
