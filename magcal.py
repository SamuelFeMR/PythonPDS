"""
magcal.py

Calibração do magnetômetro por elipsoide (esferoide):

  hard iron  = deslocamento fixo do centro (ferro magnetizado junto do sensor)    -> centro c
  soft iron  = deformação da esfera em elipsoide (materiais que distorcem o campo) -> matriz W

Girando o sensor em todas as direções, o campo terrestre (módulo constante) desenha uma
esfera. Com hard/soft iron ela vira um elipsoide fora do centro:  (m - c)' Q (m - c) = 1.
Ajustado o elipsoide, a correção que o devolve a uma esfera de raio r é

    m_cal = W @ (m - c),      W = r * Q^(1/2),      r = média geométrica dos semi-eixos

Métodos: "ellipsoid" (hard + soft iron, 3x3 completa) e "sphere" (só hard iron, W = I).
Resultado salvo em mag_calibration.json. Sem interface: serve para scripts e testes.
"""

import json
import os
from dataclasses import dataclass

import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_NAME = "mag_calibration.json"
METHODS = ("ellipsoid", "sphere")
MIN_POINTS = 50

# 26 direções (faces, arestas e cantos do cubo) para medir a cobertura da esfera
_DIRS = np.array([(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)],
                 dtype=float)
_DIRS /= np.linalg.norm(_DIRS, axis=1, keepdims=True)


@dataclass
class MagCal:
    center: np.ndarray          # hard iron (mesma unidade do magnetômetro)
    W: np.ndarray               # soft iron: matriz 3x3 (já inclui o raio)
    radius: float               # raio da esfera depois da correção
    method: str = "ellipsoid"
    samples: int = 0
    std_before: float = float("nan")   # desvio relativo de |m| antes da correção
    std_after: float = float("nan")    # ... e depois
    coverage: int = 0                  # de 26 direções cobertas

    def apply(self, m) -> np.ndarray:
        return self.W @ (np.asarray(m, float) - self.center)

    @property
    def is_identity(self) -> bool:
        return not np.any(self.center) and np.allclose(self.W, np.eye(3))


def _rel_std(r) -> float:
    return float(np.std(r) / np.mean(r))


def coverage(points) -> int:
    """Quantas das 26 direções ao redor do centro estimado têm pelo menos 1 ponto (0..26)."""
    p = np.asarray(points, float)
    if len(p) < 10:
        return 0
    c = 0.5 * (p.max(axis=0) + p.min(axis=0))
    d = p - c
    n = np.linalg.norm(d, axis=1)
    d = d[n > 1e-9] / n[n > 1e-9, None]
    if not len(d):
        return 0
    # só conta a direção se o ponto estiver realmente nela (ângulo < ~30°)
    hit = (d @ _DIRS.T) > np.cos(np.radians(30.0))
    return int(hit.any(axis=0).sum())


def thin(points, cells: int = 30) -> np.ndarray:
    """Um ponto por célula de uma grade 3D (média): tirar o peso de trechos em que o sensor ficou parado."""
    p = np.asarray(points, float)
    if len(p) == 0:
        return p
    span = p.max(axis=0) - p.min(axis=0)
    size = max(float(span.max()), 1e-9) / cells
    idx = np.floor((p - p.min(axis=0)) / size).astype(np.int64)
    _, inv = np.unique(idx, axis=0, return_inverse=True)
    inv = inv.ravel()
    out = np.zeros((inv.max() + 1, 3))
    np.add.at(out, inv, p)
    return out / np.bincount(inv)[:, None]


def _fit_ellipsoid(p):
    mu = p.mean(axis=0)
    s = float(np.mean(np.linalg.norm(p - mu, axis=1)))
    if s < 1e-9:
        raise ValueError("o campo não variou: gire o sensor")
    x, y, z = ((p - mu) / s).T
    D = np.column_stack([x * x, y * y, z * z, 2 * x * y, 2 * x * z, 2 * y * z, 2 * x, 2 * y, 2 * z])
    v, *_ = np.linalg.lstsq(D, np.ones(len(p)), rcond=None)
    A = np.array([[v[0], v[3], v[4]], [v[3], v[1], v[5]], [v[4], v[5], v[2]]])
    g = v[6:9]
    c_n = -np.linalg.solve(A, g)
    k = 1.0 + c_n @ A @ c_n
    Q = A / k
    w, U = np.linalg.eigh(0.5 * (Q + Q.T))
    if k <= 0 or w.min() <= 1e-2 * w.max():     # degenerado: dados quase planos ou sem cobertura
        raise ValueError("ajuste não formou um elipsoide: faltou cobertura, gire em todas as direções")
    center = mu + s * c_n
    w = w / s ** 2                       # volta à unidade original
    radius = float(np.prod(w) ** (-1.0 / 6.0))
    W = radius * (U * np.sqrt(w)) @ U.T  # r * Q^(1/2)
    return center, W, radius


def _fit_sphere(p):
    A = np.column_stack([2 * p, np.ones(len(p))])
    b = np.sum(p * p, axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    center = sol[:3]
    radius = float(np.mean(np.linalg.norm(p - center, axis=1)))
    return center, np.eye(3), radius      # W = I (só hard iron)


def fit(points, method: str = "ellipsoid") -> MagCal:
    """points = (n, 3) leituras brutas do magnetômetro girando em todas as direções."""
    if method not in METHODS:
        raise ValueError(f"método '{method}' (use {', '.join(METHODS)})")
    p = np.asarray(points, float)
    p = p[np.all(np.isfinite(p), axis=1)]
    if len(p) < MIN_POINTS:
        raise ValueError(f"poucos pontos ({len(p)}; mínimo {MIN_POINTS}): gire mais o sensor")
    center, W, radius = (_fit_ellipsoid if method == "ellipsoid" else _fit_sphere)(p)
    cal = MagCal(center, W, radius, method, len(p), coverage=coverage(p))
    cal.std_before = _rel_std(np.linalg.norm(p, axis=1))
    cal.std_after = _rel_std(np.linalg.norm((p - center) @ W.T, axis=1))
    return cal


# ----------------------------------------------------------------- arquivo
def config_path() -> str:
    for folder in (os.getcwd(), APP_DIR):
        path = os.path.join(folder, CONFIG_NAME)
        if os.path.exists(path):
            return path
    return os.path.join(APP_DIR, CONFIG_NAME)


def save(cal: MagCal) -> str:
    path = config_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"method": cal.method, "center": cal.center.tolist(), "matrix": cal.W.tolist(),
                   "radius": cal.radius, "samples": cal.samples, "coverage": cal.coverage,
                   "std_before": cal.std_before, "std_after": cal.std_after}, f, indent=2)
    return path


def load():
    """(MagCal ou None, mensagem de erro ou "")."""
    path = config_path()
    if not os.path.exists(path):
        return None, ""
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        center, W = np.array(d["center"], float), np.array(d["matrix"], float)
        if center.shape != (3,) or W.shape != (3, 3) or not np.all(np.isfinite(W)):
            raise ValueError("formato")
        return MagCal(center, W, float(d.get("radius", 1.0)), d.get("method", "ellipsoid"),
                      int(d.get("samples", 0)), float(d.get("std_before", float("nan"))),
                      float(d.get("std_after", float("nan"))), int(d.get("coverage", 0))), ""
    except (OSError, ValueError, KeyError, TypeError) as e:
        return None, f"{CONFIG_NAME} com erro ({e}); sem calibração do mag"


def delete() -> None:
    path = config_path()
    if os.path.exists(path):
        os.remove(path)
