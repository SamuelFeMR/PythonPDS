"""
filters/reference_filters.py

Filtros de referência, úteis para DIAGNÓSTICO: comparar com o Madgwick mostra
rapidamente se o problema está nos sensores/eixos ou no filtro.

- Acel+Mag (TRIAD): usa só acelerômetro e magnetômetro, sem giroscópio.
  Ruidoso, mas sem deriva. Se ele já está errado, o problema são os eixos ou a
  calibração do mag, não o filtro.
- Giroscópio puro: só integra o giroscópio. Mostra a deriva e confirma se os
  eixos/sinais/escala do gyro estão certos (gire 90° e veja se dá 90°).
"""

import numpy as np

import quaternion as Q
from madgwick import Madgwick

from .base import FilterInput, Param, PoseFilter


class TriadFilter(PoseFilter):
    name = "Acel+Mag (TRIAD)"
    description = "Sem giroscópio: orientação direta de acelerômetro e magnetômetro."

    def __init__(self, **params):
        super().__init__(**params)
        self._core = Madgwick(beta=0.0)

    def reset(self):
        pass

    def update(self, inp: FilterInput) -> np.ndarray:
        return self._core.initialize(inp.accel, inp.mag)


class GyroOnlyFilter(PoseFilter):
    name = "Giroscópio puro"
    description = "Só integra o giroscópio (começa da orientação TRIAD). Deriva com o tempo."

    def __init__(self, **params):
        super().__init__(**params)
        self.q = None

    def reset(self):
        self.q = None

    def update(self, inp: FilterInput) -> np.ndarray:
        if self.q is None:
            self.q = Madgwick(beta=0.0).initialize(inp.accel, inp.mag)
            return self.q.copy()
        q_dot = 0.5 * Q.multiply(self.q, [0.0, *inp.gyro])
        self.q = Q.normalize(self.q + q_dot * inp.dt)
        return self.q.copy()


class ComplementaryFilter(PoseFilter):
    name = "Complementar simples"
    description = ("Integra o giroscópio e puxa devagar a vertical para o acelerômetro "
                   "(roll/pitch) e o yaw para o magnetômetro. Ganho fixo, sem gradiente.")
    params = (
        Param("gain", "ganho acel (1/s)", 1.0),
        Param("mag_rate", "mag_rate (1/s)", 0.02),
        Param("accel_tol", "tolerância |a| (0 = off)", 0.25),
    )

    def __init__(self, **params):
        super().__init__(**params)
        self.q = None

    def reset(self):
        self.q = None

    def _accel_ok(self, a_norm):
        if a_norm < 1e-9:
            return False
        tol = self.values["accel_tol"]
        if tol <= 0:
            return True
        ref = 1.0 if a_norm < 4.0 else 9.80665      # g ou m/s²
        return abs(a_norm / ref - 1.0) <= tol

    def update(self, inp: FilterInput) -> np.ndarray:
        if self.q is None:
            self.q = Madgwick(beta=0.0).initialize(inp.accel, inp.mag)
            return self.q.copy()
        q, dt = self.q, inp.dt
        w = np.asarray(inp.gyro, dtype=float).copy()

        # 1) vertical: gira o corpo para alinhar a "vertical prevista" com o acelerômetro
        a_norm = np.linalg.norm(inp.accel)
        if dt > 0 and self._accel_ok(a_norm):
            up_pred = Q.rotate_inverse(q, [0.0, 0.0, 1.0])
            w += self.values["gain"] * np.cross(inp.accel / a_norm, up_pred)
        q = Q.normalize(q + 0.5 * Q.multiply(q, [0.0, *w]) * dt)

        # 2) yaw: gira em torno de Z (mundo) para levar o campo horizontal do mag a +X (norte)
        m_norm = np.linalg.norm(inp.mag)
        rate = self.values["mag_rate"]
        if dt > 0 and rate > 0 and m_norm > 1e-9:
            h = Q.rotate(q, inp.mag / m_norm)
            if np.hypot(h[0], h[1]) > 1e-3:
                th = -min(1.0, rate * dt) * np.arctan2(h[1], h[0])
                q = Q.normalize(Q.multiply([np.cos(th / 2), 0.0, 0.0, np.sin(th / 2)], q))
        self.q = q
        return q.copy()
