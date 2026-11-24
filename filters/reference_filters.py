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

from .base import FilterInput, PoseFilter


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
