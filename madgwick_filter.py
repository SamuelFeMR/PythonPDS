"""
filters/madgwick_filter.py

Adaptador do filtro de Madgwick (madgwick.py) para a interface PoseFilter.
Se o filtro estiver errado, o conserto é em madgwick.py / quaternion.py.
"""

import numpy as np

from madgwick import Madgwick

from .base import FilterInput, Param, PoseFilter


class MadgwickFilter(PoseFilter):
    name = "Madgwick"
    description = ("Giroscópio + correção por descida de gradiente com acelerômetro "
                   "e magnetômetro.")
    params = (Param("beta", "beta (ganho)", 0.1, 0.0, 1.0, 0.005),)

    def __init__(self, **params):
        super().__init__(**params)
        self.core = Madgwick(beta=self.values["beta"])

    def on_param_changed(self, key):
        if key == "beta":
            self.core.beta = self.values["beta"]

    def reset(self):
        self.core.q = None

    def update(self, inp: FilterInput) -> np.ndarray:
        return self.core.update(inp.gyro, inp.accel, inp.mag, inp.dt)
