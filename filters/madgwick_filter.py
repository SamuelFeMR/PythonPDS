"""
filters/madgwick_filter.py

Adaptadores do filtro de Madgwick (madgwick.py, que usa a biblioteca AHRS) para a
interface PoseFilter.

Dois filtros:
  - "Madgwick (mag só no yaw)": acelerômetro corrige roll/pitch; o magnetômetro
    corrige SÓ o yaw, devagar e proporcional ao erro. Estável mesmo com o
    magnetômetro imperfeito. É o recomendado.
  - "Madgwick MARG (original)": gradiente normalizado com acel+mag juntos,
    como no artigo. Com o sensor parado, qualquer discordância do magnetômetro
    puxa o yaw com a força total (beta); com mag mal calibrado o yaw "volta".
"""

import numpy as np

from madgwick import Madgwick

from .base import FilterInput, Param, PoseFilter


class _MadgwickBase(PoseFilter):
    mag_mode = "marg"
    # chave do parâmetro -> atributo do Madgwick
    _attrs = {"beta": "beta", "mag_rate": "mag_rate",
              "accel_tol": "accel_tol", "mag_tol": "mag_tol"}

    def __init__(self, **params):
        super().__init__(**params)
        self.core = Madgwick(mag_mode=self.mag_mode)
        for key in self.values:
            self.on_param_changed(key)

    def on_param_changed(self, key):
        attr = self._attrs.get(key)
        if attr:
            setattr(self.core, attr, self.values[key])

    def reset(self):
        self.core.q = None

    def update(self, inp: FilterInput) -> np.ndarray:
        return self.core.update(inp.gyro, inp.accel, inp.mag, inp.dt)


class MadgwickYawFilter(_MadgwickBase):
    name = "Madgwick (mag só no yaw)"
    description = ("Acelerômetro corrige roll/pitch. O magnetômetro corrige só o yaw, "
                   "devagar. Mag_rate = 0 desliga o magnetômetro; aumente só se o mag "
                   "estiver bem calibrado.")
    mag_mode = "yaw"
    params = (
        Param("beta", "beta (ganho acel)", 0.1, 0.0, 0.5, 0.005),
        Param("mag_rate", "mag_rate (1/s)", 0.02, 0.0, 0.3, 0.005),
        Param("accel_tol", "tolerância |a| (0 = off)", 0.25, 0.0, 0.5, 0.01),
        Param("mag_tol", "tolerância |m| (0 = off)", 0.3, 0.0, 0.5, 0.01),
    )


class MadgwickMargFilter(_MadgwickBase):
    name = "Madgwick MARG (original)"
    description = ("Madgwick MARG do artigo: gradiente normalizado com acel+mag. "
                   "Sensível a magnetômetro mal calibrado.")
    mag_mode = "marg"
    params = (
        Param("beta", "beta (ganho)", 0.1, 0.0, 0.5, 0.005),
        Param("accel_tol", "tolerância |a| (0 = off)", 0.25, 0.0, 0.5, 0.01),
        Param("mag_tol", "tolerância |m| (0 = off)", 0.3, 0.0, 0.5, 0.01),
    )
