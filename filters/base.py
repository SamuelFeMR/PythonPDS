"""
filters/base.py

Contrato que todo filtro de pose precisa seguir. A interface gráfica e o
pipeline só conhecem ESTA interface: para mexer num filtro (ou criar outro)
não é preciso tocar em mais nada.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Param:
    """Parâmetro do filtro. O valor vem de filter_config.json; o padrão é `default`."""
    key: str
    label: str
    default: float


@dataclass
class FilterInput:
    accel: np.ndarray   # referencial do acelerômetro, unidade qualquer (g ou m/s²)
    gyro: np.ndarray    # rad/s, já alinhado e SEM bias
    mag: np.ndarray     # já alinhado e calibrado; zeros se o mag estiver desligado
    dt: float           # s (0 na primeira amostra ou se o intervalo for inválido)


class PoseFilter(ABC):
    name = "Base"
    description = ""
    params: tuple = ()

    def __init__(self, **params):
        self.values = {p.key: float(params.get(p.key, p.default)) for p in self.params}

    def on_param_changed(self, key: str):
        pass

    @abstractmethod
    def reset(self):
        """Esquece a orientação atual (reinicializa na próxima amostra)."""

    @abstractmethod
    def update(self, inp: FilterInput) -> np.ndarray:
        """Retorna o quaternion [w, x, y, z] sensor -> navegação (Z para cima)."""
