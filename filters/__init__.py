"""
Registro de filtros.

Para criar um filtro novo:
  1) crie uma classe que herda de PoseFilter (ver base.py);
  2) adicione a classe na tupla abaixo.
Para usá-lo, ponha o nome dele em "filter" no filter_config.json.
"""

from .base import FilterInput, Param, PoseFilter
from .madgwick_filter import MadgwickMargFilter, MadgwickYawFilter
from .reference_filters import ComplementaryFilter, GyroOnlyFilter, TriadFilter

FILTERS = {cls.name: cls for cls in (
    MadgwickYawFilter, MadgwickMargFilter, ComplementaryFilter, TriadFilter, GyroOnlyFilter)}
DEFAULT_FILTER = MadgwickYawFilter.name

# Os 3 métodos de saída oferecidos na interface: (rótulo, nome do filtro)
MODES = (
    ("1 · Completo (Madgwick + acel + mag)", MadgwickYawFilter.name),
    ("2 · Complementar simples", ComplementaryFilter.name),
    ("3 · Só giroscópio (mostra a deriva)", GyroOnlyFilter.name),
)


def make_filter(name: str, **params) -> PoseFilter:
    return FILTERS[name](**params)
