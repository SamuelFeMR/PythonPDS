"""
Registro de filtros.

Para criar um filtro novo:
  1) crie uma classe que herda de PoseFilter (ver base.py);
  2) adicione a classe na tupla abaixo.
Ele aparece sozinho na lista da página de pose.
"""

from .base import FilterInput, Param, PoseFilter
from .madgwick_filter import MadgwickMargFilter, MadgwickYawFilter
from .reference_filters import GyroOnlyFilter, TriadFilter

FILTERS = {cls.name: cls for cls in (
    MadgwickYawFilter, MadgwickMargFilter, TriadFilter, GyroOnlyFilter)}
DEFAULT_FILTER = MadgwickYawFilter.name


def make_filter(name: str, **params) -> PoseFilter:
    return FILTERS[name](**params)
