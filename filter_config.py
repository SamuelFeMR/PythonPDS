"""
filter_config.py

Parâmetros FIXOS do filtro de pose, lidos de filter_config.json (na pasta de
trabalho ou ao lado deste arquivo). Edite o JSON com o programa fechado, ou
edite e clique em "Reiniciar orientação" para reler.

    {
      "filter": "Madgwick (mag só no yaw)",     # nome do filtro (ver filters/__init__.py)
      "use_mag": true,                          # false = ignora o magnetômetro
      "params": {"beta": 0.1, "mag_rate": 0.02, "accel_tol": 0.25, "mag_tol": 0.3}
    }

Parâmetros que faltam usam o padrão do filtro. Se o arquivo não existir, ele é
criado com os valores padrão. Se estiver com erro, usa os padrões e avisa.
"""

import json
import math
import os
from dataclasses import dataclass, field

from filters import DEFAULT_FILTER, FILTERS

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_NAME = "filter_config.json"


@dataclass
class FilterConfig:
    filter: str = DEFAULT_FILTER
    use_mag: bool = True
    params: dict = field(default_factory=dict)
    message: str = ""          # aviso para mostrar na interface ("" = tudo certo)


def config_path() -> str:
    for folder in (os.getcwd(), APP_DIR):
        path = os.path.join(folder, CONFIG_NAME)
        if os.path.exists(path):
            return path
    return os.path.join(APP_DIR, CONFIG_NAME)


def default_dict() -> dict:
    cls = FILTERS[DEFAULT_FILTER]
    return {"filter": DEFAULT_FILTER, "use_mag": True,
            "params": {p.key: p.default for p in cls.params}}


def load() -> FilterConfig:
    path = config_path()
    if not os.path.exists(path):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(default_dict(), f, indent=2, ensure_ascii=False)
        except OSError:
            pass
        return FilterConfig()
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        name = d.get("filter", DEFAULT_FILTER)
        if name not in FILTERS:
            raise ValueError(f"filtro '{name}' não existe (use: {', '.join(FILTERS)})")
        valid = {p.key for p in FILTERS[name].params}
        params, ignored = {}, []
        for k, v in (d.get("params") or {}).items():
            v = float(v)
            if k not in valid or not math.isfinite(v):
                ignored.append(k)
            else:
                params[k] = v
        msg = f"{CONFIG_NAME}: parâmetro ignorado: {', '.join(ignored)}" if ignored else ""
        return FilterConfig(name, bool(d.get("use_mag", True)), params, msg)
    except (OSError, ValueError, TypeError, AttributeError) as e:
        return FilterConfig(message=f"{CONFIG_NAME} com erro ({e}); usando padrões")
