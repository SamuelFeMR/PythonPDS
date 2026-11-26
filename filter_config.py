"""
filter_config.py

Parâmetros FIXOS do filtro de pose, lidos de filter_config.json (na pasta de trabalho ou ao
lado deste arquivo). Normalmente você ajusta pela interface ("Parâmetros…") e clica em
Salvar; também dá para editar o JSON com o programa fechado.

    {
      "filter": "Madgwick (mag só no yaw)",     # filtro usado ao abrir (ver filters/__init__.py)
      "use_mag": true,                          # false = ignora o magnetômetro
      "params": {                               # parâmetros de cada filtro
        "Madgwick (mag só no yaw)": {"beta": 0.07, "mag_rate": 0.02},
        "Complementar simples": {"gain": 1.0}
      }
    }

Também vale o formato antigo, com "params" plano (só do filtro escolhido). Parâmetros que
faltam usam o padrão do filtro. Se o arquivo não existir, ele é criado com os valores
padrão. Se estiver com erro, usa os padrões e avisa.
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
    params: dict = field(default_factory=dict)       # do filtro escolhido
    all_params: dict = field(default_factory=dict)   # {nome do filtro: parâmetros}
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
            "params": {DEFAULT_FILTER: {p.key: p.default for p in cls.params}}}


def _clean(name: str, raw: dict, ignored: list) -> dict:
    valid = {p.key for p in FILTERS[name].params}
    out = {}
    for k, v in raw.items():
        v = float(v)
        if k not in valid or not math.isfinite(v):
            ignored.append(k)
        else:
            out[k] = v
    return out


def _read(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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
        d = _read(path)
        name = d.get("filter", DEFAULT_FILTER)
        if name not in FILTERS:
            raise ValueError(f"filtro '{name}' não existe (use: {', '.join(FILTERS)})")
        raw = d.get("params") or {}
        nested = raw if raw and all(isinstance(v, dict) for v in raw.values()) else {name: raw}
        ignored, all_params = [], {}
        for fname, p in nested.items():
            if fname in FILTERS:
                all_params[fname] = _clean(fname, p, ignored)
            else:
                ignored.append(fname)
        msg = f"{CONFIG_NAME}: ignorado: {', '.join(ignored)}" if ignored else ""
        return FilterConfig(name, bool(d.get("use_mag", True)), all_params.get(name, {}), all_params, msg)
    except (OSError, ValueError, TypeError, AttributeError) as e:
        return FilterConfig(message=f"{CONFIG_NAME} com erro ({e}); usando padrões")


def save(filter_name: str, use_mag: bool, values: dict) -> str:
    """Grava o filtro atual (e seus parâmetros) como padrão; mantém os parâmetros dos outros filtros."""
    cfg = load()
    all_params = dict(cfg.all_params)
    all_params[filter_name] = {k: float(v) for k, v in values.items()}
    path = config_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"filter": filter_name, "use_mag": bool(use_mag), "params": all_params},
                  f, indent=2, ensure_ascii=False)
    return path
