"""
sphere_color.py

Cor da esfera -> longitude, na convenção do gomos_r9.py (o script que imprime os gomos).

No gomos_r9.py a cor depende só da longitude (gradiente HSV horizontal):

        matiz = (lon / 360) · H_MAX      saturação = valor = 1      (calota do polo: vermelho)

e `posicao(h)` é a inversa. Aqui o sensor RGB faz o caminho de volta: lê (R, G, B), tira
o matiz e devolve a LONGITUDE do ponto em que está. A latitude (distância ao polo) vem da
orientação × α (position.py); juntas dão o ponto na esfera.

Parâmetros fixos em sphere_config.json (pasta de trabalho ou ao lado deste arquivo; criado com
os padrões se não existir):

    {
      "rgb_gain": [1.0, 1.0, 1.0],   # ganho por canal (equilíbrio de branco do sensor)
      "min_signal": 50,              # maior canal (já com ganho) abaixo disso = sem sinal
      "min_saturation": 0.3,         # cor lavada demais (ex.: papel branco) = rejeitada
      "hue_offset_deg": 0.0,         # soma ao matiz medido: corrige o desvio do sensor
      "pole_diameter_mm": 10.0,      # calota vermelha do polo (1 cm no gomos_r9.py)
      "fusion_rate": 0.5,            # 1/s: quão rápido a cor corrige a longitude da orientação
      "fusion_gate_deg": 60.0        # cor mais distante que isso da prevista é descartada
    }
"""

import colorsys
import json
import math
import os
import re
from dataclasses import dataclass

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_NAME = "sphere_config.json"

H_MAX = 1.0   # igual ao gomos_r9.py


@dataclass
class ColorConfig:
    rgb_gain: tuple = (1.0, 1.0, 1.0)
    min_signal: float = 50.0
    min_saturation: float = 0.3
    hue_offset_deg: float = 0.0
    pole_diameter_mm: float = 10.0
    fusion_rate: float = 0.5
    fusion_gate_deg: float = 60.0
    message: str = ""


def longitude_to_hue(lon_deg: float) -> float:
    """Matiz (0..1) da cor impressa na longitude dada (gomos_r9.cor)."""
    return (lon_deg % 360.0) / 360.0 * H_MAX


def hue_to_longitude(h: float) -> float:
    """Inversa: longitude em graus (0..360) de um matiz (gomos_r9.posicao)."""
    return (h / H_MAX) * 360.0 % 360.0


def read_longitude(r: float, g: float, b: float, cfg: ColorConfig):
    """(R, G, B) brutos do sensor -> (longitude em graus, "") ou (None, motivo)."""
    v = (r * cfg.rgb_gain[0], g * cfg.rgb_gain[1], b * cfg.rgb_gain[2])
    m = max(v)
    if m < cfg.min_signal:
        return None, "sinal de cor fraco"
    sat = (m - min(v)) / m
    if sat < cfg.min_saturation:
        return None, "cor pouco saturada"
    h = colorsys.rgb_to_hsv(v[0] / m, v[1] / m, v[2] / m)[0]
    return (hue_to_longitude(h) + cfg.hue_offset_deg) % 360.0, ""


def pole_cap_deg(R_mm: float, cfg: ColorConfig) -> float:
    """Raio angular da calota vermelha do polo (arco de raio diâmetro/2, como no gomos_r9.py)."""
    return math.degrees((cfg.pole_diameter_mm / 2.0) / R_mm) if R_mm > 0 else 0.0


# ---------------------------------------------------------------- arquivo
def config_path() -> str:
    for folder in (os.getcwd(), APP_DIR):
        path = os.path.join(folder, CONFIG_NAME)
        if os.path.exists(path):
            return path
    return os.path.join(APP_DIR, CONFIG_NAME)


def default_dict() -> dict:
    c = ColorConfig()
    return {"rgb_gain": list(c.rgb_gain), "min_signal": c.min_signal,
            "min_saturation": c.min_saturation, "hue_offset_deg": c.hue_offset_deg,
            "pole_diameter_mm": c.pole_diameter_mm, "fusion_rate": c.fusion_rate,
            "fusion_gate_deg": c.fusion_gate_deg}


def load() -> ColorConfig:
    path = config_path()
    if not os.path.exists(path):
        try:
            with open(path, "w", encoding="utf-8") as f:
                text = json.dumps(default_dict(), indent=2)
                f.write(re.sub(r"\[\s+([^\]]+?)\s+\]",
                               lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]", text))
        except OSError:
            pass
        return ColorConfig()
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        gain = tuple(float(x) for x in d.get("rgb_gain", (1, 1, 1)))
        if len(gain) != 3 or not all(math.isfinite(x) and x > 0 for x in gain):
            raise ValueError("rgb_gain precisa de 3 números positivos")
        cfg = ColorConfig(
            rgb_gain=gain,
            min_signal=float(d.get("min_signal", 50.0)),
            min_saturation=float(d.get("min_saturation", 0.3)),
            hue_offset_deg=float(d.get("hue_offset_deg", 0.0)),
            pole_diameter_mm=float(d.get("pole_diameter_mm", 10.0)),
            fusion_rate=float(d.get("fusion_rate", 0.5)),
            fusion_gate_deg=float(d.get("fusion_gate_deg", 60.0)))
        if not all(math.isfinite(x) for x in (cfg.min_signal, cfg.min_saturation, cfg.hue_offset_deg,
                                               cfg.pole_diameter_mm, cfg.fusion_rate,
                                               cfg.fusion_gate_deg)):
            raise ValueError("valor não numérico")
        return cfg
    except (OSError, ValueError, TypeError, AttributeError) as e:
        return ColorConfig(message=f"{CONFIG_NAME} com erro ({e}); usando padrões")
