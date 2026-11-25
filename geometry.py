r"""
geometry.py

Geometria dos dois sensores de distância (ToF) nas pontas de uma barra.

                a ─────────────── b          a = sensor 1 (ESQUERDA)  -> mede h1
                 \ 60°        60° /          b = sensor 2 (DIREITA)   -> mede h2
              h1  \            /  h2         L = distância a–b (comprimento da barra)
                   c ·      · d              c, d = pontos onde os raios tocam o
                      \    /                       cilindro (raio R)
                       \  /
                        P                    P = onde os raios se cruzariam
                                             (triângulo a–b–P equilátero, lado L)

Eixos: x ao longo da barra (de a para b) e z perpendicular a ela, apontando para
o cilindro. Os raios saem a 60° da barra.

Saídas (todas no mesmo sistema de unidades das leituras, normalmente mm):

  alpha_deg  Ângulo entre a barra e a tangente ao cilindro no ponto h (h = ponto
             médio do arco c–d). É o mesmo ângulo da corda c–d com a barra.
             > 0 quando o sensor da DIREITA (h2) mede MAIS que o da esquerda.

                       alpha = arctan( √3 (h2 − h1) / (2L − h1 − h2) )

  H          Distância da barra até o ponto h, medida perpendicularmente à barra.

                       Y = |cd|            (corda)
                       F = R − √(R² − (Y/2)²)   (flecha do arco)
                       H = (√3/4)(h1 + h2) − F·cos(alpha)

Relação com as contas do TCC (PDF "TCC.pdf"):
  * A altura do triângulo equilátero é L·√3/2 (e não L). Com isso,
        H = L·√3/2 − f'·sen(β + ∂)/sen 30°
    vale para β = β1 e para β = β2 no caso simétrico, e fora dele a MÉDIA dos
    dois valores é igual à forma fechada acima (test_geometry.py confere isso).
  * A lei dos senos do triângulo c–d–P usa seno, não cosseno:
        (L − h2)/sen β1 = (L − h1)/sen β2 = Y/sen 60°.
"""

import json
import math
import os
from dataclasses import dataclass, field

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_NAME = "geometry_config.json"

S3 = math.sqrt(3.0)
COS60 = 0.5
SIN60 = S3 / 2.0

# Quando a corda c–d se aproxima do diâmetro (Y -> 2R), a flecha F varia muito com
# pequenas mudanças de Y e o centro pode estar de qualquer lado da corda: H perde
# precisão (alpha não depende de R e continua exato).
WARN_CHORD_RATIO = 0.95


# ---------------------------------------------------------------- configuração
AXES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")


def axes_ok(bar_axis: str, toward_axis: str) -> bool:
    """Os dois eixos precisam existir e ser de eixos DIFERENTES da placa."""
    return (bar_axis in AXES and toward_axis in AXES
            and bar_axis[-1] != toward_axis[-1])


@dataclass
class GeometryConfig:
    L: float = 150.0      # distância entre os sensores (barra a–b)
    R: float = 60.0       # raio do cilindro/esfera medido
    saved: bool = False   # False = valores de exemplo (ainda não salvos pelo usuário)
    # Eixos da PLACA (referencial do IMU) usados só para DESENHAR a esfera na página 1:
    bar_axis: str = "+X"      # sentido do sensor 1 (esquerda) para o sensor 2 (direita)
    toward_axis: str = "+Y"   # sentido da barra para o cilindro (para onde os sensores olham)


def config_path():
    """geometry_config.json na pasta de trabalho ou ao lado deste arquivo."""
    for folder in (os.getcwd(), APP_DIR):
        path = os.path.join(folder, CONFIG_NAME)
        if os.path.exists(path):
            return path
    return os.path.join(APP_DIR, CONFIG_NAME)


def load_config() -> GeometryConfig:
    path = config_path()
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        cfg = GeometryConfig(L=float(d["L"]), R=float(d["R"]), saved=True)
        ba, ta = d.get("bar_axis", cfg.bar_axis), d.get("toward_axis", cfg.toward_axis)
        if axes_ok(ba, ta):
            cfg.bar_axis, cfg.toward_axis = ba, ta
        return cfg
    except (OSError, ValueError, KeyError, TypeError):
        return GeometryConfig()


def save_config(cfg: GeometryConfig) -> str:
    path = config_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"L": cfg.L, "R": cfg.R, "bar_axis": cfg.bar_axis,
                   "toward_axis": cfg.toward_axis}, f, indent=2)
    return path


# ------------------------------------------------------------------- resultado
@dataclass
class GeometryResult:
    valid: bool = False
    reason: str = ""            # por que não é válido
    warning: str = ""           # válido, mas pouco confiável
    h1: float = math.nan
    h2: float = math.nan
    alpha_deg: float = math.nan
    H: float = math.nan
    Y: float = math.nan            # corda c–d
    F: float = math.nan            # flecha do arco c–d
    # pontos no referencial (x ao longo da barra, z em direção ao cilindro)
    c: tuple = field(default=None)        # onde o raio 1 toca o cilindro
    d: tuple = field(default=None)        # onde o raio 2 toca o cilindro
    h: tuple = field(default=None)        # ponto médio do arco c–d
    center: tuple = field(default=None)   # centro do cilindro


def compute(h1: float, h2: float, L: float, R: float) -> GeometryResult:
    """h1 = sensor da esquerda, h2 = sensor da direita (mesma unidade de L e R)."""
    res = GeometryResult(h1=float(h1), h2=float(h2))

    if not (math.isfinite(h1) and math.isfinite(h2)) or h1 <= 0 or h2 <= 0:
        res.reason = "leitura inválida (h <= 0)"
        return res
    if L <= 0 or R <= 0:
        res.reason = "L e R precisam ser positivos"
        return res

    dx = L - (h1 + h2) / 2.0                 # x_d − x_c
    if dx <= 0:
        res.reason = "h1 + h2 >= 2L: os raios se cruzam antes de tocar o cilindro"
        return res
    dz = SIN60 * (h2 - h1)                   # z_d − z_c
    Y = math.hypot(dx, dz)
    if Y >= 2 * R:
        res.reason = "corda maior que o diâmetro: R pequeno demais ou leituras inconsistentes"
        return res

    alpha = math.atan2(dz, dx)
    F = R - math.sqrt(R * R - (Y / 2.0) ** 2)

    c = (COS60 * h1, SIN60 * h1)
    d = (L - COS60 * h2, SIN60 * h2)
    mid = ((c[0] + d[0]) / 2.0, (c[1] + d[1]) / 2.0)
    n = (-math.sin(alpha), math.cos(alpha))  # normal à corda, para longe da barra
    hpt = (mid[0] - n[0] * F, mid[1] - n[1] * F)
    k = math.sqrt(R * R - (Y / 2.0) ** 2)
    center = (mid[0] + n[0] * k, mid[1] + n[1] * k)

    if Y > WARN_CHORD_RATIO * 2 * R:
        res.warning = "corda ≈ diâmetro: H pouco confiável (confira R)"

    res.valid = True
    res.alpha_deg = math.degrees(alpha)
    res.H = hpt[1]                           # = (√3/4)(h1+h2) − F·cos(alpha)
    res.Y, res.F = Y, F
    res.c, res.d, res.h, res.center = c, d, hpt, center
    return res
