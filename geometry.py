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
    # Eixos da PLACA (referencial do IMU), para posicionar a medida no mundo:
    bar_axis: str = "+X"      # sentido do sensor 1 (esquerda) para o sensor 2 (direita)
    toward_axis: str = "-Z"   # sentido para onde os sensores apontam (-Z = para baixo, placa deitada)
    # Calibração do ToF: valor somado a CADA leitura (h1 e h2), na mesma unidade delas.
    # Calculados por true_distance() a partir de uma altura H conhecida.
    tof_offset1: float = 0.0
    tof_offset2: float = 0.0


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
        old = float(d.get("tof_offset", 0.0))      # arquivo antigo: um offset só para os dois
        o1, o2 = float(d.get("tof_offset1", old)), float(d.get("tof_offset2", old))
        cfg.tof_offset1 = o1 if math.isfinite(o1) else 0.0
        cfg.tof_offset2 = o2 if math.isfinite(o2) else 0.0
        return cfg
    except (OSError, ValueError, KeyError, TypeError):
        return GeometryConfig()


def save_config(cfg: GeometryConfig) -> str:
    path = config_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"L": cfg.L, "R": cfg.R, "bar_axis": cfg.bar_axis,
                   "toward_axis": cfg.toward_axis, "tof_offset1": cfg.tof_offset1,
                   "tof_offset2": cfg.tof_offset2}, f, indent=2)
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


# ------------------------------------------------------------------ calibração
def true_distance(H_known: float, L: float, R: float):
    """
    Calibração do ToF: qual leitura os DOIS sensores deveriam ter, se lessem o mesmo valor
    (α = 0, objeto centrado), quando a altura é H_known.

    Com h1 = h2 = h:  H = (√3/2)·h − F,  F = R − √(R² − ((L − h)/2)²),  e H cresce com h,
    então a solução é única. Cada sensor recebe o próprio offset: h − leitura_i.

    Retorna (h, "") ou (None, motivo).
    """
    if not (math.isfinite(H_known) and H_known > 0):
        return None, "a altura conhecida precisa ser positiva"
    lo = max(0.0, L - 2.0 * R) + 1e-6      # corda (L − h) menor que o diâmetro
    hi = L - 1e-6                          # raios quase se encontrando

    def H_at(h):
        r = compute(h, h, L, R)
        return r.H if r.valid else math.nan

    H_lo, H_hi = H_at(lo), H_at(hi)
    if math.isnan(H_lo) or math.isnan(H_hi):
        return None, "L e R não permitem essa calibração"
    if not (H_lo <= H_known <= H_hi):
        return None, f"para esse L e R, H precisa estar entre {H_lo:.1f} e {H_hi:.1f}"
    for _ in range(80):                    # bisseção
        mid = (lo + hi) / 2.0
        if H_at(mid) < H_known:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0, ""
