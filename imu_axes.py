"""
imu_axes.py

Troca de eixos e sentido de giro do IMU, ajustados na interface ("Eixos…") e salvos em
imu_axes.json. Vale para acelerômetro, giroscópio e magnetômetro JUNTOS (o filtro precisa
que os três falem o mesmo referencial):

    saída = M @ entrada          M: cada linha escolhe um eixo (com sinal) do que o ESP32 envia
    giro  = -giro                se invert_rotation (espelha o sentido de rotação)

    {"map": ["+X", "+Y", "+Z"], "invert_rotation": false}

"map" = [de onde vem o X, de onde vem o Y, de onde vem o Z] do programa.
"""

import json
import os
from dataclasses import dataclass

import numpy as np

from geometry import AXES

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_NAME = "imu_axes.json"
DEFAULT_MAP = ("+X", "+Y", "+Z")


@dataclass
class ImuAxes:
    map: tuple = DEFAULT_MAP
    invert_rotation: bool = False
    saved: bool = False
    message: str = ""

    def matrix(self) -> np.ndarray:
        return matrix(self.map)

    @property
    def gyro_sign(self) -> float:
        return -1.0 if self.invert_rotation else 1.0

    @property
    def is_default(self) -> bool:
        return tuple(self.map) == DEFAULT_MAP and not self.invert_rotation


def valid_map(names) -> bool:
    return (len(names) == 3 and all(n in AXES for n in names)
            and len({n[1] for n in names}) == 3)


def matrix(names) -> np.ndarray:
    """["+Y", "+X", "-Z"] -> M com saída = M @ entrada (X sai do +Y, Y sai do +X, Z sai do -Z)."""
    if not valid_map(names):
        raise ValueError(f"{list(names)}: use três eixos diferentes entre {', '.join(AXES)}")
    M = np.zeros((3, 3))
    for row, n in enumerate(names):
        M[row, "XYZ".index(n[1])] = 1.0 if n[0] == "+" else -1.0
    return M


def config_path() -> str:
    for folder in (os.getcwd(), APP_DIR):
        path = os.path.join(folder, CONFIG_NAME)
        if os.path.exists(path):
            return path
    return os.path.join(APP_DIR, CONFIG_NAME)


def load() -> ImuAxes:
    path = config_path()
    if not os.path.exists(path):
        return ImuAxes()
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        names = tuple(d.get("map", DEFAULT_MAP))
        matrix(names)    # valida
        return ImuAxes(names, bool(d.get("invert_rotation", False)), saved=True)
    except (OSError, ValueError, TypeError, AttributeError) as e:
        return ImuAxes(message=f"{CONFIG_NAME} com erro ({e}); eixos sem troca")


def save(axes: ImuAxes) -> str:
    path = config_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"map": list(axes.map), "invert_rotation": axes.invert_rotation},
                  f, indent=2, ensure_ascii=False)
    return path
