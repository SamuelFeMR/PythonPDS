"""
sensor_io.py

Leitura da serial do ESP32 e aplicação do alinhamento de eixos.

Linha (18 campos):
  t_us, ax,ay,az, gx,gy,gz, mx,my,mz, dist1,dist2, r,g,b, imu_ok,tof_ok,color_ok
"""

import json
import os
import sys

import numpy as np

N_FIELDS = 18
DEG2RAD = np.pi / 180.0


def parse_line(line):
    """Retorna (t_us, accel, gyro_raw, mag_raw, imu_ok) ou None se inválida."""
    p = line.strip().split(",")
    if len(p) != N_FIELDS:
        return None
    try:
        t_us = int(p[0])
        v = [float(x) for x in p[1:10]]
        imu_ok = int(p[15]) != 0
    except ValueError:
        return None
    return t_us, np.array(v[0:3]), np.array(v[3:6]), np.array(v[6:9]), imu_ok


def open_port(port, baud):
    import serial
    import serial.tools.list_ports
    try:
        return serial.Serial(port, baud, timeout=1.0)
    except serial.SerialException as e:
        print(f"Não foi possível abrir {port}: {e}")
        ports = list(serial.tools.list_ports.comports())
        print("Portas disponíveis:" if ports else "Nenhuma porta serial encontrada.")
        for p in ports:
            print(f"  {p.device} - {p.description}")
        sys.exit(1)


def sample_stream(ser):
    while True:
        raw = ser.readline()
        if not raw:
            continue
        s = parse_line(raw.decode("utf-8", errors="ignore"))
        if s is not None:
            yield s


AXES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
REMAP_NAME = "axes_remap.json"
SENSORS = ("accel", "gyro", "mag")
_IDENTITY = ["+X", "+Y", "+Z"]


def remap_matrix(names):
    """["+Y", "+X", "-Z"] -> matriz M tal que saída = M @ bruto
    (saída X = +Y bruto, saída Y = +X bruto, saída Z = -Z bruto)."""
    if len(names) != 3 or any(n not in AXES for n in names) or len({n[1] for n in names}) != 3:
        raise ValueError(f"{list(names)}: use três eixos diferentes de {', '.join(AXES)}")
    M = np.zeros((3, 3))
    for row, n in enumerate(names):
        M[row, "XYZ".index(n[1])] = 1.0 if n[0] == "+" else -1.0
    return M


def load_remap(path):
    """Lê axes_remap.json -> ({"accel": M, "gyro": M, "mag": M}, mensagem de erro ou "")."""
    out = {k: np.eye(3) for k in SENSORS}
    if not path or not os.path.exists(path):
        return out, ""
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        for k in SENSORS:
            out[k] = remap_matrix(d.get(k, _IDENTITY))
        return out, ""
    except (OSError, ValueError, TypeError, AttributeError) as e:
        return {k: np.eye(3) for k in SENSORS}, f"{REMAP_NAME} com erro ({e}); eixos sem troca"


class Frames:
    """
    Leva gyro e mag para o referencial do ACELERÔMETRO (a referência) e, por último,
    aplica a troca manual de eixos de axes_remap.json (remap) a cada sensor:

        gyro' = P_gyro @ (G @ gyro_rad_s)
        mag'  = P_mag  @ (M @ ((mag - offset) * scale))
        acc'  = P_accel @ acc

    Sem arquivo de configuração: G = I (só graus/s -> rad/s), M = I,
    offset = 0, scale = 1. Sem remap: P = I.
    """

    def __init__(self, cfg=None, gyro_in_rad=False, remap=None):
        self.P = remap or {k: np.eye(3) for k in SENSORS}
        base = 1.0 if gyro_in_rad else DEG2RAD
        self.G = np.eye(3) * base
        self.M = np.eye(3)
        self.offset = np.zeros(3)
        self.scale = np.ones(3)
        self.from_file = False
        if cfg:
            self.G = np.array(cfg["gyro_matrix"], dtype=float)
            self.M = np.array(cfg["mag_matrix"], dtype=float)
            self.offset = np.array(cfg["mag_offset"], dtype=float)
            self.scale = np.array(cfg["mag_scale"], dtype=float)
            self.from_file = True

    def accel(self, a_raw):
        return self.P["accel"] @ a_raw

    def gyro(self, g_raw):
        return self.P["gyro"] @ (self.G @ g_raw)

    def mag(self, m_raw):
        return self.P["mag"] @ (self.M @ ((m_raw - self.offset) * self.scale))


def load_config(path="axes_config.json"):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None
