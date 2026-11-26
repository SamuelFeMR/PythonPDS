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


class Frames:
    """
    Leva gyro e mag para o referencial do ACELERÔMETRO (a referência).

        gyro' = G @ gyro_rad_s
        mag'  = M @ ((mag - offset) * scale)

    Sem arquivo de configuração: G = I (só graus/s -> rad/s), M = I,
    offset = 0, scale = 1.
    """

    def __init__(self, cfg=None, gyro_in_rad=False):
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

    def gyro(self, g_raw):
        return self.G @ g_raw

    def mag(self, m_raw):
        return self.M @ ((m_raw - self.offset) * self.scale)


def load_config(path="axes_config.json"):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None
