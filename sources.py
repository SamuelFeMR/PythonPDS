"""
sources.py

Fontes de linhas CSV. Todas têm:
    readline() -> str | None   (None = nada ainda; EOFError = acabou)
    close()
    uses_alignment             (False = dados já alinhados; ignora axes_config.json)
    label
"""

import time

import numpy as np

import quaternion as Q


class SerialSource:
    uses_alignment = True

    def __init__(self, port: str, baud: int):
        import serial  # import tardio: o app abre mesmo sem pyserial instalado
        self.label = f"{port} @ {baud}"
        self.ser = serial.Serial(port, baud, timeout=0.2)
        self.ser.reset_input_buffer()

    def readline(self):
        raw = self.ser.readline()
        if not raw:
            return None
        return raw.decode("utf-8", errors="ignore")

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass


class ReplaySource:
    """Reproduz um CSV gravado (ex.: gravacao.csv do check_axes ou dados_*.csv)."""
    uses_alignment = True

    def __init__(self, path: str, speed: float = 1.0):
        self.label = f"arquivo: {path}"
        with open(path, encoding="utf-8", errors="ignore") as f:
            self.lines = [ln for ln in f if ln.strip()]
        self.i = 0
        self.speed = speed
        self.t_first = None
        self.start = None

    def readline(self):
        if self.i >= len(self.lines):
            raise EOFError
        line = self.lines[self.i]
        self.i += 1
        try:
            t_us = int(line.split(",", 1)[0])
        except ValueError:
            return line
        if self.t_first is None:
            self.t_first = t_us
            self.start = time.perf_counter()
        wait = self.start + (t_us - self.t_first) * 1e-6 / self.speed - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        return line

    def close(self):
        pass


class DemoSource:
    """Sensor simulado (50 Hz), com eixos já alinhados. Serve para testar a interface."""
    uses_alignment = False
    label = "Demo (simulado)"

    def __init__(self, rate_hz: float = 50.0):
        self.dt = 1.0 / rate_hz
        self.k = 0
        self.q = np.array([1.0, 0, 0, 0])          # verdade (sensor -> navegação)
        self.b_nav = np.array([0.5, 0.0, -0.8]) * 300.0
        self.bias = np.radians([0.8, -1.2, 0.5])
        self.rng = np.random.default_rng(1)
        self.next_tick = time.perf_counter()

    def readline(self):
        wait = self.next_tick - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        self.next_tick += self.dt

        t = self.k * self.dt
        w = np.array([0.9 * np.sin(0.6 * t), 0.7 * np.sin(0.37 * t + 1.0), 0.6 * np.cos(0.45 * t)])
        self.q = Q.normalize(Q.multiply(self.q, np.array([1.0, *(0.5 * w * self.dt)])))

        a = Q.rotate_inverse(self.q, [0, 0, 1]) + self.rng.normal(0, 0.01, 3)
        g = np.degrees(w + self.bias) + self.rng.normal(0, 0.2, 3)
        m = Q.rotate_inverse(self.q, self.b_nav) + self.rng.normal(0, 1.5, 3)
        d1 = int(120 + 60 * np.sin(0.8 * t))
        d2 = int(200 + 80 * np.sin(0.5 * t + 2))
        r = int(1300 + 150 * np.sin(0.3 * t))
        gr = int(1100 + 120 * np.sin(0.4 * t + 1))
        b = int(1200 + 100 * np.sin(0.5 * t + 2))

        line = (f"{int(self.k * self.dt * 1e6)},"
                f"{a[0]:.4f},{a[1]:.4f},{a[2]:.4f},"
                f"{g[0]:.4f},{g[1]:.4f},{g[2]:.4f},"
                f"{m[0]:.1f},{m[1]:.1f},{m[2]:.1f},"
                f"{d1},{d2},{r},{gr},{b},1,1,1\n")
        self.k += 1
        return line

    @property
    def truth_q(self):
        return self.q.copy()

    def close(self):
        pass
