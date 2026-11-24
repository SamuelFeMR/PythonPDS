"""
datastore.py

Armazenamento compartilhado entre a thread de leitura (escreve) e a interface
(lê). Buffers circulares em numpy, protegidos por lock.
"""

import threading

import numpy as np


class RingBuffer:
    def __init__(self, cols: int, cap: int = 12000):
        self.cap = cap
        self.t = np.zeros(cap)
        self.x = np.zeros((cap, cols))
        self.n = 0
        self.head = 0  # próxima posição de escrita

    def clear(self):
        self.n = 0
        self.head = 0

    def append(self, t: float, row):
        self.t[self.head] = t
        self.x[self.head] = row
        self.head = (self.head + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def window(self, seconds: float):
        """Últimos `seconds` segundos: (t, dados) em ordem cronológica."""
        if self.n == 0:
            return np.zeros(0), np.zeros((0, self.x.shape[1]))
        start = (self.head - self.n) % self.cap
        idx = (start + np.arange(self.n)) % self.cap
        t_all = self.t[idx]
        i0 = np.searchsorted(t_all, t_all[-1] - seconds)
        idx = idx[i0:]
        return self.t[idx].copy(), self.x[idx].copy()


class DataStore:
    RAW_COLS = 14   # ver sample.RAW_COLUMNS
    POSE_COLS = 9   # roll, pitch, yaw, tilt, qw, qx, qy, qz, |a|

    def __init__(self):
        self.lock = threading.Lock()
        self.raw = RingBuffer(self.RAW_COLS)
        self.pose = RingBuffer(self.POSE_COLS)
        self.t0_us = None
        self._prev_t_us = None
        self.latest_sample = None
        self.latest_pose = None
        self.valid = 0
        self.invalid = 0
        self.rate_hz = 0.0
        self.status = "Desconectado"
        self.error = ""

    # ----------------------------------------------------------- escrita
    def clear(self):
        with self.lock:
            self.raw.clear()
            self.pose.clear()
            self.t0_us = None
            self._prev_t_us = None
            self.latest_sample = None
            self.latest_pose = None
            self.valid = 0
            self.invalid = 0
            self.rate_hz = 0.0
            self.error = ""

    def add_raw(self, s):
        with self.lock:
            if self.t0_us is None:
                self.t0_us = s.t_us
            t = (s.t_us - self.t0_us) * 1e-6
            if self._prev_t_us is not None:
                dt = (s.t_us - self._prev_t_us) * 1e-6
                if 0 < dt < 1:
                    inst = 1.0 / dt
                    self.rate_hz = inst if self.rate_hz == 0 else 0.9 * self.rate_hz + 0.1 * inst
            self._prev_t_us = s.t_us
            self.raw.append(t, s.raw_row())
            self.latest_sample = s
            self.valid += 1

    def add_pose(self, pose):
        with self.lock:
            self.pose.append(pose.t_s, pose.as_row())
            self.latest_pose = pose

    def count_bad(self):
        with self.lock:
            self.invalid += 1

    def set_status(self, text: str):
        self.status = text

    def set_error(self, text: str):
        self.error = text

    # ----------------------------------------------------------- leitura
    def snapshot_raw(self, window_s: float):
        with self.lock:
            return self.raw.window(window_s)

    def snapshot_pose(self, window_s: float):
        with self.lock:
            return self.pose.window(window_s)

    def counters(self):
        with self.lock:
            return self.valid, self.invalid, self.rate_hz
