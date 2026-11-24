"""gui/page_raw.py — página 2: dados brutos do ESP32."""

import os
import time
import tkinter as tk
from tkinter import ttk

import numpy as np

from . import theme
from .page_base import Page
from .widgets import StripChart

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (chave, rótulo, formato)
ACCEL = (("ax", "ax", "{:+.4f}"), ("ay", "ay", "{:+.4f}"), ("az", "az", "{:+.4f}"), ("|a|", "|a|", "{:.4f}"))
GYRO = (("gx", "gx", "{:+.4f}"), ("gy", "gy", "{:+.4f}"), ("gz", "gz", "{:+.4f}"))
MAG = (("mx", "mx", "{:+.1f}"), ("my", "my", "{:+.1f}"), ("mz", "mz", "{:+.1f}"), ("|m|", "|m|", "{:.1f}"))
TOF = (("dist1", "dist 1", "{:d}"), ("dist2", "dist 2", "{:d}"))
COLOR = (("red", "R", "{:d}"), ("green", "G", "{:d}"), ("blue", "B", "{:d}"))
LINK = (("t", "t ESP32 (s)", "{:.3f}"), ("rate", "taxa (Hz)", "{:.1f}"),
        ("valid", "linhas válidas", "{:d}"), ("invalid", "linhas inválidas", "{:d}"))


class RawPage(Page):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.store = app.store
        self.engine = app.engine
        self.paused = False
        self.window_s = tk.DoubleVar(value=10)

        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        self._build_controls()
        self._build_values()
        self._build_charts()

    # ----------------------------------------------------------- controles
    def _build_controls(self):
        bar = ttk.Frame(self, style="Panel.TFrame", padding=(10, 6))
        bar.grid(row=0, column=0, columnspan=2, sticky="we", padx=8, pady=(8, 0))

        self.pause_btn = ttk.Button(bar, text="⏸ Pausar gráficos", command=self._toggle_pause)
        self.pause_btn.pack(side="left")
        self.rec_btn = ttk.Button(bar, text="● Gravar CSV", command=self._toggle_rec)
        self.rec_btn.pack(side="left", padx=8)
        self.rec_label = ttk.Label(bar, text="", style="Muted.TLabel")
        self.rec_label.pack(side="left")

        ttk.Label(bar, text="Janela (s)", style="Panel.TLabel").pack(side="right")
        cb = ttk.Combobox(bar, state="readonly", width=5, values=(5, 10, 20, 30),
                          textvariable=self.window_s)
        cb.pack(side="right", padx=(0, 6))

    def _toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.configure(text="▶ Continuar gráficos" if self.paused else "⏸ Pausar gráficos")

    def _toggle_rec(self):
        if self.engine.recording:
            self.engine.stop_recording()
        else:
            if not self.engine.running:
                self.rec_label.configure(text="conecte uma fonte antes de gravar")
                return
            path = os.path.join(APP_DIR, time.strftime("dados_%Y%m%d_%H%M%S.csv"))
            self.engine.start_recording(path)

    # --------------------------------------------------------- valores
    def _build_values(self):
        panel = ttk.Frame(self, style="Panel.TFrame", padding=12)
        panel.grid(row=1, column=0, sticky="ns", padx=(8, 4), pady=8)
        self.vars = {}
        row = 0
        for title, fields in (("ACELERÔMETRO", ACCEL), ("GIROSCÓPIO", GYRO), ("MAGNETÔMETRO", MAG),
                              ("DISTÂNCIA (ToF)", TOF), ("COR (RGB)", COLOR), ("ENLACE", LINK)):
            ttk.Label(panel, text=title, style="H.TLabel").grid(
                row=row, column=0, columnspan=2, sticky="w", pady=(8 if row else 0, 0))
            row += 1
            for key, label, _ in fields:
                ttk.Label(panel, text=label, style="Muted.TLabel").grid(row=row, column=0, sticky="w")
                v = tk.StringVar(value="—")
                ttk.Label(panel, textvariable=v, style="Mono.TLabel", width=11, anchor="e"
                          ).grid(row=row, column=1, sticky="e", padx=(14, 0))
                self.vars[key] = v
                row += 1
            if title == "COR (RGB)":
                self.swatch = tk.Label(panel, text="amostra (escala relativa)", bg="#000000",
                                       fg="white", height=2, font=("Segoe UI", 8))
                self.swatch.grid(row=row, column=0, columnspan=2, sticky="we", pady=(4, 0))
                row += 1

        ttk.Label(panel, text="FLAGS", style="H.TLabel").grid(row=row, column=0, sticky="w", pady=(8, 0))
        row += 1
        flags = ttk.Frame(panel, style="Panel.TFrame")
        flags.grid(row=row, column=0, columnspan=2, sticky="we")
        self.flag_labels = {}
        for name in ("IMU", "ToF", "Cor"):
            lab = tk.Label(flags, text=name, width=6, bg=theme.PANEL2, fg=theme.MUTED,
                           font=("Segoe UI", 9, "bold"))
            lab.pack(side="left", padx=(0, 4))
            self.flag_labels[name] = lab

    # --------------------------------------------------------- gráficos
    def _build_charts(self):
        grid = ttk.Frame(self)
        grid.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=8)
        for c in (0, 1):
            grid.columnconfigure(c, weight=1, uniform="c")
        for r in (0, 1, 2):
            grid.rowconfigure(r, weight=1, uniform="r")

        self.charts = {
            "accel": StripChart(grid, "Acelerômetro", ["ax", "ay", "az"]),
            "gyro": StripChart(grid, "Giroscópio", ["gx", "gy", "gz"]),
            "mag": StripChart(grid, "Magnetômetro", ["mx", "my", "mz"]),
            "tof": StripChart(grid, "ToF", ["d1", "d2"]),
            "rgb": StripChart(grid, "Cor (RGB)", ["R", "G", "B"],
                              ["#ff6b6b", "#4ecb71", "#6fa3ff"]),
        }
        self.charts["accel"].grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=(0, 4))
        self.charts["gyro"].grid(row=0, column=1, sticky="nsew", padx=(4, 0), pady=(0, 4))
        self.charts["mag"].grid(row=1, column=0, sticky="nsew", padx=(0, 4), pady=4)
        self.charts["tof"].grid(row=1, column=1, sticky="nsew", padx=(4, 0), pady=4)
        self.charts["rgb"].grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(4, 0))

    # ------------------------------------------------------------ refresh
    def refresh(self):
        s = self.store.latest_sample
        valid, invalid, rate = self.store.counters()
        self.vars["valid"].set(f"{valid:d}")
        self.vars["invalid"].set(f"{invalid:d}")
        self.vars["rate"].set(f"{rate:.1f}")

        if s is not None:
            vals = {
                "ax": s.accel[0], "ay": s.accel[1], "az": s.accel[2], "|a|": np.linalg.norm(s.accel),
                "gx": s.gyro[0], "gy": s.gyro[1], "gz": s.gyro[2],
                "mx": s.mag[0], "my": s.mag[1], "mz": s.mag[2], "|m|": np.linalg.norm(s.mag),
                "dist1": s.dist1, "dist2": s.dist2,
                "red": s.red, "green": s.green, "blue": s.blue,
                "t": (s.t_us - (self.store.t0_us or 0)) * 1e-6,
            }
            fmts = {k: f for grp in (ACCEL, GYRO, MAG, TOF, COLOR, LINK) for k, _, f in grp}
            for k, v in vals.items():
                self.vars[k].set(fmts[k].format(int(v) if fmts[k].endswith("d}") else v))

            m = max(s.red, s.green, s.blue, 1)
            self.swatch.configure(bg="#%02x%02x%02x" % (
                int(255 * s.red / m), int(255 * s.green / m), int(255 * s.blue / m)))
            for name, ok in (("IMU", s.imu_ok), ("ToF", s.tof_ok), ("Cor", s.color_ok)):
                self.flag_labels[name].configure(
                    bg=theme.GOOD if ok else theme.BAD, fg="#0b0d10")

        if self.engine.recording:
            self.rec_btn.configure(text="■ Parar gravação", style="Rec.TButton")
            self.rec_label.configure(text=os.path.basename(self.engine.recording_path or ""))
        else:
            self.rec_btn.configure(text="● Gravar CSV", style="TButton")

        if self.paused:
            return
        w = float(self.window_s.get())
        t, d = self.store.snapshot_raw(w)
        if len(t) == 0:
            d = np.zeros((0, 14))
        self.charts["accel"].update_data(t, d[:, 0:3], w)
        self.charts["gyro"].update_data(t, d[:, 3:6], w)
        self.charts["mag"].update_data(t, d[:, 6:9], w)
        self.charts["tof"].update_data(t, d[:, 9:11], w)
        self.charts["rgb"].update_data(t, d[:, 11:14], w)
