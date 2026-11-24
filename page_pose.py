"""gui/page_pose.py — página 1: visualização da pose."""

import tkinter as tk
from tkinter import ttk

import numpy as np

from filters import FILTERS

from . import theme
from .page_base import Page
from .widgets import OrientationView, StripChart

WINDOW_S = 10


class PosePage(Page):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.pipeline = app.pipeline
        self.store = app.store

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=3)
        self.rowconfigure(1, weight=1)

        self.view = OrientationView(self)
        self.view.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=(8, 4))

        self.chart = StripChart(self, "Ângulos (°)", ["roll", "pitch", "yaw", "tilt"],
                                theme.SERIES, y_fixed=(-180, 180), wrap=360, height=170)
        self.chart.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=8, pady=(4, 8))

        self._build_side()

    # ------------------------------------------------------------- painel
    def _build_side(self):
        side = ttk.Frame(self, style="Panel.TFrame", padding=12)
        side.grid(row=0, column=1, sticky="ns", padx=(4, 8), pady=(8, 4))

        ttk.Label(side, text="FILTRO", style="H.TLabel").grid(row=0, column=0, sticky="w")
        self.filter_var = tk.StringVar(value=self.pipeline.filter_name)
        cb = ttk.Combobox(side, textvariable=self.filter_var, state="readonly",
                          values=list(FILTERS), width=26)
        cb.grid(row=1, column=0, sticky="we", pady=(4, 2))
        cb.bind("<<ComboboxSelected>>", lambda e: self._on_filter_change())
        self.desc = ttk.Label(side, text="", style="Muted.TLabel", wraplength=250, justify="left")
        self.desc.grid(row=2, column=0, sticky="w", pady=(0, 6))

        self.params_frame = ttk.Frame(side, style="Panel.TFrame")
        self.params_frame.grid(row=3, column=0, sticky="we")
        self._build_params()

        ttk.Label(side, text="ENTRADA", style="H.TLabel").grid(row=4, column=0, sticky="w", pady=(12, 0))
        self.mag_var = tk.BooleanVar(value=self.pipeline.use_mag)
        ttk.Checkbutton(side, text="Usar magnetômetro", variable=self.mag_var,
                        command=lambda: setattr(self.pipeline, "use_mag", self.mag_var.get())
                        ).grid(row=5, column=0, sticky="w", pady=(4, 0))
        self.align_var = tk.BooleanVar(value=self.pipeline.align_enabled)
        ttk.Checkbutton(side, text="Aplicar axes_config.json", variable=self.align_var,
                        command=lambda: self.pipeline.set_align_enabled(self.align_var.get())
                        ).grid(row=6, column=0, sticky="w")
        self.align_label = ttk.Label(side, text="", style="Muted.TLabel")
        self.align_label.grid(row=7, column=0, sticky="w")
        ttk.Button(side, text="Recarregar alinhamento", command=self._reload_alignment
                   ).grid(row=8, column=0, sticky="we", pady=(4, 0))
        self._update_align_label()

        ttk.Label(side, text="CALIBRAÇÃO", style="H.TLabel").grid(row=9, column=0, sticky="w", pady=(12, 0))
        ttk.Button(side, text="Calibrar giroscópio (sensor PARADO)",
                   command=lambda: self.pipeline.start_gyro_calibration(200)
                   ).grid(row=10, column=0, sticky="we", pady=(4, 0))
        ttk.Button(side, text="Reiniciar orientação", command=self.pipeline.reset
                   ).grid(row=11, column=0, sticky="we", pady=(4, 0))
        self.calib_label = ttk.Label(side, text="", style="Muted.TLabel", wraplength=250)
        self.calib_label.grid(row=12, column=0, sticky="w", pady=(2, 0))

        ttk.Label(side, text="POSE", style="H.TLabel").grid(row=13, column=0, sticky="w", pady=(12, 0))
        self.vars = {}
        for i, (key, label) in enumerate((("roll", "roll"), ("pitch", "pitch"), ("yaw", "yaw"),
                                          ("tilt", "tilt"), ("q", "quat"), ("a", "|a|"))):
            row = ttk.Frame(side, style="Panel.TFrame")
            row.grid(row=14 + i, column=0, sticky="we")
            ttk.Label(row, text=label, style="Muted.TLabel", width=6).pack(side="left")
            v = tk.StringVar(value="—")
            ttk.Label(row, textvariable=v, style="Mono.TLabel").pack(side="right")
            self.vars[key] = v

    def _build_params(self):
        for child in self.params_frame.winfo_children():
            child.destroy()
        flt = self.pipeline.filter
        self.desc.configure(text=flt.description)
        for i, p in enumerate(flt.params):
            ttk.Label(self.params_frame, text=p.label, style="Panel.TLabel"
                      ).grid(row=2 * i, column=0, sticky="w")
            val = ttk.Label(self.params_frame, text=f"{flt.values[p.key]:.3f}", style="Mono.TLabel")
            val.grid(row=2 * i, column=1, sticky="e")

            def on_move(v, key=p.key, lab=val):
                self.pipeline.set_filter_param(key, float(v))
                lab.configure(text=f"{float(v):.3f}")

            sc = ttk.Scale(self.params_frame, from_=p.minimum, to=p.maximum,
                           orient="horizontal", command=on_move)
            sc.set(flt.values[p.key])
            sc.grid(row=2 * i + 1, column=0, columnspan=2, sticky="we", pady=(0, 4))
        self.params_frame.columnconfigure(0, weight=1)

    def _on_filter_change(self):
        self.pipeline.set_filter(self.filter_var.get())
        self._build_params()

    def _reload_alignment(self):
        self.pipeline.reload_alignment()
        self._update_align_label()

    def _update_align_label(self):
        if self.pipeline.alignment_loaded:
            self.align_label.configure(text="axes_config.json carregado", foreground=theme.GOOD)
        else:
            self.align_label.configure(
                text="não encontrado (rode check_axes.py)", foreground=theme.WARN)

    # ------------------------------------------------------------ refresh
    def refresh(self):
        pose = self.store.latest_pose
        self.view.draw(pose.q if pose else None)
        self.calib_label.configure(text=self.pipeline.calibration_text())

        if pose:
            r, p, y = pose.euler_deg
            q = pose.q
            self.vars["roll"].set(f"{r:+8.2f}°")
            self.vars["pitch"].set(f"{p:+8.2f}°")
            self.vars["yaw"].set(f"{y:+8.2f}°")
            self.vars["tilt"].set(f"{pose.tilt_deg:8.2f}°")
            self.vars["q"].set(f"{q[0]:+.3f} {q[1]:+.3f} {q[2]:+.3f} {q[3]:+.3f}")
            self.vars["a"].set(f"{pose.accel_norm:8.3f}")

        t, d = self.store.snapshot_pose(WINDOW_S)
        self.chart.update_data(t, d[:, :4] if len(t) else d, WINDOW_S)
