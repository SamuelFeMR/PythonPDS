"""gui/dialog_imu_cal.py — calibração mín/máx do acelerômetro e do magnetômetro."""

import tkinter as tk
from tkinter import ttk

import numpy as np

import imu_calib

from . import theme

SENSORS = {"Magnetômetro": "mag", "Acelerômetro": "accel"}
HOW = {
    "mag": "Clique em Iniciar e gire o sensor devagar em todas as direções (longe de metais, "
           "celular e cabos) até os mínimos e máximos pararem de mudar. Corrige o deslocamento "
           "(hard iron) e a escala de cada eixo.",
    "accel": "Clique em Iniciar e apoie o sensor PARADO com cada face para cima e para baixo "
             "(6 posições, alguns segundos em cada). Só entram leituras com o sensor parado. "
             "Corrige o erro de zero e a escala de cada eixo (±1 g).",
}


class ImuCalDialog(tk.Toplevel):
    def __init__(self, master, pipeline):
        super().__init__(master, bg=theme.PANEL)
        self.title("Calibração do acelerômetro e do magnetômetro")
        self.resizable(False, False)
        self.transient(master)
        self.pipeline = pipeline
        self.result = None
        self._after = None

        box = ttk.Frame(self, style="Panel.TFrame", padding=14)
        box.pack(fill="both", expand=True)
        self.sensor = tk.StringVar(value="Magnetômetro")
        top = ttk.Frame(box, style="Panel.TFrame")
        top.grid(row=0, column=0, sticky="we")
        ttk.Combobox(top, textvariable=self.sensor, values=list(SENSORS), state="readonly", width=14
                     ).pack(side="left")
        self.sensor.trace_add("write", lambda *_: self._sensor_changed())
        self.cap_btn = ttk.Button(top, text="Iniciar", command=self._toggle)
        self.cap_btn.pack(side="left", padx=(8, 0))
        self.how = ttk.Label(box, style="Muted.TLabel", wraplength=420, justify="left")
        self.how.grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.table = ttk.Label(box, style="Mono.TLabel", justify="left")
        self.table.grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.msg = ttk.Label(box, style="Muted.TLabel", wraplength=420, justify="left")
        self.msg.grid(row=3, column=0, sticky="w", pady=(8, 0))
        btns = ttk.Frame(box, style="Panel.TFrame")
        btns.grid(row=4, column=0, sticky="e", pady=(10, 0))
        self.apply_btn = ttk.Button(btns, text="Aplicar e salvar", command=self._apply, state="disabled")
        self.apply_btn.pack(side="left")
        ttk.Button(btns, text="Remover", command=self._remove).pack(side="left", padx=(6, 0))
        ttk.Button(btns, text="Fechar", command=self._close).pack(side="left", padx=(6, 0))
        self._sensor_changed()
        self._poll()
        self.protocol("WM_DELETE_WINDOW", self._close)

    @property
    def key(self):
        return SENSORS[self.sensor.get()]

    def _sensor_changed(self):
        if self.pipeline.capturing:
            self.pipeline.capture_stop()
            self.cap_btn.configure(text="Iniciar")
        self.result = None
        self.apply_btn.configure(state="disabled")
        self.how.configure(text=HOW[self.key])
        self._show_active()

    def _show_active(self):
        c = self.pipeline.imu_cal[self.key]
        if c is None:
            self.msg.configure(text="Calibração ativa: nenhuma", foreground=theme.MUTED)
        else:
            self.msg.configure(text="Calibração ativa: offset " + " ".join(f"{v:+.3f}" for v in c.offset)
                                    + " · escala " + " ".join(f"{v:.3f}" for v in c.scale),
                               foreground=theme.MUTED)

    def _toggle(self):
        p = self.pipeline
        if not p.capturing:
            p.capture_start(self.key)
            self.result = None
            self.apply_btn.configure(state="disabled")
            self.cap_btn.configure(text="Parar e calcular")
            return
        p.capture_stop()
        self.cap_btn.configure(text="Iniciar")
        try:
            self.result = imu_calib.compute(self.key, p.capture_points())
        except ValueError as e:
            self.msg.configure(text=f"Não deu para calcular: {e}", foreground=theme.BAD)
            return
        r = self.result
        self.msg.configure(text="Resultado: offset " + " ".join(f"{v:+.3f}" for v in r.offset)
                                + " · escala " + " ".join(f"{v:.3f}" for v in r.scale),
                           foreground=theme.GOOD)
        self.apply_btn.configure(state="normal")

    def _apply(self):
        try:
            self.pipeline.set_imu_cal(self.key, self.result, save=True)
        except OSError as e:
            self.msg.configure(text=f"Aplicado, mas não consegui salvar: {e}", foreground=theme.WARN)
            return
        self.apply_btn.configure(state="disabled")
        self.msg.configure(text=f"Aplicado e salvo em {imu_calib.CONFIG_NAME}.", foreground=theme.GOOD)

    def _remove(self):
        self.pipeline.set_imu_cal(self.key, None, save=True)
        self.result = None
        self.apply_btn.configure(state="disabled")
        self._show_active()

    def _poll(self):
        pts = self.pipeline.capture_points()
        if len(pts):
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            rows = [f"{len(pts)} leituras", "eixo      mín       máx   amplitude"]
            rows += [f"  {n}  {lo[i]:+8.3f}  {hi[i]:+8.3f}  {(hi[i] - lo[i]) / 2:8.3f}"
                     for i, n in enumerate("XYZ")]
            self.table.configure(text="\n".join(rows))
        else:
            self.table.configure(text="")
        self._after = self.after(300, self._poll)

    def _close(self):
        if self._after:
            self.after_cancel(self._after)
        if self.pipeline.capturing:
            self.pipeline.capture_stop()
        self.destroy()
