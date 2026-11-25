"""gui/page_geom.py — página 3: sensores de distância (h1, h2) -> alpha e H.

A matemática está em geometry.py (e é testada por test_geometry.py). Esta página
só mostra o resultado e deixa ajustar L (distância entre os sensores) e R (raio
do cilindro).
"""

import tkinter as tk
from tkinter import ttk

import numpy as np

import geometry

from . import theme
from .page_base import Page
from .widgets import GeometryView, StripChart

WINDOW_S = 10


class GeomPage(Page):
    def __init__(self, master, app):
        super().__init__(master, app)
        self.pipeline = app.pipeline
        self.store = app.store

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=3)
        self.rowconfigure(1, weight=1)

        self.view = GeometryView(self)
        self.view.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=(8, 4))

        self._build_side()
        self._build_charts()

    # ------------------------------------------------------------- painel
    def _build_side(self):
        side = ttk.Frame(self, style="Panel.TFrame", padding=12)
        side.grid(row=0, column=1, sticky="ns", padx=(4, 8), pady=(8, 4))

        ttk.Label(side, text="GEOMETRIA", style="H.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        cfg = self.pipeline.geometry_cfg
        self.L_var = tk.StringVar(value=f"{cfg.L:g}")
        self.R_var = tk.StringVar(value=f"{cfg.R:g}")
        for i, (text, var) in enumerate((("L  (entre sensores)", self.L_var),
                                         ("R  (raio do cilindro)", self.R_var))):
            ttk.Label(side, text=text, style="Panel.TLabel").grid(row=1 + i, column=0, sticky="w", pady=(4, 0))
            ttk.Entry(side, textvariable=var, width=9, justify="right").grid(
                row=1 + i, column=1, sticky="e", padx=(10, 0), pady=(4, 0))
        self.bar_var = tk.StringVar(value=cfg.bar_axis)
        self.toward_var = tk.StringVar(value=cfg.toward_axis)
        for i, (text, var) in enumerate((("Eixo da placa S1 → S2", self.bar_var),
                                         ("Eixo da placa → esfera", self.toward_var))):
            ttk.Label(side, text=text, style="Panel.TLabel").grid(row=3 + i, column=0, sticky="w", pady=(4, 0))
            ttk.Combobox(side, textvariable=var, values=geometry.AXES, state="readonly", width=6
                         ).grid(row=3 + i, column=1, sticky="e", padx=(10, 0), pady=(4, 0))
        ttk.Button(side, text="Aplicar e salvar", command=self._apply).grid(
            row=5, column=0, columnspan=2, sticky="we", pady=(6, 0))
        self.cfg_label = ttk.Label(side, text="", style="Muted.TLabel", wraplength=260, justify="left")
        self.cfg_label.grid(row=6, column=0, columnspan=2, sticky="w", pady=(2, 0))
        ttk.Label(side, text="L e R na mesma unidade dos sensores (mm); raios a 60° da barra. "
                             "Os eixos da placa (IMU) só servem para desenhar a esfera na página 1.",
                  style="Muted.TLabel", wraplength=260, justify="left"
                  ).grid(row=7, column=0, columnspan=2, sticky="w")
        self._update_cfg_label()

        self.vars = {}
        row = 8
        for title, fields in (("LEITURAS", (("h1", "h1 (esquerda)"), ("h2", "h2 (direita)"))),
                              ("RESULTADO", (("alpha", "α"), ("H", "H"), ("Y", "corda c–d")))):
            ttk.Label(side, text=title, style="H.TLabel").grid(
                row=row, column=0, columnspan=2, sticky="w", pady=(12, 0))
            row += 1
            for key, text in fields:
                ttk.Label(side, text=text, style="Muted.TLabel").grid(row=row, column=0, sticky="w")
                v = tk.StringVar(value="—")
                ttk.Label(side, textvariable=v, style="Mono.TLabel", width=10, anchor="e").grid(
                    row=row, column=1, sticky="e")
                self.vars[key] = v
                row += 1

        self.status = ttk.Label(side, text="", style="Muted.TLabel", wraplength=260, justify="left")
        self.status.grid(row=row, column=0, columnspan=2, sticky="w", pady=(10, 0))

    def _build_charts(self):
        row = ttk.Frame(self)
        row.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=8, pady=(4, 8))
        row.rowconfigure(0, weight=1)
        for c in range(3):
            row.columnconfigure(c, weight=1, uniform="g")
        self.chart_h = StripChart(row, "Distâncias", ["h1", "h2"], [theme.SERIES[0], theme.SERIES[2]], height=150)
        self.chart_a = StripChart(row, "α (°)", ["α"], [theme.WARN], height=150)
        self.chart_H = StripChart(row, "H", ["H"], [theme.GOOD], height=150)
        self.chart_h.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.chart_a.grid(row=0, column=1, sticky="nsew", padx=4)
        self.chart_H.grid(row=0, column=2, sticky="nsew", padx=(4, 0))

    # ------------------------------------------------------------ geometria
    def _apply(self):
        try:
            L = float(self.L_var.get().replace(",", "."))
            R = float(self.R_var.get().replace(",", "."))
            self.pipeline.set_geometry(L, R, save=True, bar_axis=self.bar_var.get(),
                                       toward_axis=self.toward_var.get())
        except ValueError as e:
            self.cfg_label.configure(text=f"Valor inválido: {e}", foreground=theme.BAD)
            return
        except OSError as e:
            self.cfg_label.configure(text=f"Aplicado, mas não consegui salvar: {e}", foreground=theme.WARN)
            return
        self._update_cfg_label()

    def _update_cfg_label(self):
        if self.pipeline.geometry_cfg.saved:
            self.cfg_label.configure(text="salvo em geometry_config.json", foreground=theme.GOOD)
        else:
            self.cfg_label.configure(
                text="valores de EXEMPLO: informe o L e o R do seu protótipo e salve.",
                foreground=theme.WARN)

    # ------------------------------------------------------------ refresh
    def refresh(self):
        cfg = self.pipeline.geometry_cfg
        g = self.store.latest_geom

        if g is None:
            self.view.draw(None, cfg.L, cfg.R, "aguardando dados do ESP32…")
            self.status.configure(text="", foreground=theme.MUTED)
        else:
            self.view.draw(g.result if g.valid else None, cfg.L, cfg.R, g.reason)
            self.vars["h1"].set(f"{g.h1:g}")
            self.vars["h2"].set(f"{g.h2:g}")
            if g.valid:
                self.vars["alpha"].set(f"{g.alpha_deg:+.2f}°")
                self.vars["H"].set(f"{g.H:.2f}")
                self.vars["Y"].set(f"{g.Y:.2f}")
                self.status.configure(text=g.warning or "leitura válida",
                                      foreground=theme.WARN if g.warning else theme.GOOD)
            else:
                for k in ("alpha", "H", "Y"):
                    self.vars[k].set("—")
                self.status.configure(text=g.reason, foreground=theme.BAD)

        t, d = self.store.snapshot_geom(WINDOW_S)
        self.chart_h.update_data(t, d[:, 0:2], WINDOW_S)
        self.chart_a.update_data(t, d[:, 2:3], WINDOW_S)
        self.chart_H.update_data(t, d[:, 3:4], WINDOW_S)
