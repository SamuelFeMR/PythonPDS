"""gui/dialog_geometry.py — janelinha com L, R e os eixos da placa (raramente muda)."""

import tkinter as tk
from tkinter import ttk

import geometry

from . import theme


class GeometryDialog(tk.Toplevel):
    def __init__(self, master, pipeline):
        super().__init__(master, bg=theme.PANEL)
        self.title("Geometria dos sensores de distância")
        self.resizable(False, False)
        self.transient(master)
        self.pipeline = pipeline

        box = ttk.Frame(self, style="Panel.TFrame", padding=14)
        box.pack(fill="both", expand=True)
        cfg = pipeline.geometry_cfg
        self.vars = {"L": tk.StringVar(value=f"{cfg.L:g}"), "R": tk.StringVar(value=f"{cfg.R:g}"),
                     "bar": tk.StringVar(value=cfg.bar_axis), "toward": tk.StringVar(value=cfg.toward_axis)}

        for i, (text, key) in enumerate((("L  (distância entre os sensores, mm)", "L"),
                                         ("R  (raio da esfera/cilindro, mm)", "R"))):
            ttk.Label(box, text=text, style="Panel.TLabel").grid(row=i, column=0, sticky="w", pady=3)
            ttk.Entry(box, textvariable=self.vars[key], width=9, justify="right").grid(
                row=i, column=1, sticky="e", padx=(12, 0))
        for i, (text, key) in enumerate((("Eixo da placa: sensor 1 → sensor 2", "bar"),
                                         ("Eixo da placa: barra → esfera", "toward")), start=2):
            ttk.Label(box, text=text, style="Panel.TLabel").grid(row=i, column=0, sticky="w", pady=3)
            ttk.Combobox(box, textvariable=self.vars[key], values=geometry.AXES,
                         state="readonly", width=6).grid(row=i, column=1, sticky="e", padx=(12, 0))

        ttk.Label(box, text="Os eixos só servem para desenhar a esfera. Mudar L ou R apaga a "
                            "calibração do ToF.", style="Muted.TLabel", wraplength=340, justify="left"
                  ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.msg = ttk.Label(box, text="", style="Muted.TLabel", wraplength=340, justify="left")
        self.msg.grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))
        row = ttk.Frame(box, style="Panel.TFrame")
        row.grid(row=6, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(row, text="Aplicar e salvar", command=self._apply).pack(side="left")
        ttk.Button(row, text="Fechar", command=self.destroy).pack(side="left", padx=(6, 0))

    def _apply(self):
        try:
            L = float(self.vars["L"].get().replace(",", "."))
            R = float(self.vars["R"].get().replace(",", "."))
            self.pipeline.set_geometry(L, R, save=True, bar_axis=self.vars["bar"].get(),
                                       toward_axis=self.vars["toward"].get())
        except ValueError as e:
            self.msg.configure(text=f"Valor inválido: {e}", foreground=theme.BAD)
        except OSError as e:
            self.msg.configure(text=f"Aplicado, mas não consegui salvar: {e}", foreground=theme.WARN)
        else:
            self.msg.configure(text="salvo em geometry_config.json", foreground=theme.GOOD)
