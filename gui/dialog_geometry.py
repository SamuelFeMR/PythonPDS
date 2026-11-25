"""gui/dialog_geometry.py — janelinha com L e R (raramente muda)."""

import tkinter as tk
from tkinter import ttk

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
        self.vars = {"L": tk.StringVar(value=f"{cfg.L:g}"), "R": tk.StringVar(value=f"{cfg.R:g}")}

        for i, (text, key) in enumerate((("L  (distância entre os sensores, mm)", "L"),
                                         ("R  (raio da esfera/cilindro, mm)", "R"))):
            ttk.Label(box, text=text, style="Panel.TLabel").grid(row=i, column=0, sticky="w", pady=3)
            ttk.Entry(box, textvariable=self.vars[key], width=9, justify="right").grid(
                row=i, column=1, sticky="e", padx=(12, 0))
        ttk.Label(box, text="Mudar L ou R apaga a calibração do ToF.", style="Muted.TLabel",
                  wraplength=340, justify="left").grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.msg = ttk.Label(box, text="", style="Muted.TLabel", wraplength=340, justify="left")
        self.msg.grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))
        row = ttk.Frame(box, style="Panel.TFrame")
        row.grid(row=4, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(row, text="Aplicar e salvar", command=self._apply).pack(side="left")
        ttk.Button(row, text="Fechar", command=self.destroy).pack(side="left", padx=(6, 0))

    def _apply(self):
        try:
            L = float(self.vars["L"].get().replace(",", "."))
            R = float(self.vars["R"].get().replace(",", "."))
            self.pipeline.set_geometry(L, R, save=True)
        except ValueError as e:
            self.msg.configure(text=f"Valor inválido: {e}", foreground=theme.BAD)
        except OSError as e:
            self.msg.configure(text=f"Aplicado, mas não consegui salvar: {e}", foreground=theme.WARN)
        else:
            self.msg.configure(text="salvo em geometry_config.json", foreground=theme.GOOD)
