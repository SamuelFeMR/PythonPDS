"""gui/dialog_mag.py — calibração do magnetômetro (hard iron + soft iron, por elipsoide)."""

import tkinter as tk
from tkinter import ttk

import numpy as np

import magcal

from . import theme

INFO = ("1) Clique em Iniciar e gire o sensor devagar em TODAS as direções (como desenhar uma esfera; "
        "longe de metais, celular e cabos). Procure chegar a 26/26 de cobertura.\n"
        "2) Clique em Parar e calcular. Os gráficos mostram as leituras (amarelo) e as corrigidas (verde); "
        "depois de corrigidas devem formar um círculo centrado na cruz.\n"
        "3) Aplicar e salvar grava em mag_calibration.json (substitui offset/escala do axes_config.json).")
PANEL = 190
METHOD_LABELS = {"Elipsoide (hard + soft iron)": "ellipsoid", "Esfera (só hard iron)": "sphere"}


class MagDialog(tk.Toplevel):
    def __init__(self, master, pipeline):
        super().__init__(master, bg=theme.PANEL)
        self.title("Calibração do magnetômetro")
        self.resizable(False, False)
        self.transient(master)
        self.pipeline = pipeline
        self.cal = None          # resultado do último cálculo (ainda não aplicado)
        self.pts = None          # pontos brutos usados no cálculo
        self._after = None

        box = ttk.Frame(self, style="Panel.TFrame", padding=14)
        box.pack(fill="both", expand=True)
        self.active = ttk.Label(box, text="", style="Panel.TLabel", wraplength=3 * PANEL, justify="left")
        self.active.grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(box, text=INFO, style="Muted.TLabel", wraplength=3 * PANEL, justify="left"
                  ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 8))

        row = ttk.Frame(box, style="Panel.TFrame")
        row.grid(row=2, column=0, columnspan=3, sticky="we")
        self.method = tk.StringVar(value=next(iter(METHOD_LABELS)))
        ttk.Combobox(row, textvariable=self.method, values=list(METHOD_LABELS), state="readonly", width=26
                     ).pack(side="left")
        self.cap_btn = ttk.Button(row, text="Iniciar captura", command=self._toggle)
        self.cap_btn.pack(side="left", padx=(8, 0))
        self.status = ttk.Label(box, text="", style="Mono.TLabel")
        self.status.grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))

        self.canvas = tk.Canvas(box, width=3 * PANEL, height=PANEL + 16, bg=theme.CANVAS_BG, highlightthickness=0)
        self.canvas.grid(row=4, column=0, columnspan=3, pady=(6, 0))
        self.result = ttk.Label(box, text="", style="Mono.TLabel", wraplength=3 * PANEL, justify="left")
        self.result.grid(row=5, column=0, columnspan=3, sticky="w", pady=(6, 0))

        btns = ttk.Frame(box, style="Panel.TFrame")
        btns.grid(row=6, column=0, columnspan=3, sticky="e", pady=(10, 0))
        self.apply_btn = ttk.Button(btns, text="Aplicar e salvar", command=self._apply, state="disabled")
        self.apply_btn.pack(side="left")
        ttk.Button(btns, text="Remover calibração", command=self._remove).pack(side="left", padx=(6, 0))
        ttk.Button(btns, text="Fechar", command=self._close).pack(side="left", padx=(6, 0))
        self._show_active()
        self._draw(None, None)
        self._poll()
        self.protocol("WM_DELETE_WINDOW", self._close)

    # ---------------------------------------------------------------- ações
    def _toggle(self):
        p = self.pipeline
        if not p.mag_capturing:
            p.mag_capture_start()
            self.cal = None
            self.apply_btn.configure(state="disabled")
            self.cap_btn.configure(text="Parar e calcular")
            self.result.configure(text="", foreground=theme.MUTED)
            return
        p.mag_capture_stop()
        self.cap_btn.configure(text="Iniciar captura")
        try:
            self.cal = p.fit_mag(METHOD_LABELS[self.method.get()])
        except ValueError as e:
            self.cal = None
            self.result.configure(text=f"Não deu para calcular: {e}", foreground=theme.BAD)
            return
        self.pts = magcal.thin(p.mag_capture_points())
        c = self.cal
        warn = "" if c.coverage >= 20 else f"\n⚠ cobertura {c.coverage}/26: gire em mais direções e refaça."
        self.result.configure(
            text=(f"hard iron (centro): {c.center[0]:+.1f} {c.center[1]:+.1f} {c.center[2]:+.1f}\n"
                  f"soft iron (diag.): {c.W[0, 0]:.3f} {c.W[1, 1]:.3f} {c.W[2, 2]:.3f}"
                  f"   raio {c.radius:.1f}\n"
                  f"desvio do módulo: {100 * c.std_before:.1f}% → {100 * c.std_after:.2f}%   ({c.samples} pontos)"
                  + warn),
            foreground=theme.GOOD if c.coverage >= 20 else theme.WARN)
        self.apply_btn.configure(state="normal")

    def _apply(self):
        if self.cal is None:
            return
        try:
            self.pipeline.set_mag_cal(self.cal, save=True)
        except OSError as e:
            self.result.configure(text=f"Aplicado, mas não consegui salvar: {e}", foreground=theme.WARN)
        else:
            self.result.configure(text="Aplicado e salvo em mag_calibration.json (filtro reiniciado).",
                                  foreground=theme.GOOD)
        self._show_active()

    def _remove(self):
        self.pipeline.set_mag_cal(None, save=True)
        self.cal = None
        self.apply_btn.configure(state="disabled")
        self.result.configure(text="Calibração removida (voltou a usar offset/escala do axes_config.json).",
                              foreground=theme.MUTED)
        self._show_active()

    def _show_active(self):
        c = self.pipeline.mag_cal
        if c is None:
            text = "Calibração ativa: nenhuma"
        else:
            text = (f"Calibração ativa: {'elipsoide' if c.method == 'ellipsoid' else 'esfera'} · centro "
                    f"{c.center[0]:+.1f} {c.center[1]:+.1f} {c.center[2]:+.1f} · desvio do módulo "
                    f"{100 * c.std_before:.1f}% → {100 * c.std_after:.2f}%")
        self.active.configure(text=text)

    # ----------------------------------------------------------- gráficos
    def _poll(self):
        p = self.pipeline
        if p.mag_capturing:
            pts = magcal.thin(p.mag_capture_points())
            self.status.configure(text=f"capturando… {len(p.mag_capture_points())} leituras · "
                                       f"cobertura {magcal.coverage(pts)}/26")
            self._draw(pts, None)
        elif self.pts is not None:
            self.status.configure(text=f"{len(self.pts)} pontos · cobertura {magcal.coverage(self.pts)}/26")
            self._draw(self.pts, self.cal)
        self._after = self.after(300, self._poll)

    def _draw(self, raw, cal):
        cv = self.canvas
        cv.delete("all")
        pairs = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))
        out = None if raw is None or cal is None else (raw - cal.center) @ cal.W.T
        L = 1.0
        if raw is not None and len(raw):
            L = float(np.abs(raw).max())
            if out is not None:
                L = max(L, float(np.abs(out).max()), cal.radius)
        L *= 1.08
        half = PANEL / 2
        step = max(1, len(raw) // 500) if raw is not None else 1
        for k, (i, j, name) in enumerate(pairs):
            cx, cy = k * PANEL + half, 16 + half
            cv.create_text(cx, 8, text=name, fill=theme.MUTED, font=theme.FONT)
            cv.create_rectangle(cx - half + 4, cy - half + 4, cx + half - 4, cy + half - 4, outline=theme.GRID)
            cv.create_line(cx - half + 4, cy, cx + half - 4, cy, fill=theme.GRID)
            cv.create_line(cx, cy - half + 4, cx, cy + half - 4, fill=theme.GRID)
            if raw is None or not len(raw):
                continue
            s = (half - 8) / L
            if cal is not None:
                r = cal.radius * s
                cv.create_oval(cx - r, cy - r, cx + r, cy + r, outline=theme.MUTED)
            for pts, color in ((raw, theme.WARN), (out, theme.GOOD)):
                if pts is None:
                    continue
                for x, y in pts[::step][:, (i, j)]:
                    px, py = cx + x * s, cy - y * s
                    cv.create_rectangle(px - 1, py - 1, px + 1, py + 1, outline=color, fill=color)

    def _close(self):
        if self._after:
            self.after_cancel(self._after)
        if self.pipeline.mag_capturing:
            self.pipeline.mag_capture_stop()
        self.destroy()
