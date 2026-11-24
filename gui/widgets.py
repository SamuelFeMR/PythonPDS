"""gui/widgets.py — widgets de desenho (executam os comandos de render.py)."""

import tkinter as tk

import numpy as np

from . import render, theme


def run_commands(canvas: tk.Canvas, cmds):
    for c in cmds:
        kind = c[0]
        if kind == "line":
            canvas.create_line(*c[1], fill=c[2], width=c[3],
                               arrow=tk.LAST if c[4] else None, dash=c[5])
        elif kind == "poly":
            canvas.create_polygon(*c[1], fill=c[2], outline=c[3])
        elif kind == "text":
            font = ("Segoe UI", c[6]) + (("bold",) if c[7] else ())
            canvas.create_text(c[1], c[2], text=c[3], fill=c[4], anchor=c[5], font=font)


class StripChart(tk.Canvas):
    """Gráfico de linhas rolante. Chame update_data(t, ys, janela_s)."""

    def __init__(self, master, title, labels, colors=None, y_fixed=None, wrap=None, height=150):
        super().__init__(master, bg=theme.CANVAS_BG, highlightthickness=0, height=height)
        self.title = title
        self.labels = labels
        self.colors = colors or theme.SERIES
        self.y_fixed = y_fixed
        self.wrap = wrap
        self._lo = self._hi = None

    def update_data(self, t, ys, window_s):
        w, h = self.winfo_width(), self.winfo_height()
        if w < 80 or h < 60:
            return
        if self.y_fixed:
            lo, hi = self.y_fixed
        else:
            if len(t):
                dmin, dmax = float(np.min(ys)), float(np.max(ys))
            else:
                dmin, dmax = -1.0, 1.0
            pad = 0.1 * max(dmax - dmin, 1e-6)
            tlo, thi = dmin - pad, dmax + pad
            if self._lo is None:
                self._lo, self._hi = tlo, thi
            else:  # expande na hora, encolhe devagar (evita tremer)
                self._lo = tlo if tlo < self._lo else self._lo + (tlo - self._lo) * 0.05
                self._hi = thi if thi > self._hi else self._hi + (thi - self._hi) * 0.05
            lo, hi = self._lo, self._hi
        if hi - lo < 1e-9:
            hi = lo + 1e-9
        self.delete("all")
        run_commands(self, render.chart_commands(
            w, h, self.title, self.labels, self.colors, t, ys, window_s, lo, hi, self.wrap))


class OrientationView(tk.Canvas):
    """Placa 3D que mostra a orientação do sensor. Arraste para girar a câmera."""

    def __init__(self, master):
        super().__init__(master, bg=theme.CANVAS_BG, highlightthickness=0)
        self.az = np.radians(-55.0)
        self.el = np.radians(25.0)
        self._last = None
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._drag)

    def _press(self, e):
        self._last = (e.x, e.y)

    def _drag(self, e):
        if self._last is None:
            return
        dx, dy = e.x - self._last[0], e.y - self._last[1]
        self._last = (e.x, e.y)
        self.az -= dx * 0.01
        self.el = float(np.clip(self.el + dy * 0.01, -1.4, 1.4))

    def draw(self, q):
        w, h = self.winfo_width(), self.winfo_height()
        if w < 80 or h < 80:
            return
        self.delete("all")
        run_commands(self, render.scene_commands(w, h, q, self.az, self.el))
