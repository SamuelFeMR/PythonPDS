"""gui/plot_window.py — escolhe as grandezas, vê o gráfico (estilo MATLAB) e salva gráfico/dados."""

import tkinter as tk
from tkinter import filedialog, ttk

import plotting
import quantities

from . import theme


class PlotWindow(tk.Toplevel):
    LIVE_MS = 500

    def __init__(self, master, store):
        super().__init__(master, bg=theme.PANEL)
        self.title("Gráficos")
        self.geometry("1100x640")
        self.store = store
        self.vars = {q.key: tk.BooleanVar(value=q.key in ("roll", "pitch", "yaw")) for q in quantities.QUANTITIES}
        self.mode = tk.StringVar(value="overlay")
        self.live = tk.BooleanVar(value=True)
        self.window_s = tk.StringVar(value="30")
        self.status = tk.StringVar(value="")
        self._after = None

        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
        except ImportError:
            ttk.Label(self, text="matplotlib não instalado.\nRode:  pip install matplotlib",
                      style="Panel.TLabel", padding=20).pack()
            return

        side = ttk.Frame(self, style="Panel.TFrame", padding=8)
        side.pack(side="left", fill="y")
        self._build_side(side)

        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side="left", fill="both", expand=True)
        self.redraw()
        self._tick()
        self.protocol("WM_DELETE_WINDOW", self._close)

    # ------------------------------------------------------------ layout
    def _build_side(self, side):
        # lista rolável de grandezas, agrupadas
        box = ttk.Frame(side, style="Panel.TFrame")
        box.pack(fill="both", expand=True)
        cv = tk.Canvas(box, bg=theme.PANEL, highlightthickness=0, width=215)
        sb = ttk.Scrollbar(box, orient="vertical", command=cv.yview)
        inner = ttk.Frame(cv, style="Panel.TFrame")
        inner.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.create_window((0, 0), window=inner, anchor="nw")
        cv.configure(yscrollcommand=sb.set)
        cv.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")
        cv.bind("<Enter>", lambda e: cv.bind_all("<MouseWheel>", lambda ev: cv.yview_scroll(-1 * (ev.delta // 120), "units")))
        cv.bind("<Leave>", lambda e: cv.unbind_all("<MouseWheel>"))

        for group in quantities.GROUPS:
            ttk.Label(inner, text=group.upper(), style="Muted.TLabel").pack(anchor="w", pady=(8, 2))
            for q in (q for q in quantities.QUANTITIES if q.group == group):
                text = f"{q.label} ({q.unit})" if q.unit else q.label
                ttk.Checkbutton(inner, text=text, variable=self.vars[q.key],
                                command=self.redraw).pack(anchor="w")

        opt = ttk.Frame(side, style="Panel.TFrame")
        opt.pack(fill="x", pady=(8, 0))
        ttk.Radiobutton(opt, text="Sobrepor num eixo", value="overlay", variable=self.mode,
                        command=self.redraw).pack(anchor="w")
        ttk.Radiobutton(opt, text="Um eixo por grandeza", value="stacked", variable=self.mode,
                        command=self.redraw).pack(anchor="w")
        row = ttk.Frame(opt, style="Panel.TFrame")
        row.pack(fill="x", pady=(4, 0))
        ttk.Label(row, text="Janela (s, 0 = tudo)", style="Panel.TLabel").pack(side="left")
        e = ttk.Entry(row, textvariable=self.window_s, width=6, justify="right")
        e.pack(side="right")
        e.bind("<Return>", lambda ev: self.redraw())
        ttk.Checkbutton(opt, text="Atualizar ao vivo", variable=self.live).pack(anchor="w", pady=(4, 0))
        ttk.Button(opt, text="Atualizar agora", command=self.redraw).pack(fill="x", pady=(4, 0))
        ttk.Button(opt, text="Salvar gráfico…", command=self.save_graph).pack(fill="x", pady=(8, 0))
        ttk.Button(opt, text="Salvar dados…", command=self.save_data).pack(fill="x", pady=(4, 0))
        ttk.Label(opt, textvariable=self.status, style="Muted.TLabel", wraplength=210,
                  justify="left").pack(anchor="w", pady=(6, 0))

    # ------------------------------------------------------------- dados
    def _keys(self):
        return [q.key for q in quantities.QUANTITIES if self.vars[q.key].get()]

    def _window(self):
        try:
            w = float(self.window_s.get().replace(",", "."))
        except ValueError:
            w = 0.0
        return w if w > 0 else 1e12

    def _data(self, whole=False):
        t, x = self.store.snapshot_history(1e12 if whole else self._window())
        return t, plotting.columns(t, x)

    def redraw(self):
        t, cols = self._data()
        keys = self._keys()
        plotting.build_figure(t, cols, keys, mode=self.mode.get(), fig=self.fig, max_points=3000)
        self.canvas.draw_idle()
        self.status.set(f"{len(t)} amostras" if len(t) else "sem dados — conecte uma fonte")

    def _tick(self):
        if self.live.get():
            self.redraw()
        self._after = self.after(self.LIVE_MS, self._tick)

    def _close(self):
        if self._after:
            self.after_cancel(self._after)
        self.destroy()

    # ------------------------------------------------------------ salvar
    def save_graph(self):
        keys = self._keys()
        if not keys:
            self.status.set("escolha ao menos uma grandeza")
            return
        exts = [f"*{e}" for e in plotting.GRAPH_FORMATS]
        path = filedialog.asksaveasfilename(parent=self, defaultextension=".png",
                                            filetypes=[("Imagem", " ".join(exts))], title="Salvar gráfico")
        if not path:
            return
        try:
            t, cols = self._data()
            fig = plotting.build_figure(t, cols, keys, mode=self.mode.get())   # resolução total
            self.status.set("salvo: " + plotting.save_figure(fig, path))
        except Exception as e:  # noqa: BLE001
            self.status.set(f"erro ao salvar: {e}")

    def save_data(self):
        keys = self._keys()
        if not keys:
            self.status.set("escolha ao menos uma grandeza")
            return
        path = filedialog.asksaveasfilename(parent=self, defaultextension=".csv", title="Salvar dados",
                                            filetypes=[("CSV", "*.csv"), ("MATLAB .mat", "*.mat")])
        if not path:
            return
        try:
            t, cols = self._data()
            self.status.set("salvo: " + plotting.export_data(path, t, cols, keys))
        except Exception as e:  # noqa: BLE001
            self.status.set(f"erro ao salvar: {e}")
