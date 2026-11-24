"""gui/app.py — janela principal: barra superior (páginas + conexão) e rodapé."""

import os
import time
import tkinter as tk
import traceback
from tkinter import filedialog, ttk

from datastore import DataStore
from engine import Engine
from pipeline import Pipeline
from sources import DemoSource, ReplaySource, SerialSource

from . import theme
from .page_geom import GeomPage
from .page_pose import PosePage
from .page_raw import RawPage

BAUDS = ("115200", "230400", "460800", "921600")
REFRESH_MS = 40


class App(tk.Tk):
    def __init__(self, port=None, baud=None, demo=False):
        super().__init__()
        self.title("TCC · IMU / Pose")
        self.geometry("1280x800")
        self.minsize(1000, 640)
        theme.apply_style(self)

        self.store = DataStore()
        self.pipeline = Pipeline()
        self.engine = Engine(self.store, self.pipeline)

        self._build_topbar()
        self.container = ttk.Frame(self)
        self.container.pack(fill="both", expand=True)
        self.container.rowconfigure(0, weight=1)
        self.container.columnconfigure(0, weight=1)

        # Para criar outra página: classe nova (herda de Page) + uma linha aqui.
        self.page_defs = (("pose", "1 · Pose", PosePage), ("raw", "2 · Dados brutos", RawPage),
                          ("geom", "3 · Distância", GeomPage))
        self.pages = {}
        for key, _, cls in self.page_defs:
            page = cls(self.container, self)
            page.grid(row=0, column=0, sticky="nsew")
            self.pages[key] = page

        self._build_statusbar()
        self.current = None
        self.show_page("pose")

        for i, (key, _, _) in enumerate(self.page_defs, start=1):
            self.bind(f"<F{i}>", lambda e, k=key: self.show_page(k))
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(REFRESH_MS, self._tick)

        # Conexão automática (opcional): python main_gui.py --port COM7  /  --demo
        if baud:
            self.baud_var.set(str(baud))
        if demo:
            self.after(300, self._start_demo)
        elif port:
            self.port_var.set(port)
            self.after(300, self._toggle_connect)

    # ----------------------------------------------------------- barra topo
    def _build_topbar(self):
        bar = ttk.Frame(self, style="Top.TFrame", padding=(8, 6))
        bar.pack(fill="x")

        self.nav_buttons = {}
        left = ttk.Frame(bar, style="Top.TFrame")
        left.pack(side="left")
        # os botões são criados depois de page_defs existir
        self._nav_parent = left

        right = ttk.Frame(bar, style="Top.TFrame")
        right.pack(side="right")

        ttk.Button(right, text="Arquivo…", command=self._open_file).pack(side="right", padx=(6, 0))
        ttk.Button(right, text="Demo", command=self._start_demo).pack(side="right", padx=(6, 0))
        self.connect_btn = ttk.Button(right, text="Conectar", command=self._toggle_connect)
        self.connect_btn.pack(side="right", padx=(6, 0))

        self.baud_var = tk.StringVar(value=BAUDS[0])
        ttk.Combobox(right, textvariable=self.baud_var, values=BAUDS, width=8
                     ).pack(side="right", padx=(6, 0))
        ttk.Label(right, text="baud", style="Muted.TLabel").pack(side="right", padx=(10, 0))

        ttk.Button(right, text="⟳", width=3, command=self._refresh_ports).pack(side="right", padx=(4, 0))
        self.port_var = tk.StringVar()
        self.port_box = ttk.Combobox(right, textvariable=self.port_var, width=28)
        self.port_box.pack(side="right")
        ttk.Label(right, text="porta", style="Muted.TLabel").pack(side="right", padx=(10, 6))
        self._refresh_ports()

    def _build_nav_buttons(self):
        for key, label, _ in self.page_defs:
            b = ttk.Button(self._nav_parent, text=label, style="Nav.TButton",
                           command=lambda k=key: self.show_page(k))
            b.pack(side="left", padx=(0, 4))
            self.nav_buttons[key] = b

    def _build_statusbar(self):
        self._build_nav_buttons()
        bar = ttk.Frame(self, style="Top.TFrame", padding=(10, 4))
        bar.pack(fill="x", side="bottom")
        self.status_var = tk.StringVar()
        ttk.Label(bar, textvariable=self.status_var, style="Status.TLabel").pack(side="left")
        self.error_var = tk.StringVar()
        self.error_label = ttk.Label(bar, textvariable=self.error_var, style="Status.TLabel",
                                     foreground=theme.BAD)
        self.error_label.pack(side="right")

    # -------------------------------------------------------------- páginas
    def show_page(self, key):
        if self.current:
            self.pages[self.current].on_hide()
        self.current = key
        self.pages[key].tkraise()
        self.pages[key].on_show()
        for k, b in self.nav_buttons.items():
            b.configure(style="NavActive.TButton" if k == key else "Nav.TButton")

    # -------------------------------------------------------------- conexão
    def _refresh_ports(self):
        try:
            from serial.tools import list_ports
            ports = [f"{p.device} - {p.description}" for p in list_ports.comports()]
        except Exception:
            ports = []
        self.port_box.configure(values=ports)
        if ports and not self.port_var.get():
            self.port_var.set(ports[0])

    def _toggle_connect(self):
        if self.engine.running:
            self.engine.stop()
            return
        port = self.port_var.get().split(" - ")[0].strip()
        if not port:
            self.store.set_error("Escolha uma porta (ou digite, ex.: COM5).")
            return
        baud = int(self.baud_var.get())
        self.engine.start(lambda: SerialSource(port, baud))

    def _start_demo(self):
        self.engine.start(lambda: DemoSource(geometry=self.pipeline.geometry_cfg))

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="CSV gravado", filetypes=[("CSV", "*.csv *.txt"), ("Todos", "*.*")])
        if path:
            self.engine.start(lambda: ReplaySource(path))

    # ----------------------------------------------------------------- laço
    def _tick(self):
        try:
            self.pages[self.current].refresh()
            valid, invalid, rate = self.store.counters()
            self.status_var.set(f"{self.store.status}   |   {rate:.1f} Hz   |   "
                                f"válidas {valid}   inválidas {invalid}")
            self.error_var.set(self.store.error)
            self.connect_btn.configure(text="Desconectar" if self.engine.running else "Conectar")
        except Exception:
            traceback.print_exc()
        self.after(REFRESH_MS, self._tick)

    def _on_close(self):
        self.engine.stop()
        self.destroy()
