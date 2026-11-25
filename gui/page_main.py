"""gui/page_main.py — tela única: linha de dados brutos no topo, cena 3D (hemisfério com o
sensor, H e α), painel de ações e gráficos."""

import os
import time
import tkinter as tk
from tkinter import ttk

from . import theme
from .dialog_geometry import GeometryDialog
from .render import ToFInfo
from .widgets import OrientationView, StripChart

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOW_S = 10


def raw_line(s) -> str:
    """Uma linha com tudo o que o ESP32 mandou na última amostra."""
    if s is None:
        return "aguardando dados do ESP32…"
    bad = [n for n, ok in (("IMU", s.imu_ok), ("ToF", s.tof_ok), ("Cor", s.color_ok)) if not ok]
    return (f"acc {s.accel[0]:+.3f} {s.accel[1]:+.3f} {s.accel[2]:+.3f}  │  "
            f"gyr {s.gyro[0]:+7.1f} {s.gyro[1]:+7.1f} {s.gyro[2]:+7.1f}  │  "
            f"mag {s.mag[0]:+5.0f} {s.mag[1]:+5.0f} {s.mag[2]:+5.0f}  │  "
            f"tof {s.dist1:4d} {s.dist2:4d}  │  rgb {s.red} {s.green} {s.blue}"
            + (f"  │  ⚠ sem {', '.join(bad)}" if bad else ""))


class MainPage(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.pipeline = app.pipeline
        self.store = app.store
        self.engine = app.engine
        self._cal_msg = ""
        self._dialog = None

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # linha de dados brutos
        self.raw_var = tk.StringVar()
        ttk.Label(self, textvariable=self.raw_var, style="RawLine.TLabel", padding=(10, 5)
                  ).grid(row=0, column=0, columnspan=2, sticky="we", padx=8, pady=(8, 0))

        self.view = OrientationView(self)
        self.view.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=(8, 4))
        self._build_side()

        charts = ttk.Frame(self)
        charts.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=8, pady=(4, 8))
        charts.rowconfigure(0, weight=1)
        charts.columnconfigure(0, weight=3, uniform="c")
        charts.columnconfigure(1, weight=2, uniform="c")
        self.chart = StripChart(charts, "Ângulos (°)", ["roll", "pitch", "yaw", "tilt"],
                                theme.SERIES, y_fixed=(-180, 180), wrap=360, height=140)
        self.chart.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.chart_H = StripChart(charts, "H", ["H"], [theme.GOOD], height=140)
        self.chart_H.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

    # ------------------------------------------------------------- painel
    def _build_side(self):
        side = ttk.Frame(self, style="Panel.TFrame", padding=12)
        side.grid(row=1, column=1, sticky="ns", padx=(4, 8), pady=(8, 4))
        side.columnconfigure(0, weight=1, uniform="s")
        side.columnconfigure(1, weight=1, uniform="s")
        r = [0]   # próxima linha da grade

        def nxt():
            r[0] += 1
            return r[0] - 1

        def header(text):
            ttk.Label(side, text=text, style="H.TLabel").grid(
                row=nxt(), column=0, columnspan=2, sticky="w", pady=(10 if r[0] else 0, 2))

        header("POSE")
        self.vars = {}
        row = nxt()
        for i, key in enumerate(("roll", "pitch", "yaw", "tilt", "polo", "azim")):
            v = tk.StringVar(value=f"{key:<5} —")
            ttk.Label(side, textvariable=v, style="Mono.TLabel").grid(
                row=row + i // 2, column=i % 2, sticky="w")
            self.vars[key] = v
        nxt(), nxt()   # 3 linhas: roll/pitch, yaw/tilt, polo/azim
        self.pos_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(side, text="Posição do sensor RGB na esfera", variable=self.pos_var
                        ).grid(row=nxt(), column=0, columnspan=2, sticky="w", pady=(6, 0))

        header("CALIBRAÇÃO")
        ttk.Button(side, text="Giroscópio (sensor PARADO)",
                   command=lambda: self.pipeline.start_gyro_calibration(200)
                   ).grid(row=nxt(), column=0, columnspan=2, sticky="we")
        row = nxt()
        ttk.Label(side, text="ToF: H conhecida (mm)", style="Panel.TLabel").grid(
            row=row, column=0, sticky="w", pady=(8, 0))
        self.H_var = tk.StringVar()
        ttk.Entry(side, textvariable=self.H_var, width=8, justify="right").grid(
            row=row, column=1, sticky="e", pady=(8, 0))
        row = nxt()
        ttk.Button(side, text="Calibrar ToF (α = 0)", command=self._calibrate_tof).grid(
            row=row, column=0, sticky="we", pady=(4, 0), padx=(0, 3))
        ttk.Button(side, text="Limpar", command=self._clear_tof).grid(
            row=row, column=1, sticky="we", pady=(4, 0), padx=(3, 0))
        self.cal_label = ttk.Label(side, text="", style="Muted.TLabel", wraplength=250, justify="left")
        self.cal_label.grid(row=nxt(), column=0, columnspan=2, sticky="w", pady=(4, 0))

        header("AÇÕES")
        row = nxt()
        ttk.Button(side, text="Reiniciar orientação", command=self._reset
                   ).grid(row=row, column=0, sticky="we", padx=(0, 3))
        ttk.Button(side, text="Geometria…", command=self._open_geometry
                   ).grid(row=row, column=1, sticky="we", padx=(3, 0))
        self.rec_btn = ttk.Button(side, text="● Gravar CSV", command=self._toggle_rec)
        self.rec_btn.grid(row=nxt(), column=0, columnspan=2, sticky="we", pady=(4, 0))
        self.info = ttk.Label(side, text="", style="Muted.TLabel", wraplength=250, justify="left")
        self.info.grid(row=nxt(), column=0, columnspan=2, sticky="w", pady=(8, 0))
        self._update_info()

    # ------------------------------------------------------------- ações
    def _calibrate_tof(self):
        try:
            H = float(self.H_var.get().replace(",", "."))
        except ValueError:
            self._cal_msg = "digite a altura H em mm (ex.: 85)"
            return
        if not self.engine.running:
            self._cal_msg = "conecte uma fonte antes de calibrar"
            return
        self._cal_msg = self.pipeline.start_tof_calibration(H)

    def _clear_tof(self):
        self._cal_msg = ""
        self.pipeline.clear_tof_calibration()

    def _reset(self):
        """Reinicia a orientação e relê o filter_config.json."""
        self.pipeline.reload_filter_config()
        self.pipeline.reset()
        self._update_info()

    def _open_geometry(self):
        if self._dialog is not None and self._dialog.winfo_exists():
            self._dialog.lift()
            return
        self._dialog = GeometryDialog(self.app, self.pipeline)

    def _toggle_rec(self):
        if self.engine.recording:
            self.engine.stop_recording()
        elif self.engine.running:
            self.engine.start_recording(os.path.join(APP_DIR, time.strftime("dados_%Y%m%d_%H%M%S.csv")))

    def _update_info(self):
        p = self.pipeline
        lines = [f"Filtro: {p.filter_name}" + ("" if p.use_mag else " (sem mag)") + " · filter_config.json"]
        if p.filter_message:
            lines.append("⚠ " + p.filter_message)
        lines.append("axes_config.json: " + ("carregado" if p.alignment_loaded
                                             else "não encontrado (rode check_axes.py)"))
        self.info.configure(text="\n".join(lines))

    # --------------------------------------------------------------- H / α
    def _tof_info(self):
        g = self.store.latest_geom
        if g is None:
            return ToFInfo(None, "aguardando leitura dos ToF…")
        return ToFInfo(g.result if g.valid else None, "" if g.valid else g.reason)

    # ------------------------------------------------------------ refresh
    def refresh(self):
        self.raw_var.set(raw_line(self.store.latest_sample))

        pose = self.store.latest_pose
        pos = self.store.latest_position
        self.view.draw(pose.q if pose else None, self._tof_info(), pos if self.pos_var.get() else None)

        gyro_txt = self.pipeline.calibration_text()
        tof_txt = self._cal_msg or self.pipeline.tof_calibration_text()
        if not self.pipeline.geometry_cfg.saved:
            tof_txt += "\n⚠ L e R são de exemplo: ajuste em Geometria…"
        self.cal_label.configure(text=f"{gyro_txt}\n{tof_txt}")
        self.rec_btn.configure(text="■ Parar gravação" if self.engine.recording else "● Gravar CSV",
                               style="Rec.TButton" if self.engine.recording else "TButton")

        if pose:
            r, p, y = pose.euler_deg
            for key, val in (("roll", r), ("pitch", p), ("yaw", y), ("tilt", pose.tilt_deg)):
                self.vars[key].set(f"{key:<5}{val:+7.1f}°")
        if pos:   # ponto na esfera: orientação × α
            self.vars["polo"].set(f"polo {pos.polar_deg:7.1f}°")
            self.vars["azim"].set(f"azim {pos.azimuth_deg:+7.1f}°")

        t, d = self.store.snapshot_pose(WINDOW_S)
        self.chart.update_data(t, d[:, :4] if len(t) else d, WINDOW_S)
        t, d = self.store.snapshot_geom(WINDOW_S)
        self.chart_H.update_data(t, d[:, 3:4], WINDOW_S)
