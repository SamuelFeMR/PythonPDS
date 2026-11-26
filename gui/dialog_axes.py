"""gui/dialog_axes.py — troca de eixos e sentido de giro do IMU (vale para acel, giro e mag juntos)."""

import tkinter as tk
from tkinter import ttk

import imu_axes
from geometry import AXES

from . import theme

HINT = ("Cada eixo do programa recebe um eixo do que o ESP32 envia (X, Y, Z do programa; Z = para cima). "
        "Vale para acelerômetro, giroscópio e magnetômetro juntos.\n\n"
        "Como acertar: incline e gire o sensor por alguns segundos e olhe o indicador. "
        "'Discordam' = o giro está invertido em relação ao acelerômetro (comum quando o firmware troca "
        "dois eixos sem inverter o sinal): inverta o sinal de UM eixo, ou marque a inversão do giro.")


class AxesDialog(tk.Toplevel):
    def __init__(self, master, pipeline):
        super().__init__(master, bg=theme.PANEL)
        self.title("Eixos e sentido de giro do IMU")
        self.resizable(False, False)
        self.transient(master)
        self.pipeline = pipeline
        self._after = None

        box = ttk.Frame(self, style="Panel.TFrame", padding=14)
        box.pack(fill="both", expand=True)
        cur = pipeline.imu_axes
        self.vars = [tk.StringVar(value=n) for n in cur.map]
        self.invert = tk.BooleanVar(value=cur.invert_rotation)
        for i, name in enumerate("XYZ"):
            ttk.Label(box, text=f"{name} do programa vem de", style="Panel.TLabel").grid(
                row=i, column=0, sticky="w", pady=3)
            ttk.Combobox(box, textvariable=self.vars[i], values=AXES, state="readonly", width=5
                         ).grid(row=i, column=1, sticky="e", padx=(12, 0))
        ttk.Checkbutton(box, text="Inverter sentido de rotação (giroscópio)", variable=self.invert
                        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

        self.indicator = ttk.Label(box, text="", style="Mono.TLabel", wraplength=380, justify="left")
        self.indicator.grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Label(box, text=HINT, style="Muted.TLabel", wraplength=380, justify="left"
                  ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.msg = ttk.Label(box, text="", style="Muted.TLabel", wraplength=380, justify="left")
        self.msg.grid(row=6, column=0, columnspan=2, sticky="w", pady=(6, 0))
        row = ttk.Frame(box, style="Panel.TFrame")
        row.grid(row=7, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(row, text="Aplicar e salvar", command=self._apply).pack(side="left")
        ttk.Button(row, text="Sem troca", command=self._identity).pack(side="left", padx=(6, 0))
        ttk.Button(row, text="Fechar", command=self._close).pack(side="left", padx=(6, 0))
        self._poll()
        self.protocol("WM_DELETE_WINDOW", self._close)

    # ---------------------------------------------------------------- ações
    def _apply(self):
        try:
            self.pipeline.set_axes([v.get() for v in self.vars], self.invert.get(), save=True)
        except ValueError as e:
            self.msg.configure(text=str(e), foreground=theme.BAD)
        except OSError as e:
            self.msg.configure(text=f"Aplicado, mas não consegui salvar: {e}", foreground=theme.WARN)
        else:
            self.msg.configure(text=f"Aplicado e salvo em {imu_axes.CONFIG_NAME}. "
                                    "O bias do giro foi zerado: recalibre com o sensor parado.",
                               foreground=theme.GOOD)

    def _identity(self):
        for v, n in zip(self.vars, imu_axes.DEFAULT_MAP):
            v.set(n)
        self.invert.set(False)
        self._apply()

    def _poll(self):
        r = self.pipeline.gyro_consistency()
        if r is None:
            text, color = "Giro × acelerômetro: incline/gire o sensor…", theme.MUTED
        elif r > 0.3:
            text, color = f"Giro × acelerômetro: concordam ✓  (r = {r:+.2f})", theme.GOOD
        elif r < -0.3:
            text, color = f"Giro × acelerômetro: DISCORDAM ✗  (r = {r:+.2f})", theme.BAD
        else:
            text, color = f"Giro × acelerômetro: inconclusivo  (r = {r:+.2f}) — gire mais", theme.WARN
        self.indicator.configure(text=text, foreground=color)
        self._after = self.after(300, self._poll)

    def _close(self):
        if self._after:
            self.after_cancel(self._after)
        self.destroy()
