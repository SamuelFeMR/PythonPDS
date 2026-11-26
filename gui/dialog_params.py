"""gui/dialog_params.py — ajusta os parâmetros do método de orientação ao vivo e salva em filter_config.json."""

import tkinter as tk
from tkinter import ttk

from . import theme


class ParamsDialog(tk.Toplevel):
    def __init__(self, master, pipeline):
        super().__init__(master, bg=theme.PANEL)
        self.title("Parâmetros do algoritmo")
        self.resizable(False, False)
        self.transient(master)
        self.pipeline = pipeline
        self._shown = None
        self._busy = False
        self._after = None
        self.vars = {}
        self.body = None
        self._build()
        self._watch()
        self.protocol("WM_DELETE_WINDOW", self._close)

    # --------------------------------------------------------------- layout
    def _build(self):
        """(Re)monta o conteúdo para o método atual."""
        if self.body is not None:
            self.body.destroy()
        p = self.pipeline
        self._shown = p.filter_name
        self.body = box = ttk.Frame(self, style="Panel.TFrame", padding=14)
        box.pack(fill="both", expand=True)
        ttk.Label(box, text=p.filter_name, style="H.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(box, text=p.filter.description, style="Muted.TLabel", wraplength=380, justify="left"
                  ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 8))
        values = p.filter_params()
        self.vars = {}
        row = 2
        for par in p.filter.params:
            var = tk.DoubleVar(value=values[par.key])
            self.vars[par.key] = var
            ttk.Label(box, text=par.label, style="Panel.TLabel").grid(row=row, column=0, sticky="w", pady=3)
            ttk.Scale(box, from_=par.minimum, to=par.maximum, variable=var, length=170,
                      command=lambda _v, k=par.key: self._changed(k)
                      ).grid(row=row, column=1, padx=8)
            sp = ttk.Spinbox(box, from_=par.minimum, to=par.maximum, increment=par.step, width=7,
                             textvariable=var, justify="right", format="%.3f",
                             command=lambda k=par.key: self._changed(k))
            sp.grid(row=row, column=2)
            sp.bind("<Return>", lambda _e, k=par.key: self._changed(k))
            sp.bind("<FocusOut>", lambda _e, k=par.key: self._changed(k))
            row += 1
        if not p.filter.params:
            ttk.Label(box, text="Este método não tem parâmetros.", style="Panel.TLabel"
                      ).grid(row=row, column=0, columnspan=3, sticky="w")
            row += 1
        self.mag_var = tk.BooleanVar(value=p.use_mag)
        ttk.Checkbutton(box, text="Usar magnetômetro", variable=self.mag_var,
                        command=lambda: p.set_use_mag(self.mag_var.get())
                        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(8, 0))
        row += 1
        self.msg = ttk.Label(box, text="Mudanças valem na hora. Salvar grava este método e estes valores "
                                       "como padrão ao abrir o programa (filter_config.json).",
                             style="Muted.TLabel", wraplength=380, justify="left")
        self.msg.grid(row=row, column=0, columnspan=3, sticky="w", pady=(8, 0))
        btns = ttk.Frame(box, style="Panel.TFrame")
        btns.grid(row=row + 1, column=0, columnspan=3, sticky="e", pady=(10, 0))
        ttk.Button(btns, text="Salvar", command=self._save).pack(side="left")
        ttk.Button(btns, text="Valores padrão", command=self._defaults).pack(side="left", padx=(6, 0))
        ttk.Button(btns, text="Fechar", command=self._close).pack(side="left", padx=(6, 0))

    # ---------------------------------------------------------------- ações
    def _changed(self, key):
        if self._busy:
            return
        try:
            value = float(self.vars[key].get())
        except (tk.TclError, ValueError):
            return
        par = next(p for p in self.pipeline.filter.params if p.key == key)
        value = min(par.maximum, max(par.minimum, value))
        value = round(round(value / par.step) * par.step, 6)      # encaixa no passo
        self._busy = True
        self.vars[key].set(value)
        self._busy = False
        self.pipeline.set_filter_param(key, value)

    def _defaults(self):
        self.pipeline.reset_filter_params()
        self._busy = True
        for k, v in self.pipeline.filter_params().items():
            self.vars[k].set(v)
        self._busy = False
        self.msg.configure(text="Valores padrão aplicados (ainda não salvos).", foreground=theme.MUTED)

    def _save(self):
        try:
            path = self.pipeline.save_filter_config()
        except OSError as e:
            self.msg.configure(text=f"Não consegui salvar: {e}", foreground=theme.BAD)
        else:
            self.msg.configure(text=f"Salvo em {path}", foreground=theme.GOOD)

    def _watch(self):
        """Se o método for trocado na tela principal, remonta o diálogo."""
        if self.pipeline.filter_name != self._shown:
            self._build()
        self._after = self.after(500, self._watch)

    def _close(self):
        if self._after:
            self.after_cancel(self._after)
        self.destroy()
