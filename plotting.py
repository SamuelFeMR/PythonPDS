"""
plotting.py

Gráficos no estilo MATLAB (fundo branco, caixa fechada, grade, cores padrão do MATLAB, legenda
com moldura) e exportação dos dados. Sem interface: só matplotlib (Figure, sem pyplot), então
serve tanto para a janela quanto para scripts:

    t, x = store.snapshot_history(30)
    cols = plotting.columns(t, x)
    fig = plotting.build_figure(t, cols, ["roll", "pitch"], mode="stacked")
    plotting.save_figure(fig, "pose.png")
    plotting.export_data("pose.csv", t, cols, ["roll", "pitch"])      # ou .mat

Formatos do gráfico: .png .jpg .pdf .svg .eps. Dados: .csv, .mat (precisa de scipy).
"""

import csv
import math
import os

import numpy as np
from matplotlib.figure import Figure

import quantities

# cores padrão do MATLAB (get(groot,'defaultAxesColorOrder'))
MATLAB_COLORS = ("#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F")
GRAPH_FORMATS = (".png", ".jpg", ".pdf", ".svg", ".eps")
MODES = ("overlay", "stacked")      # sobrepor numa só / um eixo por grandeza (subplot)


def columns(t, x) -> dict:
    """Matriz do histórico -> {chave: vetor}."""
    return {k: x[:, i] for i, k in enumerate(quantities.KEYS)}


def label(key: str) -> str:
    q = quantities.BY_KEY[key]
    return f"{q.label} ({q.unit})" if q.unit else q.label


def style_axes(ax):
    """Aparência de eixos do MATLAB: caixa, grade fina, marcas para dentro."""
    ax.set_facecolor("white")
    ax.grid(True, color="#d9d9d9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(direction="in", top=True, right=True, labelsize=9, width=0.6)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
        sp.set_color("#262626")


def _decimate(t, y, max_points):
    if max_points and len(t) > max_points:
        step = int(math.ceil(len(t) / max_points))
        return t[::step], y[::step]
    return t, y


def build_figure(t, cols: dict, keys, mode: str = "overlay", title: str = "",
                 fig: Figure = None, max_points: int = 0) -> Figure:
    """
    t = tempos (s); cols = {chave: vetor} (columns()); keys = grandezas a plotar.
    mode "overlay": todas no mesmo eixo; "stacked": um eixo por grandeza, com o tempo ligado.
    max_points > 0 reduz a quantidade de pontos desenhados (só para a tela; ao salvar, use 0).
    """
    keys = [k for k in keys if k in quantities.BY_KEY]
    fig = fig if fig is not None else Figure(figsize=(9, 5.4), dpi=100)
    fig.clear()
    fig.set_facecolor("white")
    if not keys:
        ax = fig.add_subplot(111)
        style_axes(ax)
        ax.set_xlabel("Tempo (s)")
        ax.text(0.5, 0.5, "selecione ao menos uma grandeza", ha="center", va="center",
                transform=ax.transAxes, color="#7f7f7f")
        return fig
    t = np.asarray(t, float)
    t0 = t[0] if len(t) else 0.0

    def draw(ax, key, color):
        tt, yy = _decimate(t - t0, np.asarray(cols[key], float), max_points)
        ax.plot(tt, yy, color=color, linewidth=1.4, label=label(key))

    if mode == "stacked" and len(keys) > 1:
        axes = fig.subplots(len(keys), 1, sharex=True)
        for i, (ax, key) in enumerate(zip(axes, keys)):
            style_axes(ax)
            draw(ax, key, MATLAB_COLORS[i % len(MATLAB_COLORS)])
            ax.set_ylabel(label(key), fontsize=9)
        axes[-1].set_xlabel("Tempo (s)", fontsize=10)
        if title:
            axes[0].set_title(title, fontsize=11, fontweight="normal")
        fig.align_ylabels(axes)
    else:
        ax = fig.add_subplot(111)
        style_axes(ax)
        for i, key in enumerate(keys):
            draw(ax, key, MATLAB_COLORS[i % len(MATLAB_COLORS)])
        units = {quantities.BY_KEY[k].unit for k in keys}
        ax.set_xlabel("Tempo (s)", fontsize=10)
        ax.set_ylabel(f"({units.pop()})" if len(units) == 1 and units != {""} else "Valor", fontsize=10)
        if title:
            ax.set_title(title, fontsize=11, fontweight="normal")
        leg = ax.legend(loc="best", fontsize=9, frameon=True, edgecolor="#262626", fancybox=False)
        leg.get_frame().set_linewidth(0.6)
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, path: str, dpi: int = 200) -> str:
    """Salva o gráfico; o formato vem da extensão (padrão .png). Retorna o caminho final."""
    root, ext = os.path.splitext(path)
    if ext.lower() not in GRAPH_FORMATS:
        path, ext = path + ".png", ".png"
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
    return path


def export_data(path: str, t, cols: dict, keys) -> str:
    """Salva as grandezas escolhidas (e o tempo) em .csv ou .mat. Retorna o caminho final."""
    keys = [k for k in keys if k in quantities.BY_KEY]
    root, ext = os.path.splitext(path)
    ext = ext.lower()
    t = np.asarray(t, float)
    if ext == ".mat":
        try:
            from scipy.io import savemat
        except ImportError as e:
            raise RuntimeError("salvar .mat precisa do scipy (pip install scipy); use .csv") from e
        savemat(path, {"t": t, **{k: np.asarray(cols[k], float) for k in keys}})
        return path
    if ext != ".csv":
        path += ".csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_s"] + [f"{k}[{quantities.BY_KEY[k].unit}]" if quantities.BY_KEY[k].unit else k
                              for k in keys])
        for i in range(len(t)):
            w.writerow([f"{t[i]:.4f}"] + ["" if math.isnan(cols[k][i]) else f"{cols[k][i]:.6g}"
                                          for k in keys])
    return path
