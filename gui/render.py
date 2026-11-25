"""
gui/render.py

Geometria dos desenhos, SEM depender de tkinter: as funções devolvem listas de
comandos e o widget (widgets.py) só os executa no Canvas. Isso permite testar
os desenhos fora da interface.

Comandos:
    ("line", [x1,y1,x2,y2,...], cor, largura, seta, tracejado)
    ("poly", [x1,y1,...], preenchimento, contorno)
    ("text", x, y, texto, cor, âncora, tamanho, negrito)
    ("oval", x, y, raio, preenchimento, contorno)
"""

from dataclasses import dataclass

import numpy as np

import quaternion as Q

from . import theme

_C_H1 = theme.SERIES[0]     # sensor 1 (esquerda)
_C_H2 = theme.SERIES[2]     # sensor 2 (direita)


# ---------------------------------------------------------------- gráfico
def _fmt(v: float) -> str:
    return f"{v:.0f}" if abs(v) >= 1000 else f"{v:.3g}"


def chart_commands(w, h, title, labels, colors, t, ys, window_s, lo, hi, wrap=None):
    PL, PR, PT, PB = 54, 10, 24, 18
    x0, x1, y0, y1 = PL, w - PR, PT, h - PB
    cmds = []

    for i in range(5):
        frac = i / 4
        yy = y1 - (y1 - y0) * frac
        cmds.append(("line", [x0, yy, x1, yy], theme.GRID, 1, False, None))
        cmds.append(("text", x0 - 6, yy, _fmt(lo + (hi - lo) * frac), theme.MUTED, "e", 8, False))
    cmds.append(("line", [x0, y0, x0, y1], theme.GRID, 1, False, None))
    cmds.append(("text", x0, y1 + 3, f"-{window_s:g} s", theme.MUTED, "nw", 8, False))
    cmds.append(("text", x1, y1 + 3, "0", theme.MUTED, "ne", 8, False))

    if len(t) >= 2:
        px_all = x0 + (t - t[-1] + window_s) / window_s * (x1 - x0)
        stride = max(1, len(t) // max(1, int(x1 - x0)))
        px = px_all[::stride]
        for k, color in enumerate(colors[:ys.shape[1]]):
            yk = ys[::stride, k]
            py = y1 - (yk - lo) / (hi - lo) * (y1 - y0)
            py = np.clip(py, y0, y1)
            if wrap:
                cuts = np.where(np.abs(np.diff(yk)) > wrap / 2)[0] + 1
                segs = zip(np.split(px, cuts), np.split(py, cuts))
            else:
                segs = [(px, py)]
            for sx, sy in segs:
                if len(sx) >= 2:
                    cmds.append(("line", np.column_stack((sx, sy)).ravel().tolist(),
                                 color, 1.6, False, None))

    x = x0
    cmds.append(("text", x, 4, title, theme.FG, "nw", 9, True))
    x += 8 * len(title) + 18
    for k, lab in enumerate(labels):
        val = f" {_fmt(ys[-1, k])}" if len(t) else ""
        txt = f"{lab}{val}"
        cmds.append(("text", x, 5, txt, colors[k % len(colors)], "nw", 9, False))
        x += 7 * len(txt) + 14
    return cmds


# ------------------------------------------------------------ orientação
_HALF = np.array([1.0, 0.65, 0.12])   # meia-dimensão da "placa" (X, Y, Z do sensor)
_FACE_RGB = {  # (eixo, sinal) -> cor base
    (0, 1): (230, 80, 80), (0, -1): (135, 45, 45),
    (1, 1): (80, 200, 110), (1, -1): (45, 118, 65),
    (2, 1): (80, 140, 255), (2, -1): (45, 80, 150),
}


def _faces():
    faces = []
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        for s in (1, -1):
            corners = []
            for sb, sc in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                v = np.zeros(3)
                v[a], v[b], v[c] = s * _HALF[a], sb * _HALF[b], sc * _HALF[c]
                corners.append(v)
            n = np.zeros(3)
            n[a] = s
            faces.append((np.array(corners), n, np.array(_FACE_RGB[(a, s)], float)))
    return faces


_FACES = _faces()
_LIGHT = np.array([0.4, -0.3, 0.85]) / np.linalg.norm([0.4, -0.3, 0.85])


def _hex(rgb):
    r, g, b = (int(c) for c in np.clip(rgb, 0, 255))
    return f"#{r:02x}{g:02x}{b:02x}"


@dataclass
class SphereInfo:
    """O que a cena 3D precisa para desenhar a esfera medida pelos ToF (dados puros)."""
    res: object = None            # geometry.GeometryResult válido, ou None
    L: float = 150.0              # distância entre sensores
    R: float = 60.0               # raio da esfera
    bar_axis: str = "+X"          # eixo da placa: de S1 (esquerda) para S2 (direita)
    toward_axis: str = "+Y"       # eixo da placa: da barra para a esfera
    message: str = ""             # por que não há resultado


_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


def _axis(name):
    """'+X', '-Y', ... -> (vetor unitário, índice do eixo)."""
    sign = -1.0 if name.startswith("-") else 1.0
    i = _AXIS_INDEX[name.strip("+-").upper()]
    v = np.zeros(3)
    v[i] = sign
    return v, i


def _sphere_layout(info):
    """
    Pontos da medição no referencial da PLACA, ou None (sem resultado válido ou eixos iguais).
    A barra (L) ocupa o lado maior da placa (2 unidades) e fica na borda do lado dos sensores;
    z (em direção à esfera) sai dessa borda.
    """
    if info.res is None:
        return None
    bar, ib = _axis(info.bar_axis)
    tw, it = _axis(info.toward_axis)
    if ib == it:
        return None
    k = 2 * _HALF[0] / info.L
    t0 = _HALF[it]

    def B(x, z):
        return bar * ((x - info.L / 2) * k) + tw * (z * k + t0)

    res = info.res
    return dict(k=k, bar=bar, tw=tw, B=B, a=B(0, 0), b=B(info.L, 0), c=B(*res.c), d=B(*res.d),
                h=B(*res.h), foot=B(res.h[0], 0), center=B(*res.center), r=info.R * k)


def _sphere_cmds(proj, Cw, r, cdir, s):
    """Esfera: disco sombreado + paralelos e meridianos (só a metade voltada para a câmera)."""
    px, py = proj(Cw)[0]
    cmds = [("oval", px, py, r * s, "#1b2437", "#4a5a80")]

    def ring(points):
        vis = ((points - Cw) / r) @ cdir > 0.02
        pts = proj(points)
        run = []
        for ok, p in zip(vis, pts):
            if ok:
                run.append(p)
                continue
            if len(run) >= 2:
                cmds.append(("line", np.array(run).ravel().tolist(), "#34425f", 1, False, None))
            run = []
        if len(run) >= 2:
            cmds.append(("line", np.array(run).ravel().tolist(), "#34425f", 1, False, None))

    th = np.linspace(0, 2 * np.pi, 73)
    for lat in np.radians((-50, 0, 50)):
        ring(Cw + r * np.column_stack((np.cos(lat) * np.cos(th), np.cos(lat) * np.sin(th),
                                       np.full_like(th, np.sin(lat)))))
    for lon in np.radians((0, 45, 90, 135)):
        ring(Cw + r * np.column_stack((np.cos(lon) * np.cos(th), np.sin(lon) * np.cos(th), np.sin(th))))
    return cmds


def _sphere_annotations(proj, Rm, lay, info, h):
    """Raios, barra, ponto h, H, tangente e alpha — por cima da placa e da esfera."""
    import math

    cmds = []
    if lay is None:
        same = info.bar_axis.strip("+-").upper() == info.toward_axis.strip("+-").upper()
        msg = ("eixo da barra = eixo para a esfera (ajuste na página 3)" if same
               else (info.message or "aguardando leitura dos ToF…"))
        cmds.append(("text", 10, 10, "esfera: " + msg, theme.WARN, "nw", 9, True))
        return cmds

    res = info.res
    W = lambda v: Rm @ v                      # placa -> mundo

    def seg(p, q_, color, width=2, arrow=False, dash=None):
        cmds.append(("line", [*proj(W(p))[0], *proj(W(q_))[0]], color, width, arrow, dash))

    def dot(p, rad, fill):
        x, y = proj(W(p))[0]
        cmds.append(("oval", x, y, rad, fill, "#0d0f13"))

    def text(p, t, color, anchor="center", size=10, bold=True, dx=0, dy=0):
        """Texto com fundo escuro (fica legível sobre a placa e os eixos). anchor: 'w' ou 'center'."""
        x, y = proj(W(p))[0]
        x, y = x + dx, y + dy
        wd, ht = 0.85 * size * len(t) + 8, 1.9 * size
        x0 = x - 3 if anchor == "w" else x - wd / 2
        cmds.append(("poly", [x0, y - ht / 2, x0 + wd, y - ht / 2, x0 + wd, y + ht / 2, x0, y + ht / 2],
                     "#0d0f13", "#2e323d"))
        cmds.append(("text", x, y, t, color, anchor, size, bold))

    # raios dos ToF até a esfera, corda e barra
    seg(lay["a"], lay["c"], _C_H1, 2)
    seg(lay["b"], lay["d"], _C_H2, 2)
    seg(lay["c"], lay["d"], theme.WARN, 1.5)
    seg(lay["a"], lay["b"], "#f2f4f8", 4)
    for p, col in ((lay["a"], _C_H1), (lay["b"], _C_H2)):
        dot(p, 6, col)
    dot(lay["c"], 4, _C_H1)
    dot(lay["d"], 4, _C_H2)

    # H: da barra (perpendicular) até o ponto h da esfera
    seg(lay["foot"], lay["h"], theme.GOOD, 2.5, True, (6, 3))
    dot(lay["h"], 7, theme.GOOD)
    text((lay["foot"] + lay["h"]) / 2, f"H = {res.H:.1f}", theme.GOOD, "w", 10, True, dx=10)

    # tangente em h (paralela à corda) e paralela à barra, com o arco de alpha
    al = math.radians(res.alpha_deg)
    k, bar, tw = lay["k"], lay["bar"], lay["tw"]
    tl = 0.32 * info.L * k
    tdir = bar * math.cos(al) + tw * math.sin(al)
    seg(lay["h"] - tdir * tl, lay["h"] + tdir * tl, theme.GOOD, 2)
    seg(lay["h"] - bar * tl, lay["h"] + bar * tl, theme.MUTED, 1, False, (4, 4))
    if abs(res.alpha_deg) > 0.3:
        rad = 0.16 * info.L * k
        n = 14
        arc = []
        for i in range(n + 1):
            t = al * i / n
            arc += list(proj(W(lay["h"] + rad * (bar * math.cos(t) + tw * math.sin(t))))[0])
        cmds.append(("line", arc, theme.GOOD, 1.5, False, None))
        text(lay["h"] + 1.35 * rad * (bar * math.cos(al / 2) + tw * math.sin(al / 2)),
             "α", theme.GOOD, "center", 11, True)

    # leitura grande no canto
    cmds.append(("text", 10, 10, f"H = {res.H:.1f}", theme.GOOD, "nw", 15, True))
    cmds.append(("text", 10, 34, f"α = {res.alpha_deg:+.1f}°", theme.GOOD, "nw", 15, True))
    cmds.append(("text", 10, 60, f"esfera R = {info.R:g}", theme.MUTED, "nw", 9, False))
    if res.warning:
        cmds.append(("text", 10, 76, res.warning, theme.WARN, "nw", 9, True))
    cmds.append(("text", 8, h - 24,
                 "posição absoluta ainda não implementada: a esfera é posicionada em relação à placa",
                 theme.MUTED, "sw", 8, False))
    return cmds


def scene_commands(w, h, q, az, el, sphere=None):
    """
    Placa 3D com a orientação q (sensor -> navegação, Z para cima). Se `sphere`
    (SphereInfo) vier, desenha na mesma cena a esfera medida pelos ToF, a altura H
    e o ângulo alpha.
    """
    cdir = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    fwd = -cdir
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    cup = np.cross(right, fwd)

    lay = _sphere_layout(sphere) if sphere is not None else None
    ext = 1.9                                   # meia-largura (em unidades da cena) que cabe na tela
    if lay is not None:
        ext = max(ext, float(np.linalg.norm(lay["center"])) + lay["r"] + 0.3)
    s = 0.27 * min(w, h) * 1.9 / ext
    cx, cy = w / 2, h / 2

    def proj(P):
        P = np.atleast_2d(P)
        return np.column_stack((cx + (P @ right) * s, cy - (P @ cup) * s))

    cmds = []

    # chão (referencial de navegação): grade + marca do norte (+X)
    z0 = -1.05
    for v in np.arange(-1.6, 1.61, 0.4):
        for a, b in (((-1.6, v, z0), (1.6, v, z0)), ((v, -1.6, z0), (v, 1.6, z0))):
            cmds.append(("line", proj(np.array([a, b])).ravel().tolist(),
                         theme.GRID, 1, False, None))
    n0, n1, nt = proj(np.array([(1.6, 0, z0), (2.1, 0, z0), (2.35, 0, z0)]))
    cmds.append(("line", [*n0, *n1], theme.WARN, 2, True, None))
    cmds.append(("text", nt[0], nt[1], "N (+X)", theme.WARN, "center", 9, True))

    # placa (e esfera, na ordem certa de profundidade)
    R = Q.to_matrix(q) if q is not None else np.eye(3)
    items = []
    for corners, n, rgb in _FACES:
        nn = R @ n
        if nn @ cdir <= 0:
            continue
        P = corners @ R.T
        shade = 0.55 + 0.45 * max(0.0, nn @ _LIGHT)
        items.append((P.mean(axis=0) @ cdir, "poly", proj(P), _hex(rgb * shade)))
    if lay is not None:
        Cw = R @ lay["center"]
        items.append((float(Cw @ cdir), "sphere", Cw, None))
    items.sort(key=lambda it: it[0])
    for _, kind, a, b in items:
        if kind == "poly":
            cmds.append(("poly", a.ravel().tolist(), b, "#0d0f13"))
        else:
            cmds += _sphere_cmds(proj, a, lay["r"], cdir, s)

    # eixos do sensor
    for i, (name, color, L) in enumerate((("X", "#ff6b6b", 1.7), ("Y", "#4ecb71", 1.35),
                                           ("Z", "#6fa3ff", 1.1))):
        tip = R[:, i] * L
        p0, p1, p2 = proj(np.array([(0, 0, 0), tip, tip * 1.12]))
        cmds.append(("line", [*p0, *p1], color, 3, True, None))
        cmds.append(("text", p2[0], p2[1], name, color, "center", 11, True))

    if sphere is not None:
        cmds += _sphere_annotations(proj, R, lay, sphere, h)

    cmds.append(("text", 8, h - 8, "arraste para girar a câmera", theme.MUTED, "sw", 8, False))
    return cmds
