"""
gui/render.py

Geometria dos desenhos, SEM depender de tkinter: as funções devolvem listas de
comandos e o widget (widgets.py) só os executa no Canvas. Isso permite testar
os desenhos fora da interface.

Comandos:
    ("line", [x1,y1,x2,y2,...], cor, largura, seta, tracejado)
    ("poly", [x1,y1,...], preenchimento, contorno)
    ("text", x, y, texto, cor, âncora, tamanho, negrito)
"""

import numpy as np

import quaternion as Q

from . import theme


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


def scene_commands(w, h, q, az, el):
    """Placa 3D com a orientação q (sensor -> navegação, Z para cima)."""
    cdir = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    fwd = -cdir
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    cup = np.cross(right, fwd)
    s = 0.27 * min(w, h)
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

    # placa
    R = Q.to_matrix(q) if q is not None else np.eye(3)
    items = []
    for corners, n, rgb in _FACES:
        nn = R @ n
        if nn @ cdir <= 0:
            continue
        P = corners @ R.T
        shade = 0.55 + 0.45 * max(0.0, nn @ _LIGHT)
        items.append((P.mean(axis=0) @ cdir, proj(P), _hex(rgb * shade)))
    items.sort(key=lambda it: it[0])
    for _, pts, color in items:
        cmds.append(("poly", pts.ravel().tolist(), color, "#0d0f13"))

    # eixos do sensor
    for i, (name, color, L) in enumerate((("X", "#ff6b6b", 1.7), ("Y", "#4ecb71", 1.35),
                                           ("Z", "#6fa3ff", 1.1))):
        tip = R[:, i] * L
        p0, p1, p2 = proj(np.array([(0, 0, 0), tip, tip * 1.12]))
        cmds.append(("line", [*p0, *p1], color, 3, True, None))
        cmds.append(("text", p2[0], p2[1], name, color, "center", 11, True))

    cmds.append(("text", 8, h - 8, "arraste para girar a câmera", theme.MUTED, "sw", 8, False))
    return cmds
