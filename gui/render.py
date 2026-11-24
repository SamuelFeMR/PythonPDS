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


# ----------------------------------------------------- sensores de distância
_C_H1 = theme.SERIES[0]     # sensor 1 (esquerda)
_C_H2 = theme.SERIES[2]     # sensor 2 (direita)


def geometry_commands(w, h, res, L, R, message=""):
    """
    Desenho da barra com os dois sensores, os raios, o cilindro, a corda c–d, a
    tangente em h, a altura H e o ângulo alpha. `res` é um geometry.GeometryResult
    válido, ou None (então só a barra e `message`). Coordenadas: x ao longo da
    barra (a -> b), z em direção ao cilindro; z cresce para BAIXO na tela.
    """
    import math

    cmds = []
    s60 = math.sqrt(3) / 2
    apex = (L / 2, s60 * L)

    # área do desenho (margens para os rótulos)
    xmin, xmax, zmax = 0.0, L, apex[1]
    if res is not None:
        cx, cz = res.center
        xmin, xmax = min(xmin, cx - R), max(xmax, cx + R)
        zmax = max(zmax, cz + R)
    pad = 0.10 * max(xmax - xmin, 1e-9)
    xmin, xmax = xmin - pad, xmax + pad
    top, bottom, side = 46, 34, 14
    sc = min((w - 2 * side) / (xmax - xmin), (h - top - bottom) / max(zmax, 1e-9))
    x0 = side + ((w - 2 * side) - sc * (xmax - xmin)) / 2
    y0 = top

    def P(x, z):
        return (x0 + (x - xmin) * sc, y0 + z * sc)

    def line(a, b, color, width=2, arrow=False, dash=None):
        cmds.append(("line", [*P(*a), *P(*b)], color, width, arrow, dash))

    def dot(p, rad, fill):
        cmds.append(("oval", *P(*p), rad, fill, "#0d0f13"))

    def label(p, text, color, anchor="center", size=9, bold=False, dx=0, dy=0):
        x, y = P(*p)
        cmds.append(("text", x + dx, y + dy, text, color, anchor, size, bold))

    if res is not None:
        # cilindro
        cx, cz = res.center
        pts = []
        for i in range(97):
            t = 2 * math.pi * i / 96
            pts += list(P(cx + R * math.cos(t), cz + R * math.sin(t)))
        cmds.append(("line", pts, theme.MUTED, 2, False, None))
        label((cx, cz), "cilindro", theme.MUTED, size=10)

    # raios: sólidos até o cilindro, tracejados até o vértice P (só como guia)
    a, b = (0.0, 0.0), (L, 0.0)
    if res is not None:
        line(a, res.c, _C_H1, 2.5)
        line(res.c, apex, _C_H1, 1, dash=(3, 5))
        line(b, res.d, _C_H2, 2.5)
        line(res.d, apex, _C_H2, 1, dash=(3, 5))
    else:
        line(a, apex, _C_H1, 1, dash=(3, 5))
        line(b, apex, _C_H2, 1, dash=(3, 5))

    # barra e sensores
    line(a, b, theme.FG, 6)
    dot(a, 7, _C_H1)
    dot(b, 7, _C_H2)
    label(a, "S1 (esq.)", _C_H1, "s", 9, True, dy=-12)
    label(b, "S2 (dir.)", _C_H2, "s", 9, True, dy=-12)
    label(((a[0] + b[0]) / 2, 0), f"barra  L = {L:g}", theme.MUTED, "s", 9, dy=-10)

    if res is None:
        cmds.append(("text", w / 2, y0 + 0.55 * (h - top - bottom), message or "sem leitura válida",
                     theme.WARN, "center", 11, True))
        return cmds

    # corda c–d
    line(res.c, res.d, theme.WARN, 2)
    dot(res.c, 5, _C_H1)
    dot(res.d, 5, _C_H2)
    mid1 = (res.c[0] / 2, res.c[1] / 2)                        # meio do raio a–c
    mid2 = ((L + res.d[0]) / 2, res.d[1] / 2)                  # meio do raio b–d
    label(mid1, f"h1 = {res.h1:g}", _C_H1, "e", 10, True, dx=-8)
    label(mid2, f"h2 = {res.h2:g}", _C_H2, "w", 10, True, dx=8)

    # ponto h, tangente (paralela à corda), altura H e ângulo alpha
    al = math.radians(res.alpha_deg)
    hx, hz = res.h
    tl = 0.32 * L
    tdir = (math.cos(al), math.sin(al))
    line((hx - tdir[0] * tl, hz - tdir[1] * tl), (hx + tdir[0] * tl, hz + tdir[1] * tl), theme.GOOD, 2)
    line((hx - tl, hz), (hx + tl, hz), theme.MUTED, 1, dash=(4, 4))       # paralela à barra
    line((hx, 0.0), (hx, hz), theme.GOOD, 2, True, (6, 3))                  # H
    dot(res.h, 5, theme.GOOD)
    label((hx, hz / 2), f"H = {res.H:.1f}", theme.GOOD, "w", 10, True, dx=8)
    label(res.h, "h", theme.GOOD, "se", 10, True, dx=-7, dy=-5)

    # arco do ângulo alpha, à direita de h
    rad = 0.16 * L
    arc = []
    n = 14
    for i in range(n + 1):
        t = al * i / n
        arc += list(P(hx + rad * math.cos(t), hz + rad * math.sin(t)))
    if abs(res.alpha_deg) > 0.3:
        cmds.append(("line", arc, theme.GOOD, 1.5, False, None))
    if abs(res.alpha_deg) > 0.3:
        label((hx + rad, hz + rad * math.sin(al / 2)), "α", theme.GOOD, "w", 11, True, dx=4)

    cmds.append(("text", 10, 8, "x: ao longo da barra   z: em direção ao cilindro   (mesma unidade das leituras)",
                 theme.MUTED, "nw", 8, False))
    info = f"α = {res.alpha_deg:+.1f}°     H = {res.H:.1f}     corda c–d = {res.Y:.1f}"
    cmds.append(("text", 10, h - 8, info, theme.GOOD, "sw", 11, True))
    if res.warning:
        cmds.append(("text", w - 10, h - 8, res.warning, theme.WARN, "se", 9, True))
    return cmds
