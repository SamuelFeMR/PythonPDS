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
BOARD_S = 0.42                        # tamanho da placa na cena
DOME_R = 2.0                          # raio do hemisfério (unidades da cena)
_FOCUS = np.array([0.0, 0.0, 0.8])    # ponto para onde a câmera olha
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
                corners.append(v * BOARD_S)
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
class ToFInfo:
    """Leitura dos ToF para posicionar o sensor e escrever H e α (dados puros)."""
    res: object = None            # geometry.GeometryResult válido, ou None
    message: str = ""             # por que não há resultado
    bar_axis: str = "+X"          # eixo da placa: do sensor 1 (esquerda) para o 2 (direita)
    toward_axis: str = "-Z"       # eixo da placa: para onde os sensores olham


def _axis(name):
    """'+X', '-Z', ... -> vetor unitário no referencial da placa."""
    v = np.zeros(3)
    v["XYZ".index(name.strip("+-").upper())] = -1.0 if name.startswith("-") else 1.0
    return v


def sensor_pose(R, position_on, alpha_deg=0.0, bar=(1.0, 0.0, 0.0), toward=(0.0, 0.0, -1.0)):
    """
    Onde a placa aparece na cena: (centro, ponto da superfície, ângulo do polo em graus, limitado).

    Desligado: a placa fica acima do polo, com a orientação real R.
    Ligado: o ponto medido é onde a normal da esfera aponta. No referencial da placa essa
    normal (da esfera para o sensor) é  sen(α)·barra − cos(α)·direção_dos_sensores;  girada
    pela orientação R fica no mundo. Placa deitada, sensores para baixo e α = 0 => polo;
    α ≠ 0 (ou placa inclinada) desloca o ponto para o lado. O giro em torno da vertical
    (yaw) é retirado: ele não muda onde o sensor está na esfera, só para onde a placa aponta,
    e assim o ponto não depende do magnetômetro. A placa é desenhada com a orientação real,
    um pouco acima do ponto. Só existe o hemisfério de cima: abaixo do equador, trava nele.
    """
    lift = DOME_R + 0.5
    if not position_on:
        return np.array([0.0, 0.0, lift]), None, 0.0, False
    yaw = np.arctan2(R[1, 0], R[0, 0])
    c, s = np.cos(-yaw), np.sin(-yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    a = np.radians(alpha_deg)
    n = Rz @ R @ (np.sin(a) * np.asarray(bar) - np.cos(a) * np.asarray(toward))
    clamped = n[2] < 0
    if clamped:
        n = np.array([n[0], n[1], 0.0])
        n = n / np.linalg.norm(n) if np.linalg.norm(n) > 1e-9 else np.array([1.0, 0.0, 0.0])
    polar = float(np.degrees(np.arccos(np.clip(n[2], -1.0, 1.0))))
    return n * lift, n * DOME_R, polar, clamped


def _hull(P):
    """Envoltória convexa 2D (cadeia monótona)."""
    pts = sorted(map(tuple, np.round(P, 3)))

    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (p[1] - h[-2][1])
                                   - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(p)
        return h[:-1]

    return half(pts) + half(pts[::-1])


def _board_cmds(proj, center, Rb, cdir):
    cmds, items = [], []
    for corners, n, rgb in _FACES:
        nn = Rb @ n
        if nn @ cdir <= 0:
            continue
        P = corners @ Rb.T + center
        shade = 0.55 + 0.45 * max(0.0, nn @ _LIGHT)
        items.append((P.mean(axis=0) @ cdir, proj(P).ravel().tolist(), _hex(rgb * shade)))
    for _, pts, col in sorted(items, key=lambda it: it[0]):
        cmds.append(("poly", pts, col, "#0d0f13"))
    for i, (name, color, L) in enumerate((("X", "#ff6b6b", 0.95), ("Y", "#4ecb71", 0.8),
                                           ("Z", "#6fa3ff", 0.65))):
        tip = center + Rb[:, i] * L
        p0, p1, p2 = proj(np.array([center, tip, center + Rb[:, i] * L * 1.15]))
        cmds.append(("line", [*p0, *p1], color, 2, True, None))
        cmds.append(("text", p2[0], p2[1], name, color, "center", 10, True))
    return cmds


def _dome_cmds(proj, cdir):
    """Hemisfério fixo: silhueta preenchida, paralelos/meridianos e o meridiano de origem."""
    ang = np.linspace(0, 2 * np.pi, 97)
    lat = np.radians(np.linspace(0, 90, 19))
    surf = np.vstack([DOME_R * np.column_stack((np.cos(a) * np.cos(ang), np.cos(a) * np.sin(ang),
                                                np.full_like(ang, np.sin(a)))) for a in lat])
    cmds = [("poly", np.array(_hull(proj(surf))).ravel().tolist(), "#1b2437", "#4a5a80")]
    back, front = [], []

    def line(points, col_front, col_back, width=1):
        vis = (points / DOME_R) @ cdir > 0.02
        pts = proj(points)
        n, i = len(pts), 0
        while i < n - 1:
            j = i
            while j + 1 < n and vis[j + 1] == vis[i]:
                j += 1
            seg = pts[i:min(j + 1, n - 1) + 1]          # inclui o ponto seguinte: sem buracos
            if len(seg) >= 2:
                (front if vis[i] else back).append(
                    ("line", seg.ravel().tolist(), col_front if vis[i] else col_back, width, False, None))
            i = j + 1

    for la in (30, 60):
        a = np.radians(la)
        line(DOME_R * np.column_stack((np.cos(a) * np.cos(ang), np.cos(a) * np.sin(ang),
                                       np.full_like(ang, np.sin(a)))), "#34425f", "#232c40")
    line(DOME_R * np.column_stack((np.cos(ang), np.sin(ang), np.zeros_like(ang))),
         "#5a6c96", "#2c3650", 2)
    th = np.radians(np.linspace(0, 90, 31))
    for lon in np.radians(np.arange(30, 360, 30)):
        if abs(np.sin(lon)) < 1e-9:      # 0° e 180° (plano X–Z) são o meridiano de origem, abaixo
            continue
        line(DOME_R * np.column_stack((np.sin(th) * np.cos(lon), np.sin(th) * np.sin(lon), np.cos(th))),
             "#34425f", "#232c40")
    # meridiano de origem (plano X–Z): é por onde o ponto anda quando só o α muda
    ths = np.radians(np.linspace(-90, 90, 61))
    line(DOME_R * np.column_stack((np.sin(ths), np.zeros_like(ths), np.cos(ths))),
         theme.WARN, "#6b5a2a", 3)
    for deg in (-90, -60, -30, 0, 30, 60, 90):
        a = np.radians(deg)
        n = np.array([np.sin(a), 0.0, np.cos(a)])
        if n @ cdir > -0.1:
            x, y = proj(n * (DOME_R + 0.28))[0]
            front.append(("text", x, y, f"{deg}°", theme.WARN, "center", 8, True))
    return cmds, back, front


def scene_commands(w, h, q, az, el, tof=None, position_on=False):
    """
    Hemisfério fixo no centro com o sensor sobre ele. `position_on` posiciona o sensor
    pela orientação + α dos ToF (ver sensor_pose). `tof` (ToFInfo) dá o α e escreve H e α
    no canto da cena.
    """
    cdir = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    fwd = -cdir
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    cup = np.cross(right, fwd)
    s = 0.27 * min(w, h) * 1.9 / 2.6
    cx, cy = w / 2, h / 2

    def proj(P):
        P = np.atleast_2d(P) - _FOCUS
        return np.column_stack((cx + (P @ right) * s, cy - (P @ cup) * s))

    R = Q.to_matrix(q) if q is not None else np.eye(3)
    res = tof.res if tof is not None else None
    bar = _axis(tof.bar_axis) if tof is not None else np.array([1.0, 0.0, 0.0])
    toward = _axis(tof.toward_axis) if tof is not None else np.array([0.0, 0.0, -1.0])
    if np.linalg.norm(np.cross(bar, toward)) < 0.5:
        bar, toward = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])
    alpha = res.alpha_deg if res is not None else 0.0
    center, point, polar, clamped = sensor_pose(R, position_on, alpha, bar, toward)
    Rb = R

    dome, back, front = _dome_cmds(proj, cdir)
    board = _board_cmds(proj, center, Rb, cdir)
    in_front = center @ cdir >= 0
    cmds = dome + back + ([] if in_front else board) + front + (board if in_front else [])
    if point is not None:    # ponto medido na superfície e o fio até a placa
        (x0, y0), (x1, y1) = proj(np.array([center, point]))
        cmds.append(("line", [x0, y0, x1, y1], theme.GOOD, 1.5, False, (4, 3)))
        cmds.append(("oval", x1, y1, 5, theme.GOOD, "#0d0f13"))

    # norte (+X) na borda da base
    n0, n1, nt = proj(np.array([(DOME_R + 0.55, 0, 0), (DOME_R + 1.0, 0, 0), (DOME_R + 1.25, 0, 0)]))
    cmds.append(("line", [*n0, *n1], theme.WARN, 2, True, None))
    cmds.append(("text", nt[0], nt[1], "N (+X)", theme.WARN, "center", 9, True))

    # leitura dos ToF
    if res is not None:
        cmds.append(("text", 10, 10, f"H = {res.H:.1f}", theme.GOOD, "nw", 15, True))
        cmds.append(("text", 10, 34, f"α = {res.alpha_deg:+.1f}°", theme.GOOD, "nw", 15, True))
        if res.warning:
            cmds.append(("text", 10, 60, res.warning, theme.WARN, "nw", 9, True))
    else:
        msg = (tof.message if tof is not None else "") or "aguardando leitura dos ToF…"
        cmds.append(("text", 10, 10, "H, α: " + msg, theme.WARN, "nw", 9, True))

    if position_on:
        foot = (f"posição ligada · α {alpha:+.1f}° · a {polar:.0f}° do polo · "
                f"altura {100 * np.cos(np.radians(polar)):.0f}% do raio")
        if res is None:
            foot += " · sem ToF válido: α = 0 assumido"
        if clamped:
            foot += " · limitado ao equador"
    else:
        foot = "posição desligada: placa fixa no topo, com a orientação real"
    cmds.append(("text", 8, h - 22, foot, theme.MUTED, "sw", 8, False))
    cmds.append(("text", 8, h - 8, "arraste para girar a câmera", theme.MUTED, "sw", 8, False))
    return cmds
