"""
test_position.py — confere o cruzamento orientação × α (position.py) contra uma simulação.

    python test_position.py

Uma esfera fixa na origem do mundo; a placa fica numa pose qualquer com o centro da esfera
no plano da barra e dos sensores (o que a geometria dos ToF supõe). Os raios dão h1 e h2,
geometry.compute dá α, e position.py (só com a orientação e α) tem que achar a direção do
centro da esfera até o ponto medido — que aqui é conhecida.
"""

import math
import random

import numpy as np

import geometry as G
import position as P
import quaternion as Q
import test_geometry as T


def random_quat(rnd):
    q = np.array([rnd.gauss(0, 1) for _ in range(4)])
    return q / np.linalg.norm(q)


def main():
    rnd = random.Random(3)
    names = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
    pairs = [(b, t) for b in names for t in names if b[-1] != t[-1]]
    n = worst = worst_yaw = 0
    while n < 2000:
        L = rnd.uniform(80, 300)
        Rs = rnd.uniform(30, 120)
        O = (L / 2 + rnd.uniform(-0.5, 0.5) * Rs, rnd.uniform(1.0, 3.0) * L)
        if min(math.hypot(O[0], O[1]), math.hypot(L - O[0], O[1])) <= Rs:
            continue
        tr = T.truth(L, Rs, O)
        if tr is None:
            continue
        res = G.compute(tr[0], tr[1], L, Rs)
        if not res.valid or res.warning:
            continue
        bar_name, tow_name = rnd.choice(pairs)
        bar, tow = P.axis_vector(bar_name), P.axis_vector(tow_name)
        q = random_quat(rnd)
        R = Q.to_matrix(q)

        # verdade: ponto h no mundo, com o centro da esfera na origem
        origin = -R @ (bar * O[0] + tow * O[1])              # sensor 1 no mundo
        h_world = origin + R @ (bar * res.h[0] + tow * res.h[1])
        truth_n = h_world / np.linalg.norm(h_world)
        assert abs(np.linalg.norm(h_world) - Rs) < 1e-6

        est = P.world_normal(R, res.alpha_deg, bar, tow)
        worst = max(worst, float(np.linalg.norm(est - truth_n)))

        # o yaw não pode mexer no ponto (e a posição pública não tem yaw)
        pos = P.compute(0.0, q, res.alpha_deg, bar_name, tow_name)
        a = rnd.uniform(0, 2 * math.pi)
        qz = np.array([math.cos(a / 2), 0, 0, math.sin(a / 2)])
        pos2 = P.compute(0.0, Q.multiply(qz, q), res.alpha_deg, bar_name, tow_name)
        worst_yaw = max(worst_yaw, float(np.linalg.norm(pos.normal - pos2.normal)))
        n += 1
    print(f"{n} poses aleatórias: erro máx. da normal {worst:.2e}, erro máx. com yaw {worst_yaw:.2e}")
    assert worst < 1e-9
    assert worst_yaw < 1e-9
    # casos de referência
    I = np.array([1.0, 0, 0, 0])
    p = P.compute(0, I, 0.0)
    assert p.polar_deg < 1e-9 and abs(p.normal[2] - 1) < 1e-12              # deitada, α=0: polo
    p = P.compute(0, I, 20.0)
    assert abs(p.polar_deg - 20) < 1e-9 and abs(p.azimuth_deg) < 1e-9       # α>0: lado +X
    p = P.compute(0, I, -20.0)
    assert abs(p.polar_deg - 20) < 1e-9 and abs(abs(p.azimuth_deg) - 180) < 1e-9
    p = P.compute(0, I, None)
    assert p.alpha_assumed and p.polar_deg < 1e-9
    a = math.radians(30)                                                    # inclinada 30° em Y, α=0
    p = P.compute(0, np.array([math.cos(a / 2), 0, math.sin(a / 2), 0]), 0.0)
    assert abs(p.polar_deg - 30) < 1e-9
    # sensores para cima (placa de cabeça para baixo): abaixo do equador, trava nele
    p = P.compute(0, np.array([0.0, 1, 0, 0]), 0.0)
    assert p.clamped and abs(p.polar_deg - 90) < 1e-9
    print("OK")


if __name__ == "__main__":
    main()
