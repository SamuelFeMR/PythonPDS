"""
test_geometry.py — confere geometry.py contra uma simulação geométrica de raios.

    python test_geometry.py

Para cada posição do cilindro: dispara os dois raios (60° da barra), calcula h1 e h2
exatos, passa por geometry.compute() e compara alpha e H com a geometria verdadeira.
Também confere que a forma fechada de H é a média das duas fórmulas do PDF
(com a altura do triângulo equilátero e a lei dos senos corrigidas).
"""

import math
import random

import geometry as G

S60, C60 = math.sin(math.radians(60)), math.cos(math.radians(60))


def ray_hit(p0, d, O, R):
    wx, wy = p0[0] - O[0], p0[1] - O[1]
    b = wx * d[0] + wy * d[1]
    c = wx * wx + wy * wy - R * R
    disc = b * b - c
    return None if disc < 0 else -b - math.sqrt(disc)


def truth(L, R, O):
    h1 = ray_hit((0.0, 0.0), (C60, S60), O, R)
    h2 = ray_hit((L, 0.0), (-C60, S60), O, R)
    if h1 is None or h2 is None:
        return None
    c = (C60 * h1, S60 * h1)
    d = (L - C60 * h2, S60 * h2)
    t = (d[0] - c[0], d[1] - c[1])
    alpha = math.degrees(math.atan2(t[1], t[0]))
    tn = math.hypot(*t)
    n = (-t[1] / tn, t[0] / tn)
    if n[1] < 0:
        n = (-n[0], -n[1])
    H = (O[1] - n[1] * R)                   # z do ponto médio do arco c–d
    return h1, h2, alpha, H


def pdf_H(h1, h2, L, R):
    """Fórmulas do PDF, com lei dos senos e altura √3/2·L corrigidas. Retorna (H(β1), H(β2))."""
    u, v = L - h1, L - h2
    Y = math.sqrt(u * u + v * v - u * v)
    # lei dos senos (como no PDF, com sen em vez de cos) seria asin(v·sen60/Y); mas asin só
    # devolve ângulos agudos e β pode passar de 90° (β1+β2 = 120°). atan2 resolve:
    b1 = math.atan2(v * S60, u - v * C60)
    b2 = math.atan2(u * S60, v - u * C60)
    assert abs(b1 + b2 - math.radians(120)) < 1e-9
    F = R - math.sqrt(R * R - (Y / 2) ** 2)
    fp = math.hypot(Y / 2, F)
    dl = math.atan(2 * F / Y)
    alt = math.sqrt(3) / 2 * L
    return tuple(alt - fp * math.sin(b + dl) / math.sin(math.radians(30)) for b in (b1, b2))


def main():
    rnd = random.Random(1)
    n = worst_a = worst_h = worst_pdf = 0
    skipped_miss = n_warn = 0
    for _ in range(5000):
        L = rnd.uniform(60, 300)
        R = rnd.uniform(20, 150)
        O = (L / 2 + rnd.uniform(-0.6, 0.6) * R, rnd.uniform(0.3, 3.0) * L)
        # os dois sensores precisam estar FORA do cilindro
        if min(math.hypot(0 - O[0], 0 - O[1]), math.hypot(L - O[0], 0 - O[1])) <= R:
            continue
        tr = truth(L, R, O)
        if tr is None:
            skipped_miss += 1
            continue
        h1, h2, a_t, H_t = tr
        r = G.compute(h1, h2, L, R)
        if not r.valid:
            continue
        worst_a = max(worst_a, abs(r.alpha_deg - a_t))      # alpha não depende de R
        if r.warning:                                        # corda ≈ diâmetro: H mal condicionado
            n_warn += 1
            continue
        n += 1
        worst_h = max(worst_h, abs(r.H - H_t))
        Hb1, Hb2 = pdf_H(h1, h2, L, R)
        worst_pdf = max(worst_pdf, abs(r.H - (Hb1 + Hb2) / 2))
        # centro e ponto h devolvidos estão sobre o círculo
        assert abs(math.hypot(r.c[0] - r.center[0], r.c[1] - r.center[1]) - R) < 1e-6
        assert abs(math.hypot(r.h[0] - r.center[0], r.h[1] - r.center[1]) - R) < 1e-6
        assert abs(r.center[1] - O[1]) < 1e-6 and abs(r.center[0] - O[0]) < 1e-6

    print(f"{n} casos verificados + {n_warn} com aviso de corda≈diâmetro ({skipped_miss} sem acerto, descartados)")
    print(f"erro máx. de alpha            : {worst_a:.2e} °")
    print(f"erro máx. de H (vs verdade)   : {worst_h:.2e}")
    print(f"|H - média das fórmulas do PDF|: {worst_pdf:.2e}")
    assert worst_a < 1e-6 and worst_h < 1e-6 and worst_pdf < 1e-6

    # casos inválidos
    assert not G.compute(0, 50, 100, 50).valid
    assert not G.compute(float("nan"), 50, 100, 50).valid
    assert not G.compute(120, 120, 100, 50).valid        # h1 + h2 >= 2L
    assert not G.compute(10, 90, 100, 20).valid          # corda > diâmetro
    assert G.compute(40, 40, 100, 50).warning == ""
    assert G.compute(40, 40, 100, 31).warning != ""      # Y = 60 ≈ 2R = 62
    # sinal: sensor da direita mais longe => alpha > 0
    assert G.compute(30, 50, 100, 60).alpha_deg > 0 > G.compute(50, 30, 100, 60).alpha_deg
    print("OK")


if __name__ == "__main__":
    main()
