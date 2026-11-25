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
import sphere_color as C
import quaternion as Q
import test_geometry as T


def random_quat(rnd):
    q = np.array([rnd.gauss(0, 1) for _ in range(4)])
    return q / np.linalg.norm(q)


def main():
    rnd = random.Random(3)
    names = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
    pairs = [(b, t) for b in names for t in names if b[-1] != t[-1]]
    n = worst = worst_yaw = worst_pose = worst_surf = worst_ray = worst_H = 0
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

        # o yaw só gira o ponto em torno da vertical: o polar não muda
        pos = P.compute(0.0, q, res.alpha_deg, bar_name, tow_name)
        a = rnd.uniform(0, 2 * math.pi)
        qz = np.array([math.cos(a / 2), 0, 0, math.sin(a / 2)])
        pos2 = P.compute(0.0, Q.multiply(qz, q), res.alpha_deg, bar_name, tow_name)
        worst_yaw = max(worst_yaw, abs(pos.polar_deg - pos2.polar_deg))

        # POSE COMPLETA: com a normal do IMU (sem cor) a placa volta à pose verdadeira, e com
        # qualquer longitude da cor os pontos focais continuam NA SUPERFÍCIE e os sensores fora
        k = 1.0 / Rs
        pts = P.scene_points(res, L, Rs, pos.display_rotation(R), bar, tow, pos.normal, 1.0)
        worst_pose = max(worst_pose, float(np.linalg.norm(pts["s1"] - origin * k)),
                         float(np.linalg.norm(pts["s2"] - (origin + R @ bar * L) * k)))
        lon = rnd.uniform(0, 360)
        pc = P.with_longitude(pos, lon, "cor")
        pts = P.scene_points(res, L, Rs, pc.display_rotation(R), bar, tow, pc.normal, 1.0)
        for key in ("h", "c", "d"):
            worst_surf = max(worst_surf, abs(np.linalg.norm(pts[key]) - 1.0))
        assert np.linalg.norm(pts["s1"]) > 1.0 and np.linalg.norm(pts["s2"]) > 1.0   # fora da esfera
        # a placa olha para o ponto: o raio do sensor 1 sai na direção (60° da barra, para a esfera)
        ray = (pts["c"] - pts["s1"]) / np.linalg.norm(pts["c"] - pts["s1"])
        want = pc.display_rotation(R) @ (0.5 * bar + math.sqrt(3) / 2 * tow)
        worst_ray = max(worst_ray, float(np.linalg.norm(ray - want)))
        # altura H: do pé de h na barra até h, ao longo da direção dos sensores (negativa se h
        # ficar atrás da linha da barra, o que acontece com esfera grande e barra curta)
        H_world = float((pts["h"] - pts["foot"]) @ (pc.display_rotation(R) @ tow)) * Rs   # com sinal
        worst_H = max(worst_H, abs(H_world - res.H))
        n += 1
    print(f"{n} poses aleatórias: normal {worst:.2e}, polar com yaw {worst_yaw:.2e}")
    print(f"pose completa: sensores {worst_pose:.2e}, c/d/h na superfície {worst_surf:.2e}, "
          f"raios {worst_ray:.2e}, H {worst_H:.2e}")
    assert max(worst_pose, worst_surf, worst_ray, worst_H) < 1e-9
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
    # sensores para cima (placa de cabeça para baixo): abaixo do equador, sem travar
    p = P.compute(0, np.array([0.0, 1, 0, 0]), 0.0)
    assert p.below_equator and abs(p.polar_deg - 180) < 1e-9

    # --- cor -> longitude (convenção do gomos_r9.py: matiz = lon/360, S = V = 1)
    import colorsys
    cfg = C.ColorConfig()
    worst_lon = 0.0
    for lon in range(0, 360, 3):
        rgb = colorsys.hsv_to_rgb(C.longitude_to_hue(lon), 1.0, 1.0)
        for scale in (80.0, 1300.0):                        # qualquer intensidade do sensor
            got, why = C.read_longitude(*(scale * c for c in rgb), cfg)
            assert got is not None, why
            worst_lon = max(worst_lon, abs((got - lon + 180) % 360 - 180))
    print(f"cor -> longitude: erro máx. {worst_lon:.2e}°")
    assert worst_lon < 1e-6
    assert C.read_longitude(10, 5, 5, cfg)[0] is None                 # sem sinal
    assert C.read_longitude(1000, 950, 900, cfg)[0] is None           # pouco saturada
    off = C.ColorConfig(hue_offset_deg=10.0, rgb_gain=(1.0, 2.0, 1.0))
    # (1000, 250, 0) com ganho 2 no G = (1000, 500, 0): matiz 30° -> longitude 30 + 10 de offset
    assert abs(C.read_longitude(1000, 250, 0, off)[0] - 40.0) < 1e-9
    assert abs(C.pole_cap_deg(90.0, cfg) - math.degrees(5 / 90.0)) < 1e-9  # 1 cm de calota em R = 9 cm

    # --- polo (orientação × α) + longitude (cor)
    p0 = P.compute(0, np.array([math.cos(math.radians(15)), 0, math.sin(math.radians(15)), 0]), 0.0)
    p1 = P.with_longitude(p0, 120.0, "cor")
    assert abs(p1.polar_deg - 30) < 1e-9 and abs(p1.azimuth_deg - 120) < 1e-9
    n = p1.normal
    assert abs(math.degrees(math.acos(n[2])) - 30) < 1e-9 and abs(math.degrees(math.atan2(n[1], n[0])) - 120) < 1e-9
    assert P.with_longitude(p0, None, "cor") is p0
    # o giro do yaw leva a normal do IMU até a longitude da cor
    assert abs(p0.azimuth_deg) < 1e-9 and abs(p1.yaw_shift_deg - 120) < 1e-9
    assert np.linalg.norm(P.rotate_z(p1.yaw_shift_deg) @ p0.normal - p1.normal) < 1e-12

    # --- fusão da longitude: orientação (curto prazo) + cor (referência absoluta)
    F = P.LongitudeFusion(rate=0.5, gate_deg=60.0)
    t = 0.0
    lon, src, d = F.update(t, 30.0, 40.0, None)                 # sem cor: segue o IMU
    assert src == "imu" and abs(lon - 30) < 1e-9
    lon, src, d = F.update(t := t + 0.02, 30.0, 40.0, 100.0)    # 1ª cor: alinha (Δ = 70)
    assert src == "fusão" and abs(d - 70) < 1e-9 and abs(lon - 100) < 1e-9
    for az in range(30, 330, 3):                                # anda pela esfera: cor acompanha
        lon, src, d = F.update(t := t + 0.02, float(az), 40.0, (az + 70.0) % 360)
        assert src == "fusão" and abs(d - 70) < 1e-6
    # o caso do teste: inclina para o lado OPOSTO (azimute do IMU +180°) com a cor parada
    az0 = 330.0
    lon0 = (az0 + 70) % 360
    lon, src, d = F.update(t := t + 0.02, (az0 + 180) % 360, 40.0, lon0)
    assert src == "discordante" and abs(P.wrap180(lon - (lon0 + 180))) < 1e-6   # foi para o outro lado
    for _ in range(500):                                        # e continua lá, com a cor parada
        lon, src, d = F.update(t := t + 0.02, (az0 + 180) % 360, 40.0, lon0)
    assert src == "discordante" and abs(P.wrap180(lon - (lon0 + 180))) < 1e-6
    # sem cor válida o Δ é mantido e o IMU conduz
    lon, src, d = F.update(t := t + 0.02, 100.0, 40.0, None)
    assert src == "imu" and abs(d - 70) < 1e-6 and abs(lon - 170) < 1e-6
    # deriva do yaw: Δ verdadeiro muda de 70 para 100 (dentro do gate) -> a cor corrige devagar
    for _ in range(1500):                                       # 30 s a 50 Hz
        lon, src, d = F.update(t := t + 0.02, 100.0, 40.0, 200.0)
    assert src == "fusão" and abs(d - 100) < 0.5, d
    # perto do polo a cor não conta
    F2 = P.LongitudeFusion()
    F2.update(0.0, 0.0, 40.0, 50.0)
    lon, src, d = F2.update(0.02, 10.0, 2.0, 300.0)
    assert src == "imu" and abs(d - 50) < 1e-9
    F2.reset()
    assert F2.delta is None
    print("fusão da longitude: OK")
    print("OK")


if __name__ == "__main__":
    main()
