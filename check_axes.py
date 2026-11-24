"""
check_axes.py

Descobre, a partir de um movimento real, como os eixos do giroscópio e do
magnetômetro se relacionam com os do acelerômetro, e calibra o magnetômetro
(hard-iron / soft-iron simples). Gera `axes_config.json`.

Princípio: um vetor fixo no mundo (gravidade, campo magnético), visto no
referencial do sensor, obedece
        d(v)/dt = v x w          (w = velocidade angular no referencial do sensor)
Então, se o gyro (ou o mag) estiver com eixos trocados/invertidos em relação ao
acelerômetro, essa equação NÃO fecha. Testamos as 48 permutações com sinal de
eixos e escolhemos a que fecha melhor. Também estimamos a escala do gyro
(esperado 1.0 se for mesmo graus/s).

Como usar:
  1) Segure o sensor PARADO por ~3 s (início da gravação).
  2) Depois gire-o LENTAMENTE em todas as direções, tombando em todos os
     eixos (como desenhar uma esfera no ar), por ~40 s. Evite sacudir.

  python check_axes.py --port COM5 --seconds 45
  python check_axes.py --csv gravacao.csv         # resolve de um arquivo
"""

import argparse
import itertools
import json
import time

import numpy as np

from sensor_io import DEG2RAD, open_port, parse_line, sample_stream

LAG = 5          # janela (amostras) para a derivada
MIN_RATE = 0.25  # rad/s: só usa janelas com rotação relevante (~14 °/s)


def signed_perms():
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            P = np.zeros((3, 3))
            for i in range(3):
                P[i, perm[i]] = signs[i]
            mats.append(P)
    return mats


def unit(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def window_mean(X, lag):
    c = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, axis=0)])
    return (c[lag + 1:] - c[:-lag - 1]) / (lag + 1)  # média de k..k+lag


def derivative(t, X, lag):
    dt = t[lag:] - t[:-lag]
    return (X[lag:] - X[:-lag]) / dt[:, None], dt


def rank_candidates(y, build_w, perms, fit_scale):
    """Para cada P, calcula o resíduo de y ≈ s * w(P). Retorna lista ordenada."""
    out = []
    for P in perms:
        w = build_w(P)
        ww = np.sum(w * w)
        if ww < 1e-12:
            continue
        s = np.sum(y * w) / ww if fit_scale else 1.0
        if fit_scale and s <= 0:
            continue
        res = np.sum((y - s * w) ** 2)
        out.append((res, s, P))
    out.sort(key=lambda r: r[0])
    return out


def solve(t, acc, gyr_raw, mag_raw, still_s):
    perms = signed_perms()

    # --- parte parada: bias do gyro ---
    n_still = int(np.searchsorted(t - t[0], still_s))
    if n_still < 20:
        raise SystemExit("Poucas amostras na fase parada; aumente --still ou grave mais.")
    g_rad = gyr_raw * DEG2RAD
    bias = g_rad[:n_still].mean(axis=0)
    print(f"Bias do gyro (parado): {bias} rad/s")
    still_motion = np.linalg.norm(g_rad[:n_still] - bias, axis=1).max()
    if still_motion > 0.15:
        print("AVISO: o sensor não parece ter ficado parado nos primeiros "
              f"{still_s:.0f} s (gyro variou {np.degrees(still_motion):.1f} °/s).")

    t, acc, g_rad, mag_raw = (x[n_still:] for x in (t, acc, g_rad, mag_raw))
    g_rad = g_rad - bias

    # Descarta trechos com aceleração linear forte (|a| longe do mediano)
    an = np.linalg.norm(acc, axis=1)
    keep_acc = np.abs(an / np.median(an) - 1.0) < 0.08

    # ------------------------------------------------------------ GYRO
    a_u = unit(acc)
    y_a, dt = derivative(t, a_u, LAG)
    a_mid = window_mean(a_u, LAG)
    g_mid = window_mean(g_rad, LAG)
    ok = ((dt < LAG * 0.05) & (np.linalg.norm(g_mid, axis=1) > MIN_RATE)
          & window_mean(keep_acc[:, None].astype(float), LAG)[:, 0].round().astype(bool))
    print(f"Janelas úteis para o gyro: {ok.sum()}")
    if ok.sum() < 100:
        raise SystemExit("Movimento insuficiente. Grave mais, girando mais nos 3 eixos.")

    y, am, gm = y_a[ok], a_mid[ok], g_mid[ok]

    # da/dt = a x w  (a, w no referencial do sensor)
    ranked = rank_candidates(y, lambda P: np.cross(am, gm @ P.T), perms, True)
    best_res, s, Pg = ranked[0]
    ratio = ranked[1][0] / best_res
    tot = np.sum(y * y)
    print("\n== GIROSCÓPIO -> eixos do acelerômetro ==")
    print(Pg.astype(int))
    print(f"escala estimada: {s:.3f}  (1.0 = graus/s correto; 0.5 = valor lido é 2x maior que "
          f"o assumido, etc.)")
    print(f"resíduo: {100*best_res/tot:.1f}% da energia; 2º melhor é {ratio:.1f}x pior")
    if ratio < 1.3:
        print("AVISO: resultado ambíguo. Faça um movimento mais variado (tombar em todos os eixos).")
    if np.linalg.det(Pg) < 0:
        print("AVISO: determinante -1 (eixo espelhado). Confira o hardware/orientação do chip.")
    G_total = s * Pg * DEG2RAD       # atua sobre o valor BRUTO (graus/s)
    w_acc = (g_rad @ (s * Pg).T)     # gyro no referencial do acel, rad/s

    # ------------------------------------------------------------- MAG
    mmin, mmax = mag_raw.min(axis=0), mag_raw.max(axis=0)
    offset = (mmax + mmin) / 2
    half = (mmax - mmin) / 2
    scale = half.mean() / half
    print("\n== MAGNETÔMETRO ==")
    print(f"faixa por eixo (bruto): min={mmin}  max={mmax}")
    print(f"offset (hard-iron): {offset}")
    print(f"escala por eixo (soft-iron simples): {scale}")
    if half.min() < 0.5 * half.max():
        print("AVISO: um eixo cobriu bem menos faixa que outro; gire mais nesse eixo "
              "para o offset/escala ficarem bons.")

    m_cal = (mag_raw - offset) * scale
    m_u = unit(m_cal)
    y_m, _ = derivative(t, m_u, LAG)
    w_mid = window_mean(w_acc, LAG)
    ym, wm = y_m[ok], w_mid[ok]
    mm_list = []
    for P in perms:
        mp = m_u @ P.T
        mm_list.append(window_mean(mp, LAG)[ok])
    ranked_m = []
    # O campo magnético é um vetor axial: a equação d(m)/dt = m x w não distingue
    # P de -P. Só as 24 rotações PRÓPRIAS (det = +1) são fisicamente possíveis
    # para um chip com eixos destros, o que resolve o sinal.
    for P, mid in zip(perms, mm_list):
        if np.linalg.det(P) < 0:
            continue
        # d(m')/dt = m' x w ; m'(t) = P m(t) => derivada = P * y_m
        pred = np.cross(mid, wm)
        res = np.sum((ym @ P.T - pred) ** 2)
        ranked_m.append((res, P))
    ranked_m.sort(key=lambda r: r[0])
    mres, Pm = ranked_m[0]
    mratio = ranked_m[1][0] / mres
    mtot = np.sum(ym * ym)
    print("mag -> eixos do acelerômetro:")
    print(Pm.astype(int))
    print(f"resíduo: {100*mres/mtot:.1f}% da energia; 2º melhor é {mratio:.1f}x pior")
    if mratio < 1.3:
        print("AVISO: resultado ambíguo para o mag.")

    # Qualidade: ângulo entre gravidade e campo deve ser constante
    dots = np.sum(a_u * (m_u @ Pm.T), axis=1)
    ang = np.degrees(np.arccos(np.clip(dots, -1, 1)))
    print(f"\nÂngulo gravidade x campo magnético: média {ang.mean():.1f}°, "
          f"desvio {ang.std():.1f}°")
    print("  (desvio baixo, < ~5°, = mag calibrado e alinhado; alto = ainda há problema)")

    return {
        "gyro_matrix": G_total.tolist(),
        "mag_matrix": Pm.tolist(),
        "mag_offset": offset.tolist(),
        "mag_scale": scale.tolist(),
    }


def record(port, baud, seconds):
    ser = open_port(port, baud)
    lines = []
    print("Segure PARADO por ~3 s; depois gire lentamente em todas as direções.")
    t_end = time.time() + seconds
    try:
        for s in sample_stream(ser):
            lines.append(s)
            if time.time() > t_end:
                break
    finally:
        ser.close()
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=45)
    ap.add_argument("--still", type=float, default=3.0, help="segundos parado no início")
    ap.add_argument("--csv", help="resolver a partir de arquivo gravado")
    ap.add_argument("--save-csv", default="gravacao.csv")
    ap.add_argument("--out", default="axes_config.json")
    args = ap.parse_args()

    if args.csv:
        samples = []
        with open(args.csv) as f:
            for line in f:
                s = parse_line(line)
                if s:
                    samples.append(s)
    else:
        if not args.port:
            ap.error("informe --port ou --csv")
        samples = record(args.port, args.baud, args.seconds)
        with open(args.save_csv, "w") as f:
            for t_us, a, g, m, ok in samples:
                f.write(f"{t_us},{a[0]},{a[1]},{a[2]},{g[0]},{g[1]},{g[2]},"
                        f"{m[0]},{m[1]},{m[2]},0,0,0,0,0,{int(ok)},1,1\n")
        print(f"Gravado em {args.save_csv} ({len(samples)} amostras)")

    samples = [s for s in samples if s[4]]
    t = np.array([s[0] for s in samples]) * 1e-6
    acc = np.array([s[1] for s in samples])
    gyr = np.array([s[2] for s in samples])
    mag = np.array([s[3] for s in samples])
    print(f"{len(t)} amostras, {t[-1]-t[0]:.1f} s, taxa ~{len(t)/(t[-1]-t[0]):.0f} Hz")

    cfg = solve(t, acc, gyr, mag, args.still)
    with open(args.out, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nConfiguração salva em {args.out}")


if __name__ == "__main__":
    main()
