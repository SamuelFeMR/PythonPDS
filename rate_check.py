"""
rate_check.py

Mede a taxa real da serial e mostra de onde vem o limite:
  - taxa gerada pelo ESP32 (pelo timestamp esp_timer_get_time)
  - taxa recebida no PC
  - linhas perdidas/inválidas
  - ocupação do link serial (bytes/s x baud)

Uso:
    python rate_check.py --port COM5 --baud 115200 --seconds 10
"""

import argparse
import time

import numpy as np

from sensor_io import N_FIELDS, open_port


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=10.0)
    args = ap.parse_args()

    ser = open_port(args.port, args.baud)
    ser.reset_input_buffer()

    ts, n_bytes, n_bad = [], 0, 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.seconds:
        raw = ser.readline()
        if not raw:
            continue
        n_bytes += len(raw)
        p = raw.decode("utf-8", errors="ignore").strip().split(",")
        try:
            if len(p) != N_FIELDS:
                raise ValueError
            ts.append(int(p[0]))
        except ValueError:
            n_bad += 1
    elapsed = time.perf_counter() - t0
    ser.close()

    if len(ts) < 3:
        print("Quase nenhuma linha válida recebida.")
        return

    t = np.array(ts) * 1e-6
    dt = np.diff(t)
    med = np.median(dt)
    gaps = int(np.sum(dt > 1.5 * med))
    lost_est = int(np.sum(np.round(dt[dt > 1.5 * med] / med) - 1))
    bytes_per_line = n_bytes / (len(ts) + n_bad)
    link_use = 100 * (n_bytes * 10 / elapsed) / args.baud

    print(f"Linhas válidas: {len(ts)}  inválidas: {n_bad}  em {elapsed:.1f} s")
    print(f"Taxa recebida no PC:       {len(ts)/elapsed:6.1f} Hz")
    print(f"Taxa gerada pelo ESP32:    {1/med:6.1f} Hz (mediana do dt = {med*1e3:.2f} ms)")
    print(f"dt: média {dt.mean()*1e3:.2f} ms, desvio {dt.std()*1e3:.2f} ms, "
          f"mín {dt.min()*1e3:.2f} ms, máx {dt.max()*1e3:.2f} ms")
    print(f"Saltos de timestamp: {gaps}  (~{lost_est} linhas perdidas)")
    print(f"Tamanho médio da linha: {bytes_per_line:.0f} bytes")
    print(f"Ocupação do link a {args.baud} baud: {link_use:.0f}%  "
          f"(limite teórico ~{args.baud/10/bytes_per_line:.0f} linhas/s)")


if __name__ == "__main__":
    main()
