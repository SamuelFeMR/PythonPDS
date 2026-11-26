"""
run_serial.py

Lê o CSV do ESP32 pela serial e roda o filtro de Madgwick.

Se existir `axes_config.json` (gerado por check_axes.py), ele é aplicado:
eixos do gyro e do mag alinhados ao acelerômetro + calibração do mag.

Uso:
    python run_serial.py --port COM5
    python run_serial.py --port COM5 --no-mag      # só acel+gyro (isola o mag)
    python run_serial.py --port COM5 --beta 0.05 --calib 0

Dependências:  pip install numpy pyserial
"""

import argparse

import numpy as np

import quaternion as Q
from madgwick import Madgwick
from sensor_io import Frames, load_config, open_port, sample_stream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--calib", type=int, default=200,
                    help="amostras (sensor PARADO) para bias do gyro; 0 pula")
    ap.add_argument("--every", type=int, default=10, help="imprime a cada N amostras")
    ap.add_argument("--no-mag", action="store_true",
                    help="ignora o magnetômetro (yaw vai derivar, roll/pitch ficam corretos)")
    ap.add_argument("--config", default="axes_config.json")
    ap.add_argument("--gyro-rad", action="store_true", help="gyro já em rad/s (sem config)")
    args = ap.parse_args()

    frames = Frames(load_config(args.config), gyro_in_rad=args.gyro_rad)
    print("Alinhamento de eixos:",
          "carregado de " + args.config if frames.from_file
          else "NENHUM (rode check_axes.py para gerar axes_config.json)")
    if args.no_mag:
        print("Modo sem magnetômetro (6 eixos).")

    ser = open_port(args.port, args.baud)
    stream = sample_stream(ser)
    filt = Madgwick(beta=args.beta)
    gyro_bias = np.zeros(3)

    try:
        if args.calib > 0:
            print(f"Calibrando gyro: mantenha o sensor PARADO ({args.calib} amostras)...")
            buf = []
            for _, _, g, _, ok in stream:
                if ok:
                    buf.append(frames.gyro(g))
                if len(buf) >= args.calib:
                    break
            gyro_bias = np.mean(buf, axis=0)
            print("Bias do gyro (rad/s):", gyro_bias)

        print("Rodando. Ctrl+C para parar.")
        prev_t = None
        n = 0

        for t_us, accel, g_raw, m_raw, ok in stream:
            if not ok:
                prev_t = None
                continue

            dt = 0.0 if prev_t is None else (t_us - prev_t) * 1e-6
            prev_t = t_us
            if dt < 0 or dt > 0.5:
                continue

            gyro = frames.gyro(g_raw) - gyro_bias
            mag = np.zeros(3) if args.no_mag else frames.mag(m_raw)

            q = filt.update(gyro, accel, mag, dt)

            n += 1
            if n % args.every == 0:
                r, p, y = Q.to_euler_deg(q)
                tilt = np.degrees(np.arccos(np.clip(Q.to_matrix(q)[2, 2], -1, 1)))
                print(f"t={t_us/1e6:8.3f}s  roll={r:7.2f}  pitch={p:7.2f}  yaw={y:7.2f}  "
                      f"tilt={tilt:6.2f}  |a|={np.linalg.norm(accel):.2f}")
    except KeyboardInterrupt:
        print("\nEncerrado.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
