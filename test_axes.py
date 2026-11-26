"""Troca de eixos / sentido do giro (imu_axes) e o indicador giro × acelerômetro."""
import numpy as np

import imu_axes
import quaternion as Q
from pipeline import Pipeline
from sample import Sample

v = np.array([1.0, 2.0, 3.0])
assert np.allclose(imu_axes.matrix(["+Y", "+X", "-Z"]) @ v, [2, 1, -3])
for bad in (["+X", "+X", "+Z"], ["+X", "+Y"], ["+X", "+Y", "W"]):
    try:
        imu_axes.matrix(bad)
        raise SystemExit(f"deveria rejeitar {bad}")
    except ValueError:
        pass


def run(firmware, names=("+X", "+Y", "+Z"), invert=False, T=20.0):
    """Sensor real girando; `firmware` embaralha os eixos como o ESP32 faz. Retorna (r, erro de tilt)."""
    pl = Pipeline()
    pl.imu_axes = imu_axes.ImuAxes(tuple(names), invert)
    qt = Q.normalize(np.array([1, 0.2, -0.1, 0.3]))
    dt, err = 0.02, []
    for i in range(int(T / dt)):
        w = np.array([0.6 * np.sin(i * dt), 0.5 * np.cos(i * dt * 0.7), 0.3])        # rad/s
        qt = Q.normalize(qt + 0.5 * Q.multiply(qt, [0, *w]) * dt)
        a, m = Q.rotate_inverse(qt, [0, 0, 1.0]), Q.rotate_inverse(qt, [0.8, 0, -0.5])
        s = Sample(int(i * dt * 1e6), firmware(a), firmware(np.degrees(w)), firmware(m),
                   100, 100, 0, 0, 0, True, True, False)
        pose = pl.process(s)
        az = (imu_axes.matrix(names) @ firmware(a))[2]      # inclinação do Z (já trocado) em relação à vertical
        err.append(abs(pose.tilt_deg - np.degrees(np.arccos(np.clip(az, -1, 1)))))
    return pl.gyro_consistency(), float(np.mean(err[-250:]))


ident = lambda x: x
swap = lambda x: np.array([x[0], x[2], x[1]])        # o printf do ESP32: (x, z, y) — sistema espelhado

r, e = run(ident)
assert r > 0.8 and e < 3, (r, e)                                  # eixos corretos: concordam
r, e_bad = run(swap)
assert r < -0.8, r                                                 # troca sem sinal: giro invertido
r, e = run(swap, invert=True)
assert r > 0.8 and e < 3 and e_bad > 2 * e, (r, e, e_bad)          # inverter o giro: concordam (mundo espelhado)
r, e = run(swap, names=("+X", "+Y", "-Z"))
assert r > 0.8 and e < 3, (r, e)                                   # inverter o sinal de um eixo: fiel ao real
print("eixos / sentido do giro: OK")
