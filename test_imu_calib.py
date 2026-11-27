"""Calibração mín/máx de acelerômetro e magnetômetro: recupera offset e escala simulados."""
import os
import tempfile

import numpy as np

import imu_calib

rng = np.random.default_rng(4)


def directions(n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ---- magnetômetro: girar em todas as direções
off_m, gain_m = np.array([22.0, -35.0, 14.0]), np.array([1.15, 0.9, 1.05])
raw_m = 48.0 * directions(4000) * gain_m + off_m + rng.normal(0, 0.3, (4000, 3))
cal = imu_calib.compute("mag", raw_m)
assert np.allclose(cal.offset, off_m, atol=1.0), cal.offset
r = np.linalg.norm(cal.apply(raw_m), axis=1)
before = np.std(np.linalg.norm(raw_m - raw_m.mean(0), axis=1)) / 48
assert r.std() / r.mean() < 0.02 < before, (r.std() / r.mean(), before)

# ---- acelerômetro: 6 faces + algumas posições intermediárias, parado (1 g)
off_a, gain_a = np.array([0.03, -0.05, 0.08]), np.array([1.02, 0.97, 1.04])
faces = np.vstack([np.eye(3), -np.eye(3)])
dirs = np.vstack([np.repeat(faces, 40, axis=0), directions(200)])
raw_a = dirs * gain_a + off_a + rng.normal(0, 0.003, dirs.shape)
cal = imu_calib.compute("accel", raw_a)
assert np.allclose(cal.offset, off_a, atol=0.01) and np.allclose(cal.scale, 1 / gain_a, atol=0.01), (cal.offset, cal.scale)
assert abs(np.linalg.norm(cal.apply(raw_a[:240]), axis=1).mean() - 1) < 0.01

for sensor, bad in (("mag", raw_m[:20]), ("accel", np.repeat(faces[:2], 40, axis=0)),
                    ("mag", raw_m * [1, 1, 0.1])):
    try:
        imu_calib.compute(sensor, bad)
        raise SystemExit(f"deveria recusar ({sensor})")
    except ValueError:
        pass

# ---- no pipeline: captura -> calcula -> aplica -> salva
os.chdir(tempfile.mkdtemp())
imu_calib.APP_DIR = os.getcwd()
from pipeline import Pipeline   # noqa: E402
from sample import Sample       # noqa: E402

pl = Pipeline()
pl.capture_start("accel")
for i, a in enumerate(raw_a):
    gyro = np.zeros(3) if i % 2 == 0 else np.array([30.0, 0, 0])   # metade em movimento: ignorada
    pl.process(Sample(i * 20000, a, gyro, np.array([1.0, 0, 0]), 1, 1, 0, 0, 0, True, True, False))
pl.capture_stop()
assert len(pl.capture_points()) == (len(raw_a) + 1) // 2
pl.set_imu_cal("accel", imu_calib.compute("accel", pl.capture_points()))
back, msg = imu_calib.load()
assert msg == "" and back["accel"] is not None and back["mag"] is None
pl.set_imu_cal("accel", None)
assert imu_calib.load()[0]["accel"] is None
print("calibração mín/máx de acel e mag: OK")
