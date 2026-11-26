"""Calibração do magnetômetro por elipsoide: recupera hard iron e soft iron simulados."""
import os
import tempfile

import numpy as np

import magcal

rng = np.random.default_rng(3)


def rotations(n):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


n = 3000
B = np.array([0.8, 0.0, -0.6]) * 48.0                                   # campo terrestre (módulo 48)
true = np.einsum("nij,j->ni", np.transpose(rotations(n), (0, 2, 1)), B)  # visto pelo sensor girando
S = np.array([[1.15, 0.08, -0.05], [0.08, 0.9, 0.04], [-0.05, 0.04, 1.05]])   # soft iron
h = np.array([22.0, -35.0, 14.0])                                       # hard iron
raw = true @ S.T + h + rng.normal(0, 0.4, (n, 3))

pts = magcal.thin(raw)
assert len(pts) < n
cal = magcal.fit(pts, "ellipsoid")
assert np.allclose(cal.center, h, atol=0.3), cal.center
r = np.linalg.norm((raw - cal.center) @ cal.W.T, axis=1)
assert r.std() / r.mean() < 0.012 < cal.std_before / 10, (r.std() / r.mean(), cal.std_before)
assert cal.coverage == 26

sph = magcal.fit(pts, "sphere")                    # só hard iron: acha o centro, mas o soft iron fica
assert np.allclose(sph.center, h, atol=0.5) and sph.std_after > cal.std_after

for bad in (raw[:20], np.tile(raw[:1], (500, 1))):                      # poucos pontos / sensor parado
    try:
        magcal.fit(bad)
        raise SystemExit("deveria falhar")
    except ValueError:
        pass
flat = raw.copy(); flat[:, 2] = 0.0                                     # só girou em torno de um eixo
assert magcal.coverage(flat) < 26
try:
    magcal.fit(magcal.thin(flat))
    raise SystemExit("plano não deveria virar elipsoide")
except ValueError:
    pass

os.chdir(tempfile.mkdtemp())                                            # salvar / carregar
magcal.APP_DIR = os.getcwd()                                            # não mexe nos arquivos do projeto
magcal.save(cal)
back, msg = magcal.load()
assert msg == "" and np.allclose(back.W, cal.W) and np.allclose(back.center, cal.center)
print("calibração do magnetômetro: OK")

# ---- no pipeline: captura -> ajuste -> aplica; o filtro passa a receber o campo esférico
from pipeline import Pipeline
from sample import Sample

os.chdir(tempfile.mkdtemp())
magcal.APP_DIR = os.getcwd()


class Spy:
    name, params, description = "spy", (), ""
    values = {}

    def __init__(self):
        self.mags = []

    def reset(self):
        pass

    def update(self, inp):
        self.mags.append(inp.mag.copy())
        return np.array([1.0, 0, 0, 0])


pl = Pipeline()
pl.filter = Spy()
pl.mag_cal = None


def feed(i, m):
    pl.process(Sample(i * 20000, np.array([0, 0, 1.0]), np.zeros(3), m, 100, 100, 0, 0, 0, True, True, False))


pl.mag_capture_start()
for i, m in enumerate(raw):
    feed(i, m)
assert pl.mag_capture_stop() == n
cal = pl.fit_mag("ellipsoid")
pl.set_mag_cal(cal, save=True)
pl.filter = Spy()
for i, m in enumerate(raw[:500]):
    feed(i, m)
norms = np.linalg.norm(np.array(pl.filter.mags), axis=1)
assert norms.std() / norms.mean() < 0.012, norms.std() / norms.mean()
assert magcal.load()[0] is not None
pl.set_mag_cal(None, save=True)
assert magcal.load()[0] is None
print("pipeline com calibração do mag: OK")
