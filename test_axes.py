"""Troca manual de eixos (axes_remap.json)."""
import json, os, tempfile

import numpy as np

from sensor_io import Frames, load_remap, remap_matrix

v = np.array([1.0, 2.0, 3.0])
assert np.allclose(remap_matrix(["+Y", "+X", "-Z"]) @ v, [2, 1, -3])
assert np.allclose(remap_matrix(["+X", "+Y", "+Z"]) @ v, v)
for bad in (["+X", "+X", "+Z"], ["+X", "+Y"], ["+X", "+Y", "W"]):
    try:
        remap_matrix(bad)
        raise SystemExit(f"deveria rejeitar {bad}")
    except ValueError:
        pass

d = tempfile.mkdtemp()
p = os.path.join(d, "axes_remap.json")
json.dump({"accel": ["+Y", "+X", "+Z"], "gyro": ["-X", "+Y", "+Z"]}, open(p, "w"))   # mag omitido = sem troca
m, msg = load_remap(p)
f = Frames(None, remap=m)
assert msg == "" and np.allclose(f.accel(v), [2, 1, 3]) and np.allclose(f.gyro(v), np.array([-1, 2, 3]) * np.pi / 180)
assert np.allclose(f.mag(v), v)
open(p, "w").write("{ruim")
m, msg = load_remap(p)
assert msg and np.allclose(Frames(None, remap=m).accel(v), v)
print("troca de eixos: OK")
