"""Os 3 métodos de orientação seguem a verdade simulada; o giroscópio puro deriva."""
import numpy as np

import quaternion as Q
from filters import MODES, FilterInput, make_filter


def sim(name, bias, T=60.0):
    f = make_filter(name)
    qt = Q.normalize(np.array([1, 0.2, -0.1, 0.3]))
    dt, err = 0.02, []
    for i in range(int(T / dt)):
        w = np.radians([20 * np.sin(i * dt), 15 * np.cos(i * dt * 0.7), 10])
        qt = Q.normalize(qt + 0.5 * Q.multiply(qt, [0, *w]) * dt)
        a, m = Q.rotate_inverse(qt, [0, 0, 1.0]), Q.rotate_inverse(qt, [0.8, 0, -0.5])
        q = f.update(FilterInput(a, w + bias, m, dt if i else 0.0))
        err.append(np.degrees(2 * np.arccos(min(1.0, abs(np.dot(q, qt))))))
    return np.array(err)


if __name__ == "__main__":
    names = [n for _, n in MODES]
    for n in names:
        assert sim(n, np.zeros(3)).max() < 1.0, n          # sensor ideal: todos acertam
    b = np.radians([0.4, -0.3, 0.5])
    e = {n: sim(n, b)[-500:].mean() for n in names}
    assert e[names[0]] < e[names[2]] and e[names[1]] < e[names[2]], e   # com bias: só o giro deriva
    print("métodos de orientação:", {n: round(v, 1) for n, v in e.items()}, "\nOK")
