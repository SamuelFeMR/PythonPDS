"""
quaternion.py

Funções mínimas de quaternion, formato [w, x, y, z].

Convenção do projeto:
    q rotaciona vetores do referencial do SENSOR (body) para o
    referencial de NAVEGAÇÃO (Z para cima):   v_nav = q ⊗ v_body ⊗ q*
"""

import numpy as np


def normalize(q):
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("Módulo zero: não é possível normalizar.")
    return np.asarray(q, dtype=float) / n


def multiply(a, b):
    """Produto de Hamilton a ⊗ b."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def rotate(q, v):
    """sensor -> navegação:  q ⊗ v ⊗ q*"""
    return multiply(multiply(q, [0.0, *v]), conjugate(q))[1:]


def rotate_inverse(q, v):
    """navegação -> sensor:  q* ⊗ v ⊗ q"""
    return multiply(multiply(conjugate(q), [0.0, *v]), q)[1:]


def jacobian_of_rotate_inverse(q, d):
    """
    Jacobiana 3x4 de g(q) = q* ⊗ d ⊗ q em relação a (w, x, y, z).
    g é bilinear em q, então  dg/dq_i = conj(e_i) ⊗ d ⊗ q + q* ⊗ d ⊗ e_i.
    """
    d4 = np.array([0.0, *d])
    qc = conjugate(q)
    J = np.empty((3, 4))
    for i in range(4):
        e = np.zeros(4)
        e[i] = 1.0
        t1 = multiply(multiply(conjugate(e), d4), q)
        t2 = multiply(multiply(qc, d4), e)
        J[:, i] = (t1 + t2)[1:]
    return J


def to_matrix(q):
    """Matriz de rotação R (sensor -> navegação)."""
    w, x, y, z = normalize(q)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def from_matrix(R):
    """Matriz de rotação -> quaternion."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        q = [0.25*s, (R[2, 1]-R[1, 2])/s, (R[0, 2]-R[2, 0])/s, (R[1, 0]-R[0, 1])/s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = [(R[2, 1]-R[1, 2])/s, 0.25*s, (R[0, 1]+R[1, 0])/s, (R[0, 2]+R[2, 0])/s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = [(R[0, 2]-R[2, 0])/s, (R[0, 1]+R[1, 0])/s, 0.25*s, (R[1, 2]+R[2, 1])/s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = [(R[1, 0]-R[0, 1])/s, (R[0, 2]+R[2, 0])/s, (R[1, 2]+R[2, 1])/s, 0.25*s]
    return normalize(q)


def to_euler_deg(q):
    """
    Roll, pitch, yaw em graus (ZYX) — APENAS para exibição.
    O filtro trabalha só com quaternion, então o gimbal lock (pitch = ±90°)
    afeta somente estes números, nunca a estimativa.
    """
    w, x, y, z = normalize(q)
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.degrees([roll, pitch, yaw])
