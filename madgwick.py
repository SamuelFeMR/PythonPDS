"""
madgwick.py

Filtro de Madgwick (MARG) simples, só com quaternions.

Passos de cada update (Madgwick et al., 2011):
  0. Inicialização da atitude (primeira amostra) com acel + mag.
  1. Derivada do quaternion pelo giroscópio:  q_dot = 0.5 * q ⊗ [0, w]
  2. Função objetivo: erro entre o que o quaternion ATUAL prevê para a
     gravidade e para o campo magnético (no referencial do sensor) e o que
     os sensores MEDIRAM.
         f = [ q* ⊗ g_ref ⊗ q - a ,  q* ⊗ b_ref ⊗ q - m ]
  3. Gradiente  J^T f  e normalização do gradiente.
  4. Fusão:  q_dot = q_dot_gyro - beta * gradiente_normalizado
  5. Integração:  q = normalize(q + q_dot * dt)

Referência magnética (b): o campo medido é levado para navegação,
h = q ⊗ m ⊗ q*, e a referência fica  b = [|h_xy|, 0, h_z]  (norte em +X).
Isso elimina o efeito da inclinação magnética e de distorções que
apontariam o erro para roll/pitch.

Unidades: gyro em rad/s. Acel e mag podem estar em qualquer unidade
(são normalizados).
"""

import numpy as np

import quaternion as Q


class Madgwick:
    def __init__(self, beta=0.1):
        """
        beta: ganho do filtro (rad/s).
              Maior -> segue mais acel/mag (mais rápido, mais ruído).
              Menor -> confia mais no giroscópio (mais suave, mais deriva).
        """
        self.beta = float(beta)
        self.q = None  # [w, x, y, z], sensor -> navegação (Z para cima)

    # ------------------------------------------------------------------
    def initialize(self, accel, mag):
        """Atitude inicial por TRIAD (vertical do acel + horizontal do mag)."""
        up_b = Q.normalize(np.asarray(accel, dtype=float))

        m = np.asarray(mag, dtype=float)
        if np.linalg.norm(m) < 1e-9:
            # Sem magnetômetro: usa o eixo X do sensor como "norte" provisório
            # (ou o Y, se o X estiver alinhado com a vertical). Yaw inicial = 0.
            m = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(m, up_b)) > 0.99:
                m = np.array([0.0, 1.0, 0.0])
        north_b = Q.normalize(m - np.dot(m, up_b) * up_b)
        west_b = Q.normalize(np.cross(up_b, north_b))

        # Navegação: X = norte, Y = oeste, Z = cima (base destra: Y = Z x X).
        up_n = np.array([0.0, 0.0, 1.0])
        north_n = np.array([1.0, 0.0, 0.0])
        west_n = np.cross(up_n, north_n)

        B = np.column_stack((north_b, west_b, up_b))
        N = np.column_stack((north_n, west_n, up_n))
        self.q = Q.from_matrix(N @ B.T)
        return self.q.copy()

    # ------------------------------------------------------------------
    def update(self, gyro, accel, mag, dt):
        """Processa uma amostra. gyro em rad/s. Retorna o quaternion."""
        gyro = np.asarray(gyro, dtype=float)
        accel = np.asarray(accel, dtype=float)
        mag = np.asarray(mag, dtype=float)

        if self.q is None:
            return self.initialize(accel, mag)

        q = self.q

        # 1) Giroscópio
        q_dot = 0.5 * Q.multiply(q, [0.0, *gyro])

        # 2) Função objetivo e Jacobiana
        a_norm = np.linalg.norm(accel)
        m_norm = np.linalg.norm(mag)

        f_list, J_list = [], []

        if a_norm > 1e-9:
            a = accel / a_norm
            g_ref = np.array([0.0, 0.0, 1.0])
            f_list.append(Q.rotate_inverse(q, g_ref) - a)
            J_list.append(Q.jacobian_of_rotate_inverse(q, g_ref))

            if m_norm > 1e-9:
                m = mag / m_norm
                h = Q.rotate(q, m)                     # campo em navegação
                b_ref = np.array([np.hypot(h[0], h[1]), 0.0, h[2]])
                f_list.append(Q.rotate_inverse(q, b_ref) - m)
                J_list.append(Q.jacobian_of_rotate_inverse(q, b_ref))

        # 3) Gradiente normalizado e 4) fusão
        if f_list:
            f = np.concatenate(f_list)
            J = np.vstack(J_list)
            grad = J.T @ f
            n = np.linalg.norm(grad)
            if n > 1e-12:
                q_dot = q_dot - self.beta * grad / n

        # 5) Integração
        self.q = Q.normalize(q + q_dot * dt)
        return self.q.copy()
