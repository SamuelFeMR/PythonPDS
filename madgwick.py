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
    def __init__(self, beta=0.1, mag_mode="marg", mag_rate=0.05,
                 accel_tol=0.25, mag_tol=0.3):
        """
        beta:      ganho do filtro (rad/s). Maior -> segue mais o acelerômetro
                   (mais rápido, mais ruído). Menor -> confia mais no giroscópio.
        mag_mode:  como o magnetômetro entra no filtro:
                     "marg" -> Madgwick MARG original (gradiente normalizado).
                               Problema: com o sensor parado o acelerômetro já
                               está satisfeito, então QUALQUER discordância do
                               magnetômetro puxa o yaw com a força total (beta),
                               mesmo que seja um erro pequeno ou de calibração.
                     "yaw"  -> o acelerômetro corrige roll/pitch (Madgwick IMU) e
                               o magnetômetro corrige SÓ o yaw, de forma
                               proporcional ao erro e com taxa limitada
                               (mag_rate). Muito mais estável com mag imperfeito.
        mag_rate:  (modo "yaw") taxa da correção de yaw, em 1/s. Constante de
                   tempo = 1/mag_rate. 0 desliga o magnetômetro.
        accel_tol: tolerância relativa de |a| em torno de 1 g. Fora dela
                   (movimento forte, vibração) o acelerômetro é ignorado naquele
                   passo e o giroscópio assume. 0 desliga a rejeição.
        mag_tol:   tolerância relativa de |m| em relação à média recente. Fora
                   dela (perturbação magnética) o magnetômetro é ignorado.
                   0 desliga a rejeição.
        """
        self.beta = float(beta)
        self.mag_mode = mag_mode
        self.mag_rate = float(mag_rate)
        self.accel_tol = float(accel_tol)
        self.mag_tol = float(mag_tol)
        self.q = None  # [w, x, y, z], sensor -> navegação (Z para cima)
        self._m_ref = None   # média móvel de |m| (para rejeitar perturbações)

    # ------------------------------------------------------------------
    def _accel_ok(self, a_norm):
        if a_norm < 1e-9:
            return False
        if self.accel_tol <= 0:
            return True
        ref = 1.0 if a_norm < 4.0 else 9.80665   # acelerômetro em g ou em m/s²
        return abs(a_norm / ref - 1.0) <= self.accel_tol

    def _mag_ok(self, m_norm):
        if m_norm < 1e-9:
            return False
        if self._m_ref is None:
            self._m_ref = m_norm
        ok = self.mag_tol <= 0 or abs(m_norm / self._m_ref - 1.0) <= self.mag_tol
        if ok:
            self._m_ref += 0.01 * (m_norm - self._m_ref)
        return ok

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
            self._m_ref = None
            return self.initialize(accel, mag)

        q = self.q

        # 1) Giroscópio
        q_dot = 0.5 * Q.multiply(q, [0.0, *gyro])

        a_norm = np.linalg.norm(accel)
        m_norm = np.linalg.norm(mag)
        use_a = self._accel_ok(a_norm)
        use_m = self._mag_ok(m_norm)

        # 2) Função objetivo e Jacobiana
        f_list, J_list = [], []

        if use_a:
            a = accel / a_norm
            g_ref = np.array([0.0, 0.0, 1.0])
            f_list.append(Q.rotate_inverse(q, g_ref) - a)
            J_list.append(Q.jacobian_of_rotate_inverse(q, g_ref))

            if use_m and self.mag_mode == "marg":
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
        q = Q.normalize(q + q_dot * dt)

        # 6) Modo "yaw": o magnetômetro corrige só o yaw, proporcional ao erro
        if self.mag_mode == "yaw" and use_m and self.mag_rate > 0 and dt > 0:
            q = self._yaw_correction(q, mag / m_norm, dt)

        self.q = q
        return self.q.copy()

    # ------------------------------------------------------------------
    def _yaw_correction(self, q, m, dt):
        """
        Gira a orientação em torno da vertical para alinhar o campo magnético
        medido ao norte (+X). O erro de yaw é o ângulo do campo, visto no
        referencial de navegação, em relação a +X. Roll e pitch não mudam.
        """
        h = Q.rotate(q, m)
        if np.hypot(h[0], h[1]) < 0.2:      # campo quase vertical: rumo indefinido
            return q
        err = np.arctan2(h[1], h[0])
        ang = -self.mag_rate * err * dt     # rotação em torno de Z (navegação)
        dq = np.array([np.cos(ang / 2), 0.0, 0.0, np.sin(ang / 2)])
        return Q.normalize(Q.multiply(dq, q))
