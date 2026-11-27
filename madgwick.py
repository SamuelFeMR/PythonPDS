"""
madgwick.py

Filtro de Madgwick usando a biblioteca AHRS (https://github.com/Mayitzin/ahrs,
`pip install ahrs`). A biblioteca faz o passo do filtro (giroscópio + gradiente
descendente com acelerômetro, e magnetômetro no modo MARG); aqui ficam só:

  * a atitude inicial (TRIAD: vertical do acelerômetro + horizontal do mag);
  * a rejeição de medidas ruins (|a| longe de 1 g, |m| longe da média recente);
  * o modo "yaw": o magnetômetro corrige SÓ a guinada, devagar (mag_rate).

Convenções (iguais às da biblioteca): quaternion [w, x, y, z] do sensor para a
navegação, X = norte, Y = oeste, Z = para cima; gyro em rad/s; acel e mag em
qualquer unidade (a biblioteca normaliza).
"""

import numpy as np

import quaternion as Q

try:
    from ahrs.filters import Madgwick as _AhrsMadgwick
except ImportError as e:      # mensagem clara em vez de um traceback longo
    raise ImportError("Falta a biblioteca AHRS. Instale com:  pip install ahrs") from e


def triad(accel, mag):
    """Atitude a partir de uma amostra: vertical do acelerômetro + horizontal do mag."""
    up_b = Q.normalize(np.asarray(accel, dtype=float))
    m = np.asarray(mag, dtype=float)
    if np.linalg.norm(m) < 1e-9:
        # Sem magnetômetro: usa o eixo X do sensor como "norte" provisório
        # (ou o Y, se o X estiver alinhado com a vertical). Yaw inicial = 0.
        m = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(m, up_b)) > 0.99:
            m = np.array([0.0, 1.0, 0.0])
    north_b = Q.normalize(m - np.dot(m, up_b) * up_b)     # campo projetado no plano horizontal
    west_b = Q.normalize(np.cross(up_b, north_b))
    # Linhas = norte, oeste e vertical vistos pelo sensor  ->  matriz sensor -> navegação
    return Q.from_matrix(np.vstack((north_b, west_b, up_b)))


class Madgwick:
    def __init__(self, beta=0.1, mag_mode="marg", mag_rate=0.05,
                 accel_tol=0.25, mag_tol=0.3):
        """
        beta:      ganho do filtro (rad/s). Maior -> segue mais o acelerômetro
                   (mais rápido, mais ruído). Menor -> confia mais no giroscópio.
        mag_mode:  "marg" -> Madgwick MARG da biblioteca (mag no gradiente). Com o
                             sensor parado, qualquer discordância do mag puxa o yaw
                             com a força total (beta).
                   "yaw"  -> Madgwick IMU da biblioteca (acel corrige roll/pitch) e o
                             mag corrige SÓ o yaw, proporcional ao erro e com taxa
                             limitada (mag_rate). Mais estável com mag imperfeito.
        mag_rate:  (modo "yaw") taxa da correção de yaw, em 1/s. 0 desliga o mag.
        accel_tol: tolerância relativa de |a| em torno de 1 g; fora dela o acel é
                   ignorado naquele passo (só giroscópio). 0 desliga a rejeição.
        mag_tol:   tolerância relativa de |m| em relação à média recente; fora dela o
                   mag é ignorado. 0 desliga a rejeição.
        """
        self._lib = _AhrsMadgwick(gain=max(float(beta), 1e-9))   # a lib exige ganho > 0 ao criar
        self.beta = beta
        self.mag_mode = mag_mode
        self.mag_rate = float(mag_rate)
        self.accel_tol = float(accel_tol)
        self.mag_tol = float(mag_tol)
        self.q = None          # [w, x, y, z], sensor -> navegação (Z para cima)
        self._m_ref = None     # média móvel de |m| (para rejeitar perturbações)

    @property
    def beta(self):
        return self._lib.gain

    @beta.setter
    def beta(self, value):
        self._lib.gain = float(value)

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
        self.q = triad(accel, mag)
        return self.q.copy()

    def update(self, gyro, accel, mag, dt):
        """Processa uma amostra. gyro em rad/s. Retorna o quaternion."""
        gyro = np.asarray(gyro, dtype=float)
        accel = np.asarray(accel, dtype=float)
        mag = np.asarray(mag, dtype=float)

        if self.q is None:
            self._m_ref = None
            return self.initialize(accel, mag)

        use_a = self._accel_ok(np.linalg.norm(accel))
        m_norm = np.linalg.norm(mag)
        use_m = self._mag_ok(m_norm)
        acc = accel if use_a else np.zeros(3)     # vetor nulo = a biblioteca não corrige com o acel

        if self.mag_mode == "marg" and use_a and use_m:
            q = self._lib.updateMARG(self.q, gyr=gyro, acc=acc, mag=mag, dt=dt)
        else:
            q = self._lib.updateIMU(self.q, gyr=gyro, acc=acc, dt=dt)
        q = Q.normalize(np.asarray(q, dtype=float))

        if self.mag_mode == "yaw" and use_m and self.mag_rate > 0 and dt > 0:
            q = self._yaw_correction(q, mag / m_norm, dt)

        self.q = q
        return self.q.copy()

    # ------------------------------------------------------------------
    def _yaw_correction(self, q, m, dt):
        """
        Gira a orientação em torno da vertical para alinhar o campo magnético
        medido ao norte (+X). h = q ⊗ m ⊗ q* é o campo já sem a inclinação do
        sensor; o ângulo de h em relação a +X é o erro de guinada. Roll e pitch
        não mudam.
        """
        h = Q.rotate(q, m)
        if np.hypot(h[0], h[1]) < 0.2:      # campo quase vertical: rumo indefinido
            return q
        err = np.arctan2(h[1], h[0])
        ang = -self.mag_rate * err * dt     # rotação em torno de Z (navegação)
        dq = np.array([np.cos(ang / 2), 0.0, 0.0, np.sin(ang / 2)])
        return Q.normalize(Q.multiply(dq, q))
