"""
pose_estimator.py

Estimador de orientação e posição relativa usando acelerômetro,
giroscópio e magnetômetro.

Base conceitual:
    Kok, M.; Hol, J. D.; Schön, T. B. (2017).
    "Using Inertial Sensors for Position and Orientation Estimation".

Implementa:
- Integração do giroscópio em quaternion.
- Correção complementar da orientação com acelerômetro e magnetômetro.
- Integração da aceleração para obter posição e velocidade relativas.

Convenções:
- Vetores dos sensores: [x, y, z].
- Acelerômetro: m/s² (força específica).
- Giroscópio: rad/s.
- Magnetômetro: qualquer unidade consistente (ex.: uT).
- Quaternion: [w, x, y, z].
- Referencial de navegação: Z positivo para cima.
- O quaternion representa a rotação do referencial do sensor (body)
  para o referencial de navegação (navigation).

Dependência:
    pip install numpy

ATENÇÃO:
A posição por dupla integração inercial deriva rapidamente. Sem uma
referência externa de posição, ela é apenas uma estimativa relativa.
"""

from __future__ import annotations

import numpy as np


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Não é possível normalizar um vetor de módulo zero.")
    return v / n


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    return _normalize(np.asarray(q, dtype=float))


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Produto de Hamilton. Quaternion no formato [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def _quat_from_rotation_vector(rotation_vector: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(rotation_vector)

    if angle < 1e-10:
        return _quat_normalize(np.array([
            1.0,
            0.5 * rotation_vector[0],
            0.5 * rotation_vector[1],
            0.5 * rotation_vector[2]
        ]))

    axis = rotation_vector / angle
    half = 0.5 * angle
    return np.array([
        np.cos(half),
        *(axis * np.sin(half))
    ], dtype=float)


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Matriz R_nb: transforma vetores do sensor para navegação."""
    w, x, y, z = _quat_normalize(q)

    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]
    ], dtype=float)


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Interpolação esférica entre dois quaternions."""
    q0 = _quat_normalize(q0)
    q1 = _quat_normalize(q1)

    dot = float(np.dot(q0, q1))

    # q e -q representam a mesma orientação.
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    if dot > 0.9995:
        return _quat_normalize(q0 + t * (q1 - q0))

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    return (
        np.sin((1.0 - t) * theta) / sin_theta * q0
        + np.sin(t * theta) / sin_theta * q1
    )


def _matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Converte matriz de rotação em quaternion [w, x, y, z]."""
    trace = np.trace(R)

    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array([
            0.25 * s,
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s
        ])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array([
            (R[2, 1] - R[1, 2]) / s,
            0.25 * s,
            (R[0, 1] + R[1, 0]) / s,
            (R[0, 2] + R[2, 0]) / s
        ])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array([
            (R[0, 2] - R[2, 0]) / s,
            (R[0, 1] + R[1, 0]) / s,
            0.25 * s,
            (R[1, 2] + R[2, 1]) / s
        ])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array([
            (R[1, 0] - R[0, 1]) / s,
            (R[0, 2] + R[2, 0]) / s,
            (R[1, 2] + R[2, 1]) / s,
            0.25 * s
        ])

    return _quat_normalize(q)


def _orientation_from_accel_mag(
    accel: np.ndarray,
    mag: np.ndarray,
    magnetic_north: np.ndarray
) -> np.ndarray:
    """
    Estima orientação por uma construção TRIAD.

    O acelerômetro define a vertical e o magnetômetro define a direção
    horizontal de referência. magnetic_north é expresso no referencial
    de navegação.
    """
    up_body = _normalize(accel)

    # Projeta o campo magnético no plano horizontal do sensor.
    mag_horizontal_body = mag - np.dot(mag, up_body) * up_body
    mag_horizontal_body = _normalize(mag_horizontal_body)

    up_nav = np.array([0.0, 0.0, 1.0])
    north_nav = np.asarray(magnetic_north, dtype=float)
    north_nav = north_nav - np.dot(north_nav, up_nav) * up_nav
    north_nav = _normalize(north_nav)

    east_body = _normalize(np.cross(up_body, mag_horizontal_body))
    east_nav = _normalize(np.cross(up_nav, north_nav))

    # Bases ortonormais: colunas = norte, leste e vertical.
    B = np.column_stack((mag_horizontal_body, east_body, up_body))
    N = np.column_stack((north_nav, east_nav, up_nav))

    R_nb = N @ B.T
    return _matrix_to_quaternion(R_nb)


class PoseEstimator:
    """
    Fusão inercial para estimar pose relativa.

    Parâmetros
    ----------
    gyro_weight:
        Peso da predição do giroscópio na correção complementar.
        Valores próximos de 1 dão mais peso ao giroscópio.
        Valor inicial sugerido: 0.98.
    gravity:
        Módulo da gravidade local, em m/s².
    accel_tolerance:
        Tolerância relativa para aceitar o acelerômetro como referência
        de gravidade. Ex.: 0.20 = aceita |a| entre 0.8g e 1.2g.
    magnetic_north:
        Direção do norte magnético no referencial de navegação.
        Por padrão, o norte é o eixo +X.
    """

    def __init__(
        self,
        gyro_weight: float = 0.98,
        gravity: float = 9.80665,
        accel_tolerance: float = 0.20,
        magnetic_north=(1.0, 0.0, 0.0)
    ):
        if not 0.0 <= gyro_weight <= 1.0:
            raise ValueError("gyro_weight deve estar entre 0 e 1.")

        self.gyro_weight = float(gyro_weight)
        self.gravity = float(gravity)
        self.accel_tolerance = float(accel_tolerance)
        self.magnetic_north = _normalize(
            np.asarray(magnetic_north, dtype=float)
        )

        self.q = None
        self.position = np.zeros(3, dtype=float)
        self.velocity = np.zeros(3, dtype=float)
        self.acceleration_nav = np.zeros(3, dtype=float)

        self.gyro_bias = np.zeros(3, dtype=float)
        self.initialized = False
        self.previous_acceleration = None

    def calibrate_gyro(self, gyro_samples) -> np.ndarray:
        """
        Calcula o bias do giroscópio com o sensor completamente parado.

        gyro_samples:
            Array N x 3, em rad/s.

        Retorna o bias estimado em rad/s.
        """
        samples = np.asarray(gyro_samples, dtype=float)

        if samples.ndim != 2 or samples.shape[1] != 3 or len(samples) < 2:
            raise ValueError("Informe pelo menos duas amostras N x 3.")

        self.gyro_bias = np.mean(samples, axis=0)
        return self.gyro_bias.copy()

    def set_initial_position(self, position=(0.0, 0.0, 0.0),
                              velocity=(0.0, 0.0, 0.0)):
        """Define posição e velocidade iniciais no referencial de navegação."""
        self.position = np.asarray(position, dtype=float).reshape(3)
        self.velocity = np.asarray(velocity, dtype=float).reshape(3)
        self.previous_acceleration = None

    def reset(self):
        """Reinicia orientação, posição e velocidade."""
        self.q = None
        self.position[:] = 0.0
        self.velocity[:] = 0.0
        self.acceleration_nav[:] = 0.0
        self.previous_acceleration = None
        self.initialized = False

    def update(self, accel, gyro, mag, dt: float) -> dict:
        """
        Processa uma nova amostra dos três sensores.

        Parâmetros
        ----------
        accel: sequência de 3 valores, m/s²
        gyro: sequência de 3 valores, rad/s
        mag: sequência de 3 valores, unidade consistente
        dt: intervalo entre amostras, em segundos

        Retorna
        -------
        dict com quaternion, matriz de rotação, Euler, posição,
        velocidade e aceleração no referencial de navegação.
        """
        if dt <= 0:
            raise ValueError("dt deve ser positivo.")

        accel = np.asarray(accel, dtype=float).reshape(3)
        gyro = np.asarray(gyro, dtype=float).reshape(3)
        mag = np.asarray(mag, dtype=float).reshape(3)

        if not (np.all(np.isfinite(accel))
                and np.all(np.isfinite(gyro))
                and np.all(np.isfinite(mag))):
            raise ValueError("As leituras devem conter valores finitos.")

        gyro_corrected = gyro - self.gyro_bias

        # Inicialização a partir da vertical e do campo magnético.
        if not self.initialized:
            self.q = _orientation_from_accel_mag(
                accel, mag, self.magnetic_north
            )
            self.initialized = True

        else:
            # 1) Predição temporal: integração da velocidade angular.
            delta_q = _quat_from_rotation_vector(gyro_corrected * dt)
            q_predicted = _quat_normalize(
                _quat_multiply(self.q, delta_q)
            )

            # 2) Atualização complementar com acelerômetro e magnetômetro.
            accel_norm = np.linalg.norm(accel)
            accel_is_gravity_like = (
                abs(accel_norm - self.gravity)
                <= self.accel_tolerance * self.gravity
            )

            mag_is_valid = np.linalg.norm(mag) > 1e-9

            if accel_is_gravity_like and mag_is_valid:
                try:
                    q_reference = _orientation_from_accel_mag(
                        accel, mag, self.magnetic_north
                    )
                    self.q = _quat_slerp(
                        q_reference,
                        q_predicted,
                        self.gyro_weight
                    )
                except ValueError:
                    self.q = q_predicted
            else:
                # Durante acelerações lineares fortes, não se usa o
                # acelerômetro como se ele medisse apenas gravidade.
                self.q = q_predicted

        self.q = _quat_normalize(self.q)
        R_nb = _quat_to_matrix(self.q)

        # 3) Remoção da gravidade e transformação para navegação.
        # Convenção: Z de navegação positivo para cima.
        gravity_nav = np.array([0.0, 0.0, -self.gravity])
        self.acceleration_nav = R_nb @ accel + gravity_nav

        # 4) Integração da aceleração para velocidade e posição.
        # Integração trapezoidal da velocidade e da posição.
        if self.previous_acceleration is None:
            self.previous_acceleration = self.acceleration_nav.copy()

        acceleration_average = (
            self.previous_acceleration + self.acceleration_nav
        ) * 0.5

        old_velocity = self.velocity.copy()
        self.velocity += acceleration_average * dt
        self.position += 0.5 * (old_velocity + self.velocity) * dt

        self.previous_acceleration = self.acceleration_nav.copy()

        # Ângulos de Euler para visualização: roll, pitch, yaw.
        roll, pitch, yaw = self._to_euler(R_nb)

        return {
            "quaternion": self.q.copy(),       # [w, x, y, z]
            "rotation_matrix": R_nb.copy(),    # R_nb
            "euler_rad": np.array([roll, pitch, yaw]),
            "euler_deg": np.degrees([roll, pitch, yaw]),
            "position": self.position.copy(),  # metros, relativa
            "velocity": self.velocity.copy(),  # m/s
            "acceleration": self.acceleration_nav.copy(),  # m/s²
        }

    @staticmethod
    def _to_euler(R: np.ndarray):
        """Retorna roll, pitch e yaw (convenção ZYX)."""
        pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        return roll, pitch, yaw

    def get_pose(self) -> dict:
        """Retorna o estado atual sem processar uma nova amostra."""
        if not self.initialized:
            raise RuntimeError("O estimador ainda não foi inicializado.")

        R_nb = _quat_to_matrix(self.q)
        roll, pitch, yaw = self._to_euler(R_nb)

        return {
            "quaternion": self.q.copy(),
            "rotation_matrix": R_nb,
            "euler_rad": np.array([roll, pitch, yaw]),
            "euler_deg": np.degrees([roll, pitch, yaw]),
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "acceleration": self.acceleration_nav.copy(),
        }


if __name__ == "__main__":
    # Exemplo mínimo: sensor parado, com eixo Z apontando para cima.
    estimator = PoseEstimator(gyro_weight=0.98)

    # Calibração fictícia: substituir por amostras reais com o sensor parado.
    estimator.calibrate_gyro(np.zeros((200, 3)))

    accel = [0.0, 0.0, 9.80665]
    gyro = [0.0, 0.0, 0.0]
    mag = [1.0, 0.0, 0.0]

    for _ in range(100):
        pose = estimator.update(accel, gyro, mag, dt=0.01)

    print("Quaternion [w, x, y, z]:", pose["quaternion"])
    print("Euler [roll, pitch, yaw] (graus):", pose["euler_deg"])
    print("Posição relativa (m):", pose["position"])
