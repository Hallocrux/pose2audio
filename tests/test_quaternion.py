import numpy as np

from math3d.quaternion import (
    euler_yaw_pitch_roll_to_q,
    q_normalize,
    q_rotate_vec,
    rotmat_to_q,
)


def test_q_normalize_zero_returns_identity():
    q = q_normalize(np.zeros(4, dtype=np.float64))
    np.testing.assert_allclose(q, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))


def test_yaw_90_rotates_forward_to_right():
    q = euler_yaw_pitch_roll_to_q(90.0, 0.0, 0.0)
    v = q_rotate_vec(q, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    np.testing.assert_allclose(v, np.array([1.0, 0.0, 0.0], dtype=np.float64), atol=1e-6)


def test_rotmat_identity_to_quaternion():
    q = rotmat_to_q(np.eye(3, dtype=np.float64))
    np.testing.assert_allclose(q, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), atol=1e-8)

