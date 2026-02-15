"""Quaternion utilities for right-handed coordinates.
"""

from __future__ import annotations

import math

import numpy as np


def q_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def q_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def q_rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by unit quaternion q: v' = q*(0,v)*q^{-1}."""
    q = q_normalize(q)
    vq = np.array([0.0, float(v[0]), float(v[1]), float(v[2])], dtype=np.float64)
    return q_mul(q_mul(q, vq), q_conj(q))[1:]


def axis_angle_to_q(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    s = math.sin(angle_rad / 2.0)
    return q_normalize(
        np.array(
            [math.cos(angle_rad / 2.0), axis[0] * s, axis[1] * s, axis[2] * s],
            dtype=np.float64,
        )
    )


def euler_yaw_pitch_roll_to_q(
    yaw_deg: float, pitch_deg: float, roll_deg: float
) -> np.ndarray:
    """
    Right-handed coordinates:
      x: right, y: up, z: forward
    Euler:
      yaw around +y, pitch around +x, roll around +z
    Composition: q = q_yaw * q_pitch * q_roll
    """
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    q_yaw = axis_angle_to_q(np.array([0.0, 1.0, 0.0]), yaw)
    q_pitch = axis_angle_to_q(np.array([1.0, 0.0, 0.0]), pitch)
    q_roll = axis_angle_to_q(np.array([0.0, 0.0, 1.0]), roll)
    return q_normalize(q_mul(q_mul(q_yaw, q_pitch), q_roll))


def rotmat_to_q(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to unit quaternion [w, x, y, z]."""
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"Expected 3x3 rotation matrix, got {R.shape}")

    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    return q_normalize(np.array([qw, qx, qy, qz], dtype=np.float64))
