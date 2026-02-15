"""Pose data structures for 6DoF head tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Pose6D:
    """Head pose in world space.

    position:
      3D translation [x, y, z], meters.
    quaternion:
      Orientation quaternion [w, x, y, z], unit length.
    """

    position: np.ndarray
    quaternion: np.ndarray


def identity_pose() -> Pose6D:
    return Pose6D(
        position=np.zeros(3, dtype=np.float64),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )

