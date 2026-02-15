"""Head-relative native direction providers.

Convention note:
- This module estimates direction in each pose provider's native head frame.
- Acoustic semantic mapping (for example front/back flip) is handled later by
  ``control.acoustic_frame_provider`` to keep responsibilities decoupled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..math3d.coords import vec_to_az_el_deg
from ..math3d.quaternion import q_conj, q_rotate_vec
from .pose import Pose6D


def head_relative_source_vector(
    source_world: np.ndarray,
    pose: Pose6D,
    fallback_world_dir: np.ndarray,
) -> np.ndarray:
    """Source direction in listener/head frame."""
    rel_world = source_world - pose.position
    if float(np.dot(rel_world, rel_world)) < 1e-12:
        rel_world = fallback_world_dir
    return q_rotate_vec(q_conj(pose.quaternion), rel_world)


def _normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n2 = float(np.dot(v, v))
    if n2 < 1e-12:
        return fallback.copy()
    return v / math.sqrt(n2)


def _angle_diff_deg(a: float, b: float) -> float:
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


@dataclass(slots=True)
class DirectionSample:
    s_head: np.ndarray
    rel_az: float
    rel_el: float


class DirectionProvider:
    """Base interface for direction estimation before HRTF selection."""

    def compute(self, source_world: np.ndarray, pose: Pose6D) -> DirectionSample:
        raise NotImplementedError


class RawDirectionProvider(DirectionProvider):
    """No stabilization; keeps existing behavior."""

    def __init__(self, initial_source_world: np.ndarray):
        seed = np.asarray(initial_source_world, dtype=np.float64).reshape(3)
        if float(np.dot(seed, seed)) < 1e-12:
            seed = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self._last_rel_world = seed

    def compute(self, source_world: np.ndarray, pose: Pose6D) -> DirectionSample:
        source_world = np.asarray(source_world, dtype=np.float64).reshape(3)
        rel_world = source_world - pose.position
        if float(np.dot(rel_world, rel_world)) >= 1e-12:
            self._last_rel_world = rel_world
        s_head = head_relative_source_vector(source_world, pose, self._last_rel_world)
        rel_az, rel_el = vec_to_az_el_deg(s_head)
        return DirectionSample(s_head=s_head, rel_az=rel_az, rel_el=rel_el)


class StabilizedDirectionProvider(DirectionProvider):
    """Stabilized direction with distance gating + vector LPF + angular deadband."""

    def __init__(
        self,
        initial_source_world: np.ndarray,
        min_distance_m: float = 0.15,
        smoothing: float = 0.20,
        deadband_deg: float = 1.0,
    ):
        seed = np.asarray(initial_source_world, dtype=np.float64).reshape(3)
        if float(np.dot(seed, seed)) < 1e-12:
            seed = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self._last_rel_world = seed
        self._min_distance2 = float(max(0.0, min_distance_m) ** 2)
        self._alpha = float(max(0.01, min(1.0, smoothing)))
        self._deadband_deg = float(max(0.0, deadband_deg))
        self._prev_unit: np.ndarray | None = None
        self._prev_az: float | None = None
        self._prev_el: float | None = None

    def compute(self, source_world: np.ndarray, pose: Pose6D) -> DirectionSample:
        source_world = np.asarray(source_world, dtype=np.float64).reshape(3)
        rel_world = source_world - pose.position
        rel2 = float(np.dot(rel_world, rel_world))
        update_threshold2 = max(1e-12, self._min_distance2)
        if rel2 >= update_threshold2:
            self._last_rel_world = rel_world
        rel_for_dir = self._last_rel_world if rel2 < self._min_distance2 else rel_world

        raw_s_head = q_rotate_vec(q_conj(pose.quaternion), rel_for_dir)
        raw_unit = _normalize(raw_s_head, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        if self._prev_unit is None:
            smoothed_unit = raw_unit
        else:
            smoothed_unit = _normalize(
                (1.0 - self._alpha) * self._prev_unit + self._alpha * raw_unit,
                self._prev_unit,
            )
        self._prev_unit = smoothed_unit

        rel_az, rel_el = vec_to_az_el_deg(smoothed_unit)
        if self._prev_az is not None and self._prev_el is not None and self._deadband_deg > 0.0:
            da = abs(_angle_diff_deg(rel_az, self._prev_az))
            de = abs(rel_el - self._prev_el)
            if da < self._deadband_deg and de < self._deadband_deg:
                rel_az = self._prev_az
                rel_el = self._prev_el
            else:
                self._prev_az = rel_az
                self._prev_el = rel_el
        else:
            self._prev_az = rel_az
            self._prev_el = rel_el

        mag = max(math.sqrt(max(rel2, 0.0)), math.sqrt(self._min_distance2), 1e-6)
        s_head = smoothed_unit * mag
        return DirectionSample(s_head=s_head, rel_az=rel_az, rel_el=rel_el)
