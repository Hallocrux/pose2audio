"""Acoustic frame providers.

This module is the single place that maps a provider-native head frame to the
acoustic head frame used by HRTF lookup:

- acoustic x: right
- acoustic y: up
- acoustic z: front

Keeping this mapping isolated avoids accidental sign/axis mismatches when new
pose providers are added.
"""

from __future__ import annotations

import numpy as np

from .pose import Pose6D


class AcousticFrameProvider:
    """Maps native head-frame vectors into the acoustic head frame."""

    name: str = "identity"

    def to_acoustic(self, s_head_native: np.ndarray, pose: Pose6D) -> np.ndarray:  # noqa: ARG002
        return np.asarray(s_head_native, dtype=np.float64).reshape(3)

    def acoustic_forward_axis_native(self) -> np.ndarray:
        """Native local axis corresponding to acoustic +Z (front)."""
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)


class IdentityAcousticFrameProvider(AcousticFrameProvider):
    """Native frame already matches acoustic frame."""

    name = "identity"


class FlipFrontAcousticFrameProvider(AcousticFrameProvider):
    """Flip front/back while preserving left/right/up."""

    name = "flip-front"

    def to_acoustic(self, s_head_native: np.ndarray, pose: Pose6D) -> np.ndarray:  # noqa: ARG002
        v = np.asarray(s_head_native, dtype=np.float64).reshape(3)
        return np.array([v[0], v[1], -v[2]], dtype=np.float64)

    def acoustic_forward_axis_native(self) -> np.ndarray:
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)
