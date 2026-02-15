import numpy as np

from Pose2Audio.control.acoustic_frame_provider import (
    FlipFrontAcousticFrameProvider,
    IdentityAcousticFrameProvider,
)
from Pose2Audio.control.pose import Pose6D


def _identity_pose() -> Pose6D:
    return Pose6D(
        position=np.zeros(3, dtype=np.float64),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )


def test_identity_acoustic_frame_provider_keeps_vector():
    provider = IdentityAcousticFrameProvider()
    v = np.array([0.3, -0.2, 0.9], dtype=np.float64)
    out = provider.to_acoustic(v, _identity_pose())
    np.testing.assert_allclose(out, v)
    np.testing.assert_allclose(
        provider.acoustic_forward_axis_native(),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )


def test_flip_front_provider_flips_only_z():
    provider = FlipFrontAcousticFrameProvider()
    v = np.array([0.3, -0.2, 0.9], dtype=np.float64)
    out = provider.to_acoustic(v, _identity_pose())
    np.testing.assert_allclose(out, np.array([0.3, -0.2, -0.9], dtype=np.float64))
    np.testing.assert_allclose(
        provider.acoustic_forward_axis_native(),
        np.array([0.0, 0.0, -1.0], dtype=np.float64),
    )
