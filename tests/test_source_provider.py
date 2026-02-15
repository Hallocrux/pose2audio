import numpy as np

from Pose2Audio.control.pose import Pose6D
from Pose2Audio.control.pose_provider import PoseProvider
from Pose2Audio.control.source_provider import (
    CameraOriginSourceProvider,
    FirstHeadSourceProvider,
    FixedSourceProvider,
)


def _pose(x: float, y: float, z: float) -> Pose6D:
    return Pose6D(
        position=np.array([x, y, z], dtype=np.float64),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )


class _DummyPoseProvider(PoseProvider):
    def __init__(self):
        self.camera_origin = None
        self.tracked = True

    def get_quaternion(self) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def run(self, on_tick):
        raise NotImplementedError

    def get_camera_world_origin(self):
        return self.camera_origin

    def has_tracking(self) -> bool:
        return self.tracked


def test_fixed_source_provider_returns_fixed_world_xyz():
    p = FixedSourceProvider(np.array([0.2, -0.1, 1.3], dtype=np.float64))
    out = p.get_source_world(_pose(9.0, 9.0, 9.0))
    np.testing.assert_allclose(out, np.array([0.2, -0.1, 1.3], dtype=np.float64))


def test_camera_origin_provider_returns_origin():
    pp = _DummyPoseProvider()
    p = CameraOriginSourceProvider(
        pose_provider=pp,
        fallback_world=np.array([0.0, 0.0, 0.0], dtype=np.float64),
    )
    pp.camera_origin = np.array([0.0, 0.0, -0.6], dtype=np.float64)
    out = p.get_source_world(_pose(0.3, 0.2, 1.0))
    np.testing.assert_allclose(out, np.array([0.0, 0.0, -0.6], dtype=np.float64))


def test_camera_origin_provider_uses_fallback_before_origin_available():
    pp = _DummyPoseProvider()
    p = CameraOriginSourceProvider(
        pose_provider=pp,
        fallback_world=np.array([0.0, 0.0, 0.0], dtype=np.float64),
    )
    out = p.get_source_world(_pose(0.3, 0.2, 1.0))
    np.testing.assert_allclose(out, np.array([0.0, 0.0, 0.0], dtype=np.float64))


def test_first_head_provider_latches_first_position():
    pp = _DummyPoseProvider()
    p = FirstHeadSourceProvider(
        pose_provider=pp,
        fallback_world=np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )
    out0 = p.get_source_world(_pose(0.1, 0.0, 0.8))
    out1 = p.get_source_world(_pose(0.5, 0.2, 1.2))
    np.testing.assert_allclose(out0, np.array([0.1, 0.0, 0.8], dtype=np.float64))
    np.testing.assert_allclose(out1, np.array([0.1, 0.0, 0.8], dtype=np.float64))


def test_camera_origin_provider_is_live_when_available():
    pp = _DummyPoseProvider()
    p = CameraOriginSourceProvider(
        pose_provider=pp,
        fallback_world=np.array([0.0, 0.0, 0.0], dtype=np.float64),
    )
    pp.camera_origin = np.array([0.0, 0.0, -0.3], dtype=np.float64)
    out0 = p.get_source_world(_pose(0.1, 0.0, 0.8))
    pp.camera_origin = np.array([0.0, 0.0, -0.8], dtype=np.float64)
    out1 = p.get_source_world(_pose(0.5, 0.2, 1.2))
    np.testing.assert_allclose(out0, np.array([0.0, 0.0, -0.3], dtype=np.float64))
    np.testing.assert_allclose(out1, np.array([0.0, 0.0, -0.8], dtype=np.float64))


def test_first_head_provider_waits_for_tracking_before_latching():
    pp = _DummyPoseProvider()
    pp.tracked = False
    p = FirstHeadSourceProvider(
        pose_provider=pp,
        fallback_world=np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )
    out0 = p.get_source_world(_pose(0.4, 0.5, 0.6))
    np.testing.assert_allclose(out0, np.array([0.0, 0.0, 1.0], dtype=np.float64))

    pp.tracked = True
    out1 = p.get_source_world(_pose(0.4, 0.5, 0.6))
    out2 = p.get_source_world(_pose(1.4, 1.5, 1.6))
    np.testing.assert_allclose(out1, np.array([0.4, 0.5, 0.6], dtype=np.float64))
    np.testing.assert_allclose(out2, np.array([0.4, 0.5, 0.6], dtype=np.float64))
