import numpy as np

from Pose2Audio.control.direction_provider import (
    RawDirectionProvider,
    StabilizedDirectionProvider,
)
from Pose2Audio.control.pose import Pose6D


def _pose_at(x: float, y: float, z: float) -> Pose6D:
    return Pose6D(
        position=np.array([x, y, z], dtype=np.float64),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )


def test_stabilized_provider_deadband_suppresses_small_az_jitter():
    raw = RawDirectionProvider(initial_source_world=np.array([0.0, 0.0, 1.0], dtype=np.float64))
    stable = StabilizedDirectionProvider(
        initial_source_world=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        min_distance_m=0.0,
        smoothing=1.0,
        deadband_deg=1.0,
    )
    pose = _pose_at(0.0, 0.0, 0.0)

    xs = [0.0, 0.008, -0.009, 0.007, -0.006, 0.0]
    raw_az = []
    stable_az = []
    for x in xs:
        source = np.array([x, 0.0, 1.0], dtype=np.float64)
        raw_az.append(raw.compute(source_world=source, pose=pose).rel_az)
        stable_az.append(stable.compute(source_world=source, pose=pose).rel_az)

    assert max(abs(v) for v in raw_az) > 0.3
    assert max(abs(v) for v in stable_az) < 1e-9


def test_stabilized_provider_min_distance_prevents_near_zero_direction_blowup():
    raw = RawDirectionProvider(initial_source_world=np.array([0.0, 0.0, 1.0], dtype=np.float64))
    stable = StabilizedDirectionProvider(
        initial_source_world=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        min_distance_m=0.15,
        smoothing=1.0,
        deadband_deg=0.0,
    )
    source = np.zeros(3, dtype=np.float64)

    raw_az = []
    stable_az = []
    for pose in [
        _pose_at(0.001, 0.000, 0.001),
        _pose_at(-0.001, 0.000, 0.001),
        _pose_at(0.000, 0.001, 0.001),
    ]:
        raw_az.append(raw.compute(source_world=source, pose=pose).rel_az)
        stable_az.append(stable.compute(source_world=source, pose=pose).rel_az)

    assert max(abs(v) for v in raw_az) > 30.0
    assert max(abs(v) for v in stable_az) < 1e-9
