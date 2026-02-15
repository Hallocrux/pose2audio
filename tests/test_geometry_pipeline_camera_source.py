import numpy as np

from Pose2Audio.control.acoustic_frame_provider import (
    FlipFrontAcousticFrameProvider,
    IdentityAcousticFrameProvider,
)
from Pose2Audio.control.direction_provider import RawDirectionProvider
from Pose2Audio.control.pose import Pose6D
from Pose2Audio.math3d.coords import vec_to_az_el_deg


def test_camera_source_is_front_after_flip_front_mapping():
    # Camera source scenario with Kinect/MediaPipe style world:
    # startup head is at +Z in sensor frame -> camera origin becomes -Z in world.
    source_world_camera = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    pose = Pose6D(
        position=np.zeros(3, dtype=np.float64),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )

    raw_dir = RawDirectionProvider(initial_source_world=source_world_camera)
    s_native = raw_dir.compute(source_world=source_world_camera, pose=pose).s_head

    az_native, _ = vec_to_az_el_deg(s_native)
    assert abs(abs(az_native) - 180.0) < 1e-6

    s_acoustic = FlipFrontAcousticFrameProvider().to_acoustic(s_native, pose)
    az, el = vec_to_az_el_deg(s_acoustic)
    assert abs(az) < 1e-6
    assert abs(el) < 1e-6


def test_identity_mapping_keeps_camera_source_behind():
    source_world_camera = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    pose = Pose6D(
        position=np.zeros(3, dtype=np.float64),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )
    raw_dir = RawDirectionProvider(initial_source_world=source_world_camera)
    s_native = raw_dir.compute(source_world=source_world_camera, pose=pose).s_head
    s_acoustic = IdentityAcousticFrameProvider().to_acoustic(s_native, pose)
    az, _ = vec_to_az_el_deg(s_acoustic)
    assert abs(abs(az) - 180.0) < 1e-6
