import numpy as np

from VisualSound.control.controller import head_relative_source_vector
from VisualSound.control.pose import Pose6D
from VisualSound.math3d.coords import vec_to_az_el_deg
from VisualSound.math3d.quaternion import euler_yaw_pitch_roll_to_q


def test_head_relative_vector_identity_pose():
    pose = Pose6D(
        position=np.zeros(3, dtype=np.float64),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )
    v = head_relative_source_vector(
        source_world=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        pose=pose,
        fallback_world_dir=np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )
    np.testing.assert_allclose(v, np.array([0.0, 0.0, 1.0], dtype=np.float64))


def test_head_translation_changes_relative_direction():
    pose = Pose6D(
        position=np.array([0.2, 0.0, 0.0], dtype=np.float64),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )
    v = head_relative_source_vector(
        source_world=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        pose=pose,
        fallback_world_dir=np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )
    az, _ = vec_to_az_el_deg(v)
    assert az < 0.0


def test_head_rotation_affects_listener_relative_azimuth():
    pose = Pose6D(
        position=np.zeros(3, dtype=np.float64),
        quaternion=euler_yaw_pitch_roll_to_q(90.0, 0.0, 0.0),
    )
    v = head_relative_source_vector(
        source_world=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        pose=pose,
        fallback_world_dir=np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )
    az, _ = vec_to_az_el_deg(v)
    assert az < -80.0
