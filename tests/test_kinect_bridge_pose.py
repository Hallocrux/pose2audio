import numpy as np

from Pose2Audio.pose_providers.kinect_v2_pose import _parse_pose_packet, _parse_pose_payload


def test_parse_pose_payload_accepts_valid_schema():
    parsed = _parse_pose_payload(
        {
            "tracked": True,
            "position_m": [0.1, -0.2, 1.5],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        }
    )
    assert parsed is not None
    p, q, tracked = parsed
    np.testing.assert_allclose(p, np.array([0.1, -0.2, 1.5], dtype=np.float64))
    np.testing.assert_allclose(q, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    assert tracked is True


def test_parse_pose_packet_rejects_invalid_json():
    assert _parse_pose_packet(b"{not-json") is None


def test_parse_pose_payload_rejects_bad_vector_lengths():
    assert (
        _parse_pose_payload(
            {"position_m": [0.0, 1.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}
        )
        is None
    )
    assert (
        _parse_pose_payload(
            {"position_m": [0.0, 1.0, 2.0], "quaternion_wxyz": [1.0, 0.0, 0.0]}
        )
        is None
    )
