"""Compatibility shim; use Pose2Audio.pose_providers.kinect_v2_pose."""

from __future__ import annotations

from ..pose_providers.kinect_v2_pose import (
    KinectV2PoseProvider,
    _parse_pose_packet,
    _parse_pose_payload,
)

__all__ = [
    "KinectV2PoseProvider",
    "_parse_pose_packet",
    "_parse_pose_payload",
]
