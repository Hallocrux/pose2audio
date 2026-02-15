"""Source position providers for world-space sound source placement."""

from __future__ import annotations

import numpy as np

from .pose import Pose6D
from .pose_provider import PoseProvider

class SourceProvider:
    """Base interface for world-space source position providers."""

    def get_source_world(self, pose: Pose6D) -> np.ndarray:
        raise NotImplementedError


class FixedSourceProvider(SourceProvider):
    """Always returns a fixed world-space source position."""

    def __init__(self, source_world: np.ndarray):
        self.source_world = np.asarray(source_world, dtype=np.float64).reshape(3)

    def get_source_world(self, pose: Pose6D) -> np.ndarray:  # noqa: ARG002
        return self.source_world


class CameraOriginSourceProvider(SourceProvider):
    """Places source at camera/sensor origin in world coordinates."""

    def __init__(
        self,
        pose_provider: PoseProvider,
        fallback_world: np.ndarray,
    ):
        self.pose_provider = pose_provider
        self.fallback_world = np.asarray(fallback_world, dtype=np.float64).reshape(3)

    def get_source_world(self, pose: Pose6D) -> np.ndarray:  # noqa: ARG002
        origin = self.pose_provider.get_camera_world_origin()
        if origin is not None:
            origin = np.asarray(origin, dtype=np.float64).reshape(3)
            if np.isfinite(origin).all():
                return origin
        return self.fallback_world


class FirstHeadSourceProvider(SourceProvider):
    """Places source at first tracked head position (latched once)."""

    def __init__(
        self,
        pose_provider: PoseProvider,
        fallback_world: np.ndarray,
    ):
        self.pose_provider = pose_provider
        self.fallback_world = np.asarray(fallback_world, dtype=np.float64).reshape(3)
        self._latched = self.fallback_world
        self._has_latched = False

    def get_source_world(self, pose: Pose6D) -> np.ndarray:
        if self._has_latched:
            return self._latched
        if not self.pose_provider.has_tracking():
            return self.fallback_world
        p = np.asarray(pose.position, dtype=np.float64).reshape(3)
        if not self._has_latched:
            if np.isfinite(p).all():
                self._latched = p.copy()
                self._has_latched = True
            else:
                self._latched = self.fallback_world
        return self._latched if self._has_latched else self.fallback_world
