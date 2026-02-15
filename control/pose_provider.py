"""Pose provider interfaces for head orientation."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .pose import Pose6D


class PoseProvider:
    """Base interface for head pose providers.

    Implementations may be UI-based (ToyCV) or real CV (MediaPipe/Kinect).
    """

    # Native->acoustic default mapping policy for this provider.
    # Valid: identity | flip-front
    default_acoustic_frame_provider_name: str = "identity"

    def get_pose(self) -> Pose6D:
        """Return 6DoF pose. Default keeps backward compatibility.

        Providers that only supply orientation can override get_quaternion().
        """
        return Pose6D(
            position=np.zeros(3, dtype=np.float64),
            quaternion=self.get_quaternion(),
        )

    def get_quaternion(self) -> np.ndarray:
        raise NotImplementedError

    def signfix_enabled(self) -> bool:
        return False

    def set_status(self, text: str) -> None:
        # Optional UI hook.
        pass

    def get_camera_world_origin(self) -> np.ndarray | None:
        """Return camera/sensor origin in current world frame when available."""
        return None

    def has_tracking(self) -> bool:
        """Whether current pose sample comes from valid tracking."""
        return True

    def default_acoustic_frame_provider(self) -> str:
        """Return default native->acoustic mapping policy for this provider."""
        return self.default_acoustic_frame_provider_name

    def run(self, on_tick: Callable[[], None]) -> None:
        """Run the provider's event loop and call on_tick periodically."""
        raise NotImplementedError

    def close(self) -> None:
        pass
