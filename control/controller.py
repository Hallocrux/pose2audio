"""Control plane for mapping pose -> HRIR updates."""

from __future__ import annotations

import time

import numpy as np

from ..audio.conv_overlap_save import OverlapSaveConvolver
from ..audio.engine import BinauralEngine
from ..hrtf.selector import HrtfSelector
from ..math3d.coords import vec_to_az_el_deg
from ..math3d.quaternion import q_rotate_vec
from .acoustic_frame_provider import AcousticFrameProvider, IdentityAcousticFrameProvider
from .direction_provider import (
    DirectionProvider,
    RawDirectionProvider,
    head_relative_source_vector,
)
from .display_provider import DisplayFrame, DisplayProvider
from .pose import Pose6D
from .pose_provider import PoseProvider
from .source_provider import SourceProvider


class BinauralController:
    def __init__(
        self,
        selector: HrtfSelector,
        engine: BinauralEngine,
        pose_provider: PoseProvider,
        source_provider: SourceProvider,
        direction_provider: DirectionProvider | None,
        acoustic_frame_provider: AcousticFrameProvider | None,
        display_provider: DisplayProvider,
        initial_source_world: np.ndarray,
        initial_idx: int,
        get_underruns,
        display_hz: float = 5.0,
    ):
        self.selector = selector
        self.engine = engine
        self.pose_provider = pose_provider
        self.source_provider = source_provider
        self.direction_provider = direction_provider or RawDirectionProvider(initial_source_world)
        self.acoustic_frame_provider = acoustic_frame_provider or IdentityAcousticFrameProvider()
        self.display_provider = display_provider
        self.last_idx = initial_idx
        self.get_underruns = get_underruns

        self.prev_q = None
        self.prev_p = None
        self.last_source_world = np.asarray(initial_source_world, dtype=np.float64).reshape(3)
        self.display_interval = (1.0 / display_hz) if display_hz > 0.0 else 0.0
        self.last_display_t = 0.0

    def _maybe_prepare_convolvers(self, hL: np.ndarray, hR: np.ndarray, idx: int) -> None:
        if idx == self.last_idx:
            return

        initL_ov = (
            self.engine.convL.overlap
            if self.engine.convL.overlap.size == (len(hL) - 1)
            else None
        )
        initR_ov = (
            self.engine.convR.overlap
            if self.engine.convR.overlap.size == (len(hR) - 1)
            else None
        )
        newL = OverlapSaveConvolver(hL, self.engine.block, initial_overlap=initL_ov)
        newR = OverlapSaveConvolver(hR, self.engine.block, initial_overlap=initR_ov)
        self.engine.submit_prepared(newL, newR)
        self.last_idx = idx

    def tick(self) -> None:
        pose = self.pose_provider.get_pose()
        q = pose.quaternion

        did_fix = False
        if self.pose_provider.signfix_enabled() and self.prev_q is not None:
            dotv = float(np.dot(self.prev_q, q))
            if dotv < 0.0:
                q = -q
                did_fix = True
        pose = Pose6D(position=pose.position, quaternion=q)
        self.prev_q = q
        self.prev_p = pose.position

        source_world = self.source_provider.get_source_world(pose)
        source_world = np.asarray(source_world, dtype=np.float64).reshape(3)
        if np.isfinite(source_world).all():
            self.last_source_world = source_world
        else:
            source_world = self.last_source_world

        direction = self.direction_provider.compute(source_world=source_world, pose=pose)
        s_head_native = direction.s_head
        # Geometry contract:
        #   source/world -> native head frame (direction provider)
        #   native head frame -> acoustic head frame (acoustic frame provider)
        s_head = self.acoustic_frame_provider.to_acoustic(s_head_native, pose)
        rel_az, rel_el = vec_to_az_el_deg(s_head)

        selection = self.selector.select(rel_az, rel_el)
        self._maybe_prepare_convolvers(selection.hL, selection.hR, selection.idx)

        now = time.time()
        if self.display_interval > 0.0 and (now - self.last_display_t) >= self.display_interval:
            camera_world = self.pose_provider.get_camera_world_origin()
            self.display_provider.update(
                DisplayFrame(
                    pose=pose,
                    source_world=source_world.copy(),
                    camera_world=(
                        None
                        if camera_world is None
                        else np.asarray(camera_world, dtype=np.float64).reshape(3)
                    ),
                    s_head=s_head.copy(),
                    rel_az=rel_az,
                    rel_el=rel_el,
                    acoustic_frame_provider=self.acoustic_frame_provider.name,
                    head_forward_world=q_rotate_vec(
                        pose.quaternion,
                        self.acoustic_frame_provider.acoustic_forward_axis_native(),
                    ),
                    selection_idx=selection.idx,
                    dataset_az=selection.dataset_az,
                    dataset_el=selection.dataset_el,
                    underruns=self.get_underruns(),
                    did_signfix=did_fix,
                )
            )
            self.last_display_t = now
