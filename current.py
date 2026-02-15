"""
Spatial audio demo:
- RAW SOFA HRIR (nearest neighbor)
- 6DoF head pose (orientation + translation) from MediaPipe/Kinect/ToyCV
- Source provider (fixed/camera/first-head)
- Direction provider (native head frame)
- Acoustic frame provider (native->acoustic semantic mapping)
- Listener-relative (acoustic az,el) -> HRIR
- Display provider (tui/3d) renders the same runtime frame data
- Control plane prepares new convolvers; audio callback renders + crossfades
- Optional: play an audio file (--audio). Preload into RAM for real-time safety.

Deps:
  uv add numpy sounddevice netCDF4 soundfile
"""

from __future__ import annotations

import logging

import numpy as np
import sounddevice as sd

from .audio.conv_overlap_save import OverlapSaveConvolver
from .audio.engine import BinauralEngine
from .audio.source import AudioFileSource, SineSource
from .config import parse_args
from .control.acoustic_frame_provider import (
    FlipFrontAcousticFrameProvider,
    IdentityAcousticFrameProvider,
)
from .control.controller import BinauralController
from .control.direction_provider import RawDirectionProvider, StabilizedDirectionProvider
from .control.display_provider import (
    Matplotlib3DAnimationDisplayProvider,
    Open3DAnimationDisplayProvider,
    TuiDisplayProvider,
)
from .control.source_provider import (
    CameraOriginSourceProvider,
    FirstHeadSourceProvider,
    FixedSourceProvider,
)
from .hrtf.selector import HrtfSelector
from .hrtf.sofa_db import SofaHrirDB
from .math3d.coords import az_el_to_vec, vec_to_az_el_deg
from .pose_providers.toycv_tk import ToyCvTkPoseProvider

logger = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def build_audio_source(cfg, sr: int):
    if cfg.audio:
        if cfg.audio.lower().endswith(".mp3"):
            raise SystemExit(
                "MP3 not supported reliably via soundfile. Convert to WAV/FLAC first (ffmpeg -i in.mp3 out.wav)."
            )
        return AudioFileSource(
            cfg.audio, target_sr=sr, mono=cfg.mono, loop=cfg.loop, gain=cfg.gain
        )

    logger.warning(
        "[AUDIO] no --audio provided, using sine (note: sine is bad for localization)"
    )
    return SineSource(sr, cfg.sine_freq, amp=0.2)


def resolve_fixed_source_world_position(cfg) -> np.ndarray:
    if cfg.source_x is not None:
        source = np.array([cfg.source_x, cfg.source_y, cfg.source_z], dtype=np.float64)
    else:
        source = (
            az_el_to_vec(cfg.source_az, cfg.source_el).astype(np.float64)
            * float(cfg.source_distance)
        )
    logger.info(
        "[SCENE] fixed source world position (m): [%.3f, %.3f, %.3f]",
        source[0],
        source[1],
        source[2],
    )
    return source


def build_source_provider(cfg, fixed_source_world: np.ndarray, pose_provider):
    if cfg.source_provider == "fixed":
        provider = FixedSourceProvider(fixed_source_world)
    elif cfg.source_provider == "camera":
        provider = CameraOriginSourceProvider(
            pose_provider=pose_provider,
            fallback_world=np.zeros(3, dtype=np.float64),
        )
    elif cfg.source_provider == "first-head":
        provider = FirstHeadSourceProvider(
            pose_provider=pose_provider,
            fallback_world=fixed_source_world,
        )
    else:
        raise RuntimeError(f"Unsupported source provider: {cfg.source_provider}")

    logger.info(
        "[SCENE] source provider=%s",
        cfg.source_provider,
    )
    return provider


def build_direction_provider(cfg, initial_source_world: np.ndarray):
    if cfg.direction_provider == "raw":
        provider = RawDirectionProvider(initial_source_world=initial_source_world)
    elif cfg.direction_provider == "stabilized":
        provider = StabilizedDirectionProvider(
            initial_source_world=initial_source_world,
            min_distance_m=cfg.direction_min_distance_m,
            smoothing=cfg.direction_smoothing,
            deadband_deg=cfg.direction_deadband_deg,
        )
    else:
        raise RuntimeError(f"Unsupported direction provider: {cfg.direction_provider}")

    logger.info(
        "[SCENE] direction provider=%s min_distance=%.3fm smoothing=%.2f deadband=%.2fdeg",
        cfg.direction_provider,
        cfg.direction_min_distance_m,
        cfg.direction_smoothing,
        cfg.direction_deadband_deg,
    )
    return provider


def build_acoustic_frame_provider(cfg, pose_provider):
    choice = cfg.acoustic_frame_provider
    if choice == "auto":
        choice = pose_provider.default_acoustic_frame_provider()

    if choice == "identity":
        provider = IdentityAcousticFrameProvider()
    elif choice == "flip-front":
        provider = FlipFrontAcousticFrameProvider()
    else:
        raise RuntimeError(f"Unsupported acoustic frame provider: {choice}")

    logger.info(
        "[SCENE] acoustic frame provider=%s (requested=%s)",
        provider.name,
        cfg.acoustic_frame_provider,
    )
    return provider


def resolve_initial_source_world(cfg, fixed_source_world: np.ndarray) -> np.ndarray:
    if cfg.source_provider == "camera":
        return np.zeros(3, dtype=np.float64)
    return fixed_source_world


def build_pose_provider(cfg, sr: int, initial_source_world: np.ndarray):
    if cfg.pose_provider == "toycv":
        return ToyCvTkPoseProvider(
            title="Pose2Audio - ToyCV Head Pose",
            source_world=initial_source_world,
            sr=sr,
        )

    try:
        if cfg.pose_provider == "mediapipe":
            from .pose_providers.mediapipe_cam import MediaPipeCamPoseProvider

            return MediaPipeCamPoseProvider(
                title="Pose2Audio - MediaPipe Monocular Head Pose",
                source_world=initial_source_world,
                sr=sr,
                camera_index=cfg.camera_index,
                camera_width=cfg.camera_width,
                camera_height=cfg.camera_height,
                mirror=cfg.camera_mirror,
                smoothing=cfg.pose_smoothing,
                translation_scale=cfg.mp_translation_scale,
                position_smoothing=cfg.mp_position_smoothing,
                task_model_path=cfg.mp_task_model,
                task_model_url=cfg.mp_task_url,
            )

        if cfg.pose_provider == "kinectv2":
            from .pose_providers.kinect_v2_pose import KinectV2PoseProvider

            return KinectV2PoseProvider(
                title="Pose2Audio - Kinect V2 Face Pose",
                source_world=initial_source_world,
                sr=sr,
                smoothing=cfg.pose_smoothing,
                position_scale=cfg.kinect_position_scale,
                position_smoothing=cfg.kinect_position_smoothing,
                flip_x=cfg.kinect_flip_x,
                poll_ms=cfg.kinect_poll_ms,
                bridge_host=cfg.kinect_bridge_host,
                bridge_port=cfg.kinect_bridge_port,
            )
        raise RuntimeError(f"Unsupported pose provider: {cfg.pose_provider}")
    except (ImportError, ModuleNotFoundError, RuntimeError, OSError):
        logger.exception("[POSE] failed to init requested pose provider")
        logger.warning("[POSE] fallback to ToyCV sliders")
        return ToyCvTkPoseProvider(
            title="Pose2Audio - ToyCV Head Pose (fallback)",
            source_world=initial_source_world,
            sr=sr,
        )


def build_display_provider(cfg, pose_provider):
    if cfg.display_provider == "tui":
        return TuiDisplayProvider(
            pose_provider=pose_provider,
            cli_output=cfg.cli_output,
        )

    if cfg.display_provider == "3d":
        try:
            return Matplotlib3DAnimationDisplayProvider(title="Pose2Audio 3D Scene")
        except RuntimeError:
            logger.exception("[DISPLAY] failed to initialize 3D display provider")
            logger.warning("[DISPLAY] fallback to tui provider")
            return TuiDisplayProvider(
                pose_provider=pose_provider,
                cli_output=cfg.cli_output,
            )

    if cfg.display_provider == "open3d":
        try:
            return Open3DAnimationDisplayProvider(
                pose_provider=pose_provider,
                title="Pose2Audio Open3D Scene",
            )
        except RuntimeError:
            logger.exception("[DISPLAY] failed to initialize Open3D display provider")
            logger.warning("[DISPLAY] fallback to tui provider")
            return TuiDisplayProvider(
                pose_provider=pose_provider,
                cli_output=cfg.cli_output,
            )

    raise RuntimeError(f"Unsupported display provider: {cfg.display_provider}")


def main(argv=None):
    cfg = parse_args(argv)
    configure_logging(cfg.log_level)
    truncate_ms = cfg.truncate_ms if cfg.truncate_ms > 0 else None

    db = SofaHrirDB(cfg.sofa, truncate_ms=truncate_ms)
    sr = db.sr

    fixed_source_world = resolve_fixed_source_world_position(cfg)
    initial_source_world = resolve_initial_source_world(cfg, fixed_source_world)
    az0, el0 = vec_to_az_el_deg(initial_source_world)

    hL0, hR0, idx0 = db.get_hrir(az0, el0, swap_lr=cfg.swap_lr)
    initL = OverlapSaveConvolver(hL0, cfg.block)
    initR = OverlapSaveConvolver(hR0, cfg.block)
    engine = BinauralEngine(
        sr=sr, block=cfg.block, fade_ms=cfg.fade_ms, initL=initL, initR=initR
    )

    src = build_audio_source(cfg, sr)

    underruns = 0

    def audio_callback(outdata, frames, time_info, status):
        nonlocal underruns
        if status:
            underruns += 1
        x = src.next(frames)
        y = engine.process(x)
        np.clip(y, -0.95, 0.95, out=y)
        outdata[:] = y

    stream = sd.OutputStream(
        samplerate=sr,
        blocksize=cfg.block,
        channels=2,
        dtype="float32",
        callback=audio_callback,
    )
    stream.start()

    pose_provider = build_pose_provider(cfg, sr, initial_source_world)
    source_provider = build_source_provider(cfg, fixed_source_world, pose_provider)
    direction_provider = build_direction_provider(cfg, initial_source_world)
    acoustic_frame_provider = build_acoustic_frame_provider(cfg, pose_provider)
    display_provider = build_display_provider(cfg, pose_provider)
    selector = HrtfSelector(db, swap_lr=cfg.swap_lr)
    controller = BinauralController(
        selector=selector,
        engine=engine,
        pose_provider=pose_provider,
        source_provider=source_provider,
        direction_provider=direction_provider,
        acoustic_frame_provider=acoustic_frame_provider,
        display_provider=display_provider,
        initial_source_world=initial_source_world,
        initial_idx=idx0,
        get_underruns=lambda: underruns,
        display_hz=cfg.display_hz,
    )

    try:
        pose_provider.run(controller.tick)
    finally:
        try:
            stream.stop()
            stream.close()
        finally:
            db.close()
            display_provider.close()
            pose_provider.close()


if __name__ == "__main__":
    main()
