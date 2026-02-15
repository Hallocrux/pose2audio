"""CLI config and defaults."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class AppConfig:
    sofa: str
    block: int = 512
    fade_ms: float = 30.0
    swap_lr: bool = False
    truncate_ms: float = 0.0
    source_az: float = 30.0
    source_el: float = 0.0
    source_distance: float = 1.0
    source_x: Optional[float] = None
    source_y: Optional[float] = None
    source_z: Optional[float] = None
    source_provider: str = "fixed"
    direction_provider: str = "stabilized"
    direction_min_distance_m: float = 0.15
    direction_smoothing: float = 0.20
    direction_deadband_deg: float = 1.0
    acoustic_frame_provider: str = "auto"
    audio: str = ""
    loop: bool = False
    gain: float = 0.8
    mono: str = "avg"
    sine_freq: float = 220.0
    pose_provider: str = "mediapipe"
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_mirror: bool = True
    pose_smoothing: float = 0.35
    mp_translation_scale: float = 0.001
    mp_position_smoothing: float = 0.25
    log_level: str = "info"
    kinect_flip_x: bool = True
    kinect_position_scale: float = 1.0
    kinect_position_smoothing: float = 0.35
    kinect_poll_ms: int = 8
    kinect_bridge_host: str = "127.0.0.1"
    kinect_bridge_port: int = 24567
    display_provider: str = "tui"
    display_hz: float = 5.0
    cli_output: str = "live"
    mp_task_model: str = "assets/models/face_landmarker.task"
    mp_task_url: str = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    )


_APP_CONFIG_FIELDS = {f.name for f in fields(AppConfig)}
_BOOL_FIELDS = {
    "swap_lr",
    "loop",
    "camera_mirror",
    "kinect_flip_x",
}
_INT_FIELDS = {
    "block",
    "camera_index",
    "camera_width",
    "camera_height",
    "kinect_poll_ms",
    "kinect_bridge_port",
}
_FLOAT_FIELDS = {
    "fade_ms",
    "truncate_ms",
    "source_az",
    "source_el",
    "source_distance",
    "direction_min_distance_m",
    "direction_smoothing",
    "direction_deadband_deg",
    "gain",
    "sine_freq",
    "pose_smoothing",
    "mp_translation_scale",
    "mp_position_smoothing",
    "kinect_position_scale",
    "kinect_position_smoothing",
    "display_hz",
}
_OPTIONAL_FLOAT_FIELDS = {"source_x", "source_y", "source_z"}
_STRING_FIELDS = {
    "sofa",
    "source_provider",
    "direction_provider",
    "acoustic_frame_provider",
    "audio",
    "mono",
    "pose_provider",
    "log_level",
    "kinect_bridge_host",
    "display_provider",
    "cli_output",
    "mp_task_model",
    "mp_task_url",
}
_KEY_ALIASES = {
    "no_camera_mirror": "camera_mirror",
    "no_kinect_flip_x": "kinect_flip_x",
}


def _parse_bool(value: Any, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "true", "yes", "on"}:
            return True
        if s in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"config key '{key}' expects a bool, got {value!r}")


def _coerce_config_value(key: str, value: Any) -> Any:
    try:
        if key in _BOOL_FIELDS:
            return _parse_bool(value, key)
        if key in _INT_FIELDS:
            return int(value)
        if key in _FLOAT_FIELDS:
            return float(value)
        if key in _OPTIONAL_FLOAT_FIELDS:
            return None if value is None else float(value)
        if key in _STRING_FIELDS:
            return "" if value is None else str(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid value for config key '{key}': {value!r}") from exc
    raise ValueError(f"unsupported config key '{key}'")


def _parse_simple_yaml_scalar(raw: str) -> Any:
    s = raw.strip()
    if not s:
        return None
    if s[0] in {"'", '"'} and s[-1] == s[0] and len(s) >= 2:
        return s[1:-1]
    low = s.lower()
    if low in {"null", "~", "none"}:
        return None
    if low in {"true", "yes", "on"}:
        return True
    if low in {"false", "no", "off"}:
        return False
    try:
        if any(ch in s for ch in (".", "e", "E")):
            return float(s)
        return int(s)
    except ValueError:
        return s


def _load_simple_yaml(text: str, path: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if raw_line[:1].isspace():
            raise ValueError(
                f"{path}:{lineno}: nested YAML requires PyYAML; only flat key: value is supported without it"
            )
        if ":" not in raw_line:
            raise ValueError(f"{path}:{lineno}: invalid YAML line: {raw_line!r}")
        key_raw, value_raw = raw_line.split(":", 1)
        key = key_raw.strip()
        if not key:
            raise ValueError(f"{path}:{lineno}: empty key is not allowed")
        value = value_raw.split("#", 1)[0]
        data[key] = _parse_simple_yaml_scalar(value)
    return data


def _normalize_config_key(raw_key: Any) -> str:
    if not isinstance(raw_key, str):
        raise ValueError(f"config key must be string, got {type(raw_key).__name__}")
    key = raw_key.strip().replace("-", "_")
    if not key:
        raise ValueError("config key cannot be empty")
    return _KEY_ALIASES.get(key, key)


def _load_yaml_config(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ValueError(f"--config file not found: {p}")
    if not p.is_file():
        raise ValueError(f"--config must point to a file: {p}")

    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"failed to read --config file {p}: {exc}") from exc
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
    except ModuleNotFoundError:
        loaded = _load_simple_yaml(text, str(p))
    except Exception as exc:
        raise ValueError(f"failed to parse YAML config {p}: {exc}") from exc

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"--config root must be a mapping/object, got {type(loaded).__name__}")

    normalized: dict[str, Any] = {}
    for raw_key, raw_value in loaded.items():
        key = _normalize_config_key(raw_key)
        if key not in _APP_CONFIG_FIELDS:
            raise ValueError(f"unknown config key in {p}: {raw_key!r}")
        normalized[key] = _coerce_config_value(key, raw_value)
    return normalized


def _yaml_defaults_to_argparse_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for key, value in cfg.items():
        if key == "camera_mirror":
            defaults["no_camera_mirror"] = not bool(value)
        elif key == "kinect_flip_x":
            defaults["no_kinect_flip_x"] = not bool(value)
        else:
            defaults[key] = value
    return defaults


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default="",
        help="YAML config file path. CLI args override YAML values.",
    )
    ap.add_argument("--sofa", required=False, default=None)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--fade-ms", type=float, default=30.0)
    ap.add_argument("--swap-lr", action="store_true")
    ap.add_argument(
        "--truncate-ms",
        type=float,
        default=0.0,
        help="truncate HRIR length for CPU (0=off)",
    )

    ap.add_argument(
        "--source-az",
        type=float,
        default=30.0,
        help="fixed WORLD source azimuth (deg)",
    )
    ap.add_argument(
        "--source-el",
        type=float,
        default=0.0,
        help="fixed WORLD source elevation (deg)",
    )
    ap.add_argument(
        "--source-distance",
        type=float,
        default=1.0,
        help="Source distance in meters when using --source-az/--source-el.",
    )
    ap.add_argument(
        "--source-x",
        type=float,
        default=None,
        help="WORLD source x in meters (overrides --source-az/--source-el when x/y/z all set).",
    )
    ap.add_argument("--source-y", type=float, default=None, help="WORLD source y in meters.")
    ap.add_argument("--source-z", type=float, default=None, help="WORLD source z in meters.")
    ap.add_argument(
        "--source-provider",
        choices=["fixed", "camera", "first-head"],
        default="fixed",
        help=(
            "Source position strategy: fixed world xyz, camera origin, "
            "or first tracked head position."
        ),
    )
    ap.add_argument(
        "--direction-provider",
        choices=["raw", "stabilized"],
        default="stabilized",
        help="Head-relative direction strategy before HRTF lookup.",
    )
    ap.add_argument(
        "--direction-min-distance-m",
        type=float,
        default=0.15,
        help="Minimum source-head distance for stable direction updates.",
    )
    ap.add_argument(
        "--direction-smoothing",
        type=float,
        default=0.20,
        help="Direction vector smoothing alpha in [0.01,1]. Lower is smoother.",
    )
    ap.add_argument(
        "--direction-deadband-deg",
        type=float,
        default=1.0,
        help="Angular deadband in degrees to suppress tiny az/el jitter.",
    )
    ap.add_argument(
        "--acoustic-frame-provider",
        choices=["auto", "identity", "flip-front"],
        default="auto",
        help="Native->acoustic frame mapping policy.",
    )

    ap.add_argument(
        "--audio",
        type=str,
        default="",
        help="Path to audio file (WAV/FLAC recommended).",
    )
    ap.add_argument("--loop", action="store_true", help="Loop audio file.")
    ap.add_argument("--gain", type=float, default=0.8, help="Audio gain (0..1-ish).")
    ap.add_argument(
        "--mono",
        choices=["avg", "left", "right"],
        default="avg",
        help="Stereo->mono strategy.",
    )
    ap.add_argument(
        "--sine-freq",
        type=float,
        default=220.0,
        help="Fallback sine frequency if no --audio.",
    )
    ap.add_argument(
        "--pose-provider",
        choices=["mediapipe", "toycv", "kinectv2"],
        default="mediapipe",
        help="Head pose backend: webcam MediaPipe, Kinect bridge stream, or ToyCV sliders.",
    )
    ap.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index for --pose-provider mediapipe.",
    )
    ap.add_argument(
        "--camera-width",
        type=int,
        default=1280,
        help="Requested camera frame width.",
    )
    ap.add_argument(
        "--camera-height",
        type=int,
        default=720,
        help="Requested camera frame height.",
    )
    ap.add_argument(
        "--no-camera-mirror",
        action="store_true",
        help="Disable mirror preview for webcam UI.",
    )
    ap.add_argument(
        "--pose-smoothing",
        type=float,
        default=0.35,
        help="Quaternion smoothing factor in [0.01,1]. Lower is smoother.",
    )
    ap.add_argument(
        "--mp-translation-scale",
        type=float,
        default=0.001,
        help="MediaPipe translation scale to meters (solvePnP tvec units -> meters).",
    )
    ap.add_argument(
        "--mp-position-smoothing",
        type=float,
        default=0.25,
        help="MediaPipe position smoothing factor in [0.01,1].",
    )
    ap.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Global log level.",
    )
    ap.add_argument(
        "--no-kinect-flip-x",
        action="store_true",
        help="Disable x-axis flip for Kinect V2 camera space.",
    )
    ap.add_argument(
        "--kinect-position-scale",
        type=float,
        default=1.0,
        help="Kinect position scale multiplier.",
    )
    ap.add_argument(
        "--kinect-position-smoothing",
        type=float,
        default=0.35,
        help="Kinect position smoothing factor in [0.01,1].",
    )
    ap.add_argument(
        "--kinect-poll-ms",
        type=int,
        default=8,
        help="Kinect polling sleep in milliseconds.",
    )
    ap.add_argument(
        "--kinect-bridge-host",
        type=str,
        default="127.0.0.1",
        help="Host for Kinect bridge UDP pose stream.",
    )
    ap.add_argument(
        "--kinect-bridge-port",
        type=int,
        default=24567,
        help="Port for Kinect bridge UDP pose stream.",
    )
    ap.add_argument(
        "--display-provider",
        choices=["tui", "3d", "open3d"],
        default="tui",
        help="Display provider: terminal TUI, matplotlib 3D, or high-performance Open3D.",
    )
    ap.add_argument(
        "--display-hz",
        type=float,
        default=5.0,
        help="Display refresh rate in Hz (0 disables display updates).",
    )
    ap.add_argument(
        "--cli-stats-hz",
        dest="display_hz",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--cli-output",
        choices=["live", "scroll"],
        default="live",
        help="TUI output mode: in-place live panel or scrolling logs.",
    )
    ap.add_argument(
        "--mp-task-model",
        type=str,
        default="assets/models/face_landmarker.task",
        help="Path to MediaPipe FaceLandmarker .task model.",
    )
    ap.add_argument(
        "--mp-task-url",
        type=str,
        default=(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/latest/face_landmarker.task"
        ),
        help="Download URL for .task model when local file is missing.",
    )

    return ap


def validate_config(cfg: AppConfig) -> None:
    if not str(cfg.sofa).strip():
        raise ValueError("--sofa must be provided (CLI or --config)")
    if cfg.block <= 0:
        raise ValueError(f"--block must be > 0, got {cfg.block}")
    if cfg.fade_ms < 0.0:
        raise ValueError(f"--fade-ms must be >= 0, got {cfg.fade_ms}")
    if cfg.truncate_ms < 0.0:
        raise ValueError(f"--truncate-ms must be >= 0, got {cfg.truncate_ms}")
    if not (0.0 <= cfg.gain <= 2.0):
        raise ValueError(f"--gain must be in [0,2], got {cfg.gain}")
    if cfg.sine_freq <= 0.0:
        raise ValueError(f"--sine-freq must be > 0, got {cfg.sine_freq}")
    if cfg.camera_index < 0:
        raise ValueError(f"--camera-index must be >= 0, got {cfg.camera_index}")
    if cfg.camera_width < 0:
        raise ValueError(f"--camera-width must be >= 0, got {cfg.camera_width}")
    if cfg.camera_height < 0:
        raise ValueError(f"--camera-height must be >= 0, got {cfg.camera_height}")
    if not (0.01 <= cfg.pose_smoothing <= 1.0):
        raise ValueError(
            f"--pose-smoothing must be in [0.01,1.0], got {cfg.pose_smoothing}"
        )
    if cfg.mp_translation_scale <= 0.0:
        raise ValueError(
            f"--mp-translation-scale must be > 0, got {cfg.mp_translation_scale}"
        )
    if not (0.01 <= cfg.mp_position_smoothing <= 1.0):
        raise ValueError(
            f"--mp-position-smoothing must be in [0.01,1.0], got {cfg.mp_position_smoothing}"
        )
    if cfg.kinect_position_scale <= 0.0:
        raise ValueError(
            f"--kinect-position-scale must be > 0, got {cfg.kinect_position_scale}"
        )
    if not (0.01 <= cfg.kinect_position_smoothing <= 1.0):
        raise ValueError(
            f"--kinect-position-smoothing must be in [0.01,1.0], got {cfg.kinect_position_smoothing}"
        )
    if cfg.kinect_poll_ms <= 0:
        raise ValueError(f"--kinect-poll-ms must be > 0, got {cfg.kinect_poll_ms}")
    if not cfg.kinect_bridge_host.strip():
        raise ValueError("--kinect-bridge-host must be non-empty")
    if not (1 <= cfg.kinect_bridge_port <= 65535):
        raise ValueError(
            f"--kinect-bridge-port must be in [1,65535], got {cfg.kinect_bridge_port}"
        )
    if cfg.display_provider not in {"tui", "3d", "open3d"}:
        raise ValueError(
            f"--display-provider must be one of tui|3d|open3d, got {cfg.display_provider}"
        )
    if cfg.display_hz < 0.0:
        raise ValueError(f"--display-hz must be >= 0, got {cfg.display_hz}")
    if cfg.cli_output not in {"live", "scroll"}:
        raise ValueError(f"--cli-output must be live|scroll, got {cfg.cli_output}")
    if cfg.direction_provider not in {"raw", "stabilized"}:
        raise ValueError(
            f"--direction-provider must be one of raw|stabilized, got {cfg.direction_provider}"
        )
    if cfg.acoustic_frame_provider not in {"auto", "identity", "flip-front"}:
        raise ValueError(
            "--acoustic-frame-provider must be one of auto|identity|flip-front, "
            f"got {cfg.acoustic_frame_provider}"
        )
    if cfg.direction_min_distance_m < 0.0:
        raise ValueError(
            f"--direction-min-distance-m must be >= 0, got {cfg.direction_min_distance_m}"
        )
    if not (0.01 <= cfg.direction_smoothing <= 1.0):
        raise ValueError(
            f"--direction-smoothing must be in [0.01,1.0], got {cfg.direction_smoothing}"
        )
    if cfg.direction_deadband_deg < 0.0:
        raise ValueError(
            f"--direction-deadband-deg must be >= 0, got {cfg.direction_deadband_deg}"
        )
    if not math.isfinite(cfg.source_az) or not math.isfinite(cfg.source_el):
        raise ValueError("--source-az/--source-el must be finite numbers")
    if cfg.source_distance <= 0.0:
        raise ValueError(f"--source-distance must be > 0, got {cfg.source_distance}")

    has_xyz = any(v is not None for v in (cfg.source_x, cfg.source_y, cfg.source_z))
    if has_xyz and not all(
        v is not None for v in (cfg.source_x, cfg.source_y, cfg.source_z)
    ):
        raise ValueError("--source-x/--source-y/--source-z must be all set together")
    if has_xyz and not all(
        math.isfinite(float(v)) for v in (cfg.source_x, cfg.source_y, cfg.source_z)
    ):
        raise ValueError("--source-x/--source-y/--source-z must be finite numbers")


def parse_args(argv=None) -> AppConfig:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default="")
    bootstrap_ns, _ = bootstrap.parse_known_args(argv)

    yaml_cfg: dict[str, Any] = {}
    yaml_error: Optional[str] = None
    if bootstrap_ns.config:
        try:
            yaml_cfg = _load_yaml_config(bootstrap_ns.config)
        except ValueError as exc:
            yaml_error = str(exc)

    ap = build_arg_parser()
    if yaml_error is not None:
        ap.error(yaml_error)
    if yaml_cfg:
        ap.set_defaults(**_yaml_defaults_to_argparse_defaults(yaml_cfg))
    args = ap.parse_args(argv)

    cfg = AppConfig(
        sofa=str(args.sofa or ""),
        block=args.block,
        fade_ms=args.fade_ms,
        swap_lr=args.swap_lr,
        truncate_ms=args.truncate_ms,
        source_az=args.source_az,
        source_el=args.source_el,
        source_distance=args.source_distance,
        source_x=args.source_x,
        source_y=args.source_y,
        source_z=args.source_z,
        source_provider=args.source_provider,
        direction_provider=args.direction_provider,
        direction_min_distance_m=args.direction_min_distance_m,
        direction_smoothing=args.direction_smoothing,
        direction_deadband_deg=args.direction_deadband_deg,
        acoustic_frame_provider=args.acoustic_frame_provider,
        audio=args.audio,
        loop=args.loop,
        gain=args.gain,
        mono=args.mono,
        sine_freq=args.sine_freq,
        pose_provider=args.pose_provider,
        camera_index=args.camera_index,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_mirror=not args.no_camera_mirror,
        pose_smoothing=float(args.pose_smoothing),
        mp_translation_scale=args.mp_translation_scale,
        mp_position_smoothing=args.mp_position_smoothing,
        log_level=args.log_level,
        kinect_flip_x=not args.no_kinect_flip_x,
        kinect_position_scale=args.kinect_position_scale,
        kinect_position_smoothing=args.kinect_position_smoothing,
        kinect_poll_ms=args.kinect_poll_ms,
        kinect_bridge_host=args.kinect_bridge_host,
        kinect_bridge_port=args.kinect_bridge_port,
        display_provider=args.display_provider,
        display_hz=float(args.display_hz),
        cli_output=args.cli_output,
        mp_task_model=args.mp_task_model,
        mp_task_url=args.mp_task_url,
    )
    try:
        validate_config(cfg)
    except ValueError as exc:
        ap.error(str(exc))
    return cfg
