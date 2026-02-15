# Pose2Audio

Real-time head-tracked binaural spatial audio demo.

## Features

- SOFA HRIR loading with nearest-neighbor selection
- 6DoF head pose pipeline (translation + orientation)
- Head pose from MediaPipe webcam, Kinect bridge stream, or ToyCV sliders
- Real-time convolution with crossfade switching
- Audio from file or sine fallback

## Geometry Conventions

VisualSound uses one acoustic frame contract across all providers:

- acoustic `x`: right
- acoustic `y`: up
- acoustic `z`: front
- azimuth/elevation are computed as:
  - `az = atan2(x, z)` (`0` = front, `+90` = right)
  - `el = atan2(y, sqrt(x^2 + z^2))`

Pipeline order:

1. Pose provider outputs head pose in its **native** frame.
2. Source provider outputs source in world frame.
3. Direction provider computes source direction in native head frame.
4. Acoustic frame provider maps native head-frame direction to the acoustic frame.
5. Acoustic `az/el` drives HRIR selection.

This split keeps pose/device details separate from acoustic semantics and makes
future provider integration safer.

### What "Calibration" Means Here

Calibration in this project means runtime reference locking (not offline camera
extrinsic calibration):

- `p_ref`: startup head position baseline
- `q_ref`: startup head orientation baseline
- later pose is interpreted relative to that baseline

## Requirements

- Python 3.10+
- A valid SOFA file (`--sofa`)
- For webcam mode: camera device and MediaPipe model (auto-download supported)
- For Kinect mode: external bridge process (recommended: C#) publishing UDP JSON pose packets

## Quick Start (uv)

Install and run:

```powershell
uv run visualsound --sofa path\to\your.sofa --pose-provider toycv
```

Run with webcam head pose:

```powershell
uv run visualsound --sofa path\to\your.sofa --pose-provider mediapipe
```

Use audio file:

```powershell
uv run visualsound --sofa path\to\your.sofa --audio assets\music.wav --loop
```

Run with Kinect bridge stream (future C# provider):

```powershell
uv run visualsound --sofa path\to\your.sofa --pose-provider kinectv2 --kinect-bridge-host 127.0.0.1 --kinect-bridge-port 24567
```

Use default TUI display provider:

```powershell
uv run visualsound --sofa path\to\your.sofa --display-provider tui --cli-output live
```

Use simple 3D animation display provider:

```powershell
uv run visualsound --sofa path\to\your.sofa --display-provider 3d --display-hz 20
```

Use high-performance Open3D display provider:

```powershell
uv add open3d
uv run visualsound --sofa path\to\your.sofa --display-provider open3d --display-hz 60
```

Open3D controls:
- `Ctrl + Mouse Wheel`: zoom in/out
- `Shift + Mouse Wheel`: pan left/right in camera view
- `Mouse Wheel`: pan up/down in camera view
- `Left Mouse Drag`: orbit camera around a center
- `F`: return to auto-follow scene center

Use source-at-camera provider:

```powershell
uv run visualsound --sofa path\to\your.sofa --source-provider camera
```

Force acoustic frame mapping explicitly (defaults to `auto`):

```powershell
uv run visualsound --sofa path\to\your.sofa --source-provider camera --acoustic-frame-provider flip-front
```

Use source-at-first-head provider (latched at first tracked head pose):

```powershell
uv run visualsound --sofa path\to\your.sofa --source-provider first-head
```

Use stabilized direction provider to reduce az/el jitter (default):

```powershell
uv run visualsound --sofa path\to\your.sofa --direction-provider stabilized --direction-min-distance-m 0.15 --direction-smoothing 0.2 --direction-deadband-deg 1.0
```

Run with YAML startup config:

```powershell
uv run visualsound --config .\configs\kinect_open3d.yaml
```

Example `configs/kinect_open3d.yaml`:

```yaml
sofa: assets/kemar_windowed_48k.sofa
pose_provider: kinectv2
kinect_bridge_host: 127.0.0.1
kinect_bridge_port: 24567
display_provider: open3d
display_hz: 60
source_provider: fixed
direction_provider: stabilized
direction_min_distance_m: 0.15
direction_smoothing: 0.2
direction_deadband_deg: 1.0
acoustic_frame_provider: auto
source_az: 30
source_el: 0
source_distance: 1.0
```

Start C# bridge (mock mode for integration test):

```powershell
dotnet run --project bridge\VisualSound.KinectBridge -- --mode mock --host 127.0.0.1 --port 24567 --hz 60 --duration-sec 0 --cli-output live
```

Start C# bridge (real Kinect SDK mode):

```powershell
dotnet run --project bridge\VisualSound.KinectBridge.KinectSdk -- --host 127.0.0.1 --port 24567 --hz 60 --duration-sec 0 --cli-output live
```

Start C# bridge (HD Face Kinect SDK mode):

```powershell
dotnet run --project bridge\VisualSound.KinectBridge.KinectSdk.HD -- --host 127.0.0.1 --port 24567 --hz 60 --duration-sec 0 --cli-output live
```

## Useful Options

- `--source-az`, `--source-el`, `--source-distance`: fixed world-space source position from spherical coords
- `--source-x`, `--source-y`, `--source-z`: fixed world-space source position (overrides az/el)
- `--source-provider`: `fixed|camera|first-head` (default `fixed`)
- `--direction-provider`: `raw|stabilized` (default `stabilized`)
- `--direction-min-distance-m`: near-field stability threshold
- `--direction-smoothing`: unit-vector smoothing alpha
- `--direction-deadband-deg`: small-angle deadband to suppress flicker
- `--acoustic-frame-provider`: `auto|identity|flip-front` (default `auto`)
- `--truncate-ms`: shorten HRIR length for lower CPU
- `--block`: audio block size
- `--fade-ms`: crossfade duration when HRIR switches
- `--kinect-bridge-host`, `--kinect-bridge-port`: UDP endpoint for external 6DoF stream
- `--config`: YAML startup config file (CLI args override YAML)
- `--display-provider`: `tui|3d|open3d` (default `tui`)
- `--display-hz`: display refresh rate
- `--cli-output`: `live|scroll` (TUI display mode, default `live`)
- `--log-level`: `debug|info|warning|error`

### Kinect Bridge Packet Schema

Each UDP packet should be JSON:

```json
{
  "tracked": true,
  "position_m": [0.0, 0.0, 0.0],
  "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]
}
```

## Troubleshooting

### Facing camera but source sounds behind

Use `--source-provider camera` with acoustic mapping set to `auto` (or explicit
`flip-front` for camera-style sensors):

```powershell
uv run visualsound --sofa path\to\your.sofa --source-provider camera --acoustic-frame-provider auto
```

`auto` chooses provider defaults:

- `toycv` -> `identity`
- `mediapipe` -> `flip-front`
- `kinectv2` -> `flip-front`

## Tests

```powershell
uv run --with pytest -- python -m pytest -q -p no:cacheprovider
```
