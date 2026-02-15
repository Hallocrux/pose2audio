# VisualSound.KinectBridge

UDP pose bridge for `VisualSound` `kinectv2` provider.

This bridge does not access Kinect hardware directly.  
It publishes 6DoF pose packets so you can plug in a future C# Kinect module cleanly.

## Run

```powershell
dotnet run --project bridge\VisualSound.KinectBridge -- --mode mock --host 127.0.0.1 --port 24567 --hz 60 --duration-sec 0
```

Use non-scrolling live console output (default):

```powershell
dotnet run --project bridge\VisualSound.KinectBridge -- --mode mock --cli-output live
```

Print HRTF az/el against a chosen world source:

```powershell
dotnet run --project bridge\VisualSound.KinectBridge -- --mode mock --source-x 0 --source-y 0 --source-z 1
```

Kinect SDK mode now uses dedicated project:

```powershell
dotnet run --project bridge\VisualSound.KinectBridge.KinectSdk -- --host 127.0.0.1 --port 24567 --hz 60 --duration-sec 0
```

## Modes

- `mock`: synthetic pose stream for end-to-end validation.
- `stdin-json`: read one JSON packet per line from stdin and forward to UDP.
- `kinect-sdk`: delegated to `VisualSound.KinectBridge.KinectSdk` project.
- `--duration-sec`: auto-stop seconds (`0` means run forever).
- `--cli-output`: `live|scroll` (default `live`)

Example (`stdin-json`):

```powershell
echo {"tracked":true,"position_m":[0,0,0],"quaternion_wxyz":[1,0,0,0]} | dotnet run --project bridge\VisualSound.KinectBridge -- --mode stdin-json
```

## Packet Schema

```json
{
  "tracked": true,
  "position_m": [0.0, 0.0, 0.0],
  "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]
}
```

## Kinect Integration Plan (C#)

1. Real Kinect SDK capture lives in `bridge/VisualSound.KinectBridge.KinectSdk`.
2. It selects nearest tracked body, uses head position, and prefers face quaternion.
3. Keep coordinate transforms and smoothing primarily in Python side for consistency.
4. If your machine uses a non-default Kinect SDK path, override `KinectSdkRoot` in MSBuild.
