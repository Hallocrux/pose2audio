# Pose2Audio.KinectBridge.KinectSdk.HD

Kinect V2 SDK HD Face bridge (C# / .NET Framework 4.8).

This process reads Kinect body + face orientation using:

- `Microsoft.Kinect`
- `Microsoft.Kinect.Face`

Then publishes UDP JSON packets compatible with Python `--pose-provider kinectv2`.

## Run

```powershell
dotnet run --project bridge\Pose2Audio.KinectBridge.KinectSdk.HD -- --host 127.0.0.1 --port 24567 --hz 60 --duration-sec 0
```

Use non-scrolling live console output (default):

```powershell
dotnet run --project bridge\Pose2Audio.KinectBridge.KinectSdk.HD -- --cli-output live
```

Print HRTF az/el against a chosen world source:

```powershell
dotnet run --project bridge\Pose2Audio.KinectBridge.KinectSdk.HD -- --source-x 0 --source-y 0 --source-z 1
```

## Output Packet

```json
{
  "tracked": true,
  "position_m": [x, y, z],
  "quaternion_wxyz": [w, x, y, z]
}
```

## Notes

- Requires Kinect for Windows SDK v2 installation.
- If your SDK path is non-default, set MSBuild property `KinectSdkRoot`.
- Pose is sensor-space raw output; coordinate normalization/smoothing is handled in Python provider.
- HD Face orientation is gated by `FaceAlignmentQuality`. When quality is low, the bridge freezes pose
  and sets `tracked=false` in the UDP packet.
- `--cli-output`: `live|scroll` (default `live`).
