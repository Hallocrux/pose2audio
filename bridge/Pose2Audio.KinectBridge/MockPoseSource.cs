using System.Diagnostics;

namespace Pose2Audio.KinectBridge;

internal sealed class MockPoseSource : IPoseSource
{
    private readonly Stopwatch _sw = Stopwatch.StartNew();

    public bool TryGetLatest(out PoseSample sample)
    {
        var t = _sw.Elapsed.TotalSeconds;

        var x = 0.25 * Math.Sin(0.7 * t);
        var y = 0.05 * Math.Sin(1.1 * t);
        var z = 0.15 * Math.Cos(0.5 * t);

        var yaw = 0.4 * Math.Sin(0.8 * t);
        var half = yaw * 0.5;
        var qw = Math.Cos(half);
        var qy = Math.Sin(half);

        sample = new PoseSample(
            Tracked: true,
            X: x,
            Y: y,
            Z: z,
            Qw: qw,
            Qx: 0.0,
            Qy: qy,
            Qz: 0.0);
        return true;
    }

    public void Dispose()
    {
    }
}

