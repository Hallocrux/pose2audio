namespace Pose2Audio.KinectBridge;

internal interface IPoseSource : IDisposable
{
    bool TryGetLatest(out PoseSample sample);
}

