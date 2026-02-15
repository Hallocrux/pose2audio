namespace VisualSound.KinectBridge;

internal interface IPoseSource : IDisposable
{
    bool TryGetLatest(out PoseSample sample);
}

