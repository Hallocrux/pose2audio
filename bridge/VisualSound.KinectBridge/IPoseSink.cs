namespace VisualSound.KinectBridge;

internal interface IPoseSink : IAsyncDisposable
{
    ValueTask SendAsync(PoseSample sample, CancellationToken cancellationToken);
}

