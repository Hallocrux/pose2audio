namespace Pose2Audio.KinectBridge;

internal interface IPoseSink : IAsyncDisposable
{
    ValueTask SendAsync(PoseSample sample, CancellationToken cancellationToken);
}

