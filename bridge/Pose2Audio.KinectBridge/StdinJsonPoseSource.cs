using System.Text.Json;

namespace Pose2Audio.KinectBridge;

internal sealed class StdinJsonPoseSource : IPoseSource
{
    private readonly CancellationToken _shutdown;
    private readonly CancellationTokenSource _localCts = new();
    private readonly Task _readerTask;
    private readonly object _gate = new();
    private PoseSample _latest = PoseSample.Identity;
    private bool _hasSample;

    public StdinJsonPoseSource(CancellationToken shutdown)
    {
        _shutdown = shutdown;
        _readerTask = Task.Run(ReadLoopAsync);
    }

    public bool TryGetLatest(out PoseSample sample)
    {
        lock (_gate)
        {
            sample = _latest;
            return _hasSample;
        }
    }

    private async Task ReadLoopAsync()
    {
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(_shutdown, _localCts.Token);
        var ct = linked.Token;
        while (!ct.IsCancellationRequested)
        {
            string? line;
            try
            {
                line = await Console.In.ReadLineAsync(ct).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                break;
            }

            if (line is null)
            {
                await Task.Delay(5, ct).ConfigureAwait(false);
                continue;
            }
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            if (TryParse(line, out var sample))
            {
                lock (_gate)
                {
                    _latest = sample;
                    _hasSample = true;
                }
            }
        }
    }

    private static bool TryParse(string line, out PoseSample sample)
    {
        try
        {
            using var doc = JsonDocument.Parse(line);
            var root = doc.RootElement;
            var tracked = root.TryGetProperty("tracked", out var trackedNode) && trackedNode.ValueKind == JsonValueKind.False
                ? false
                : true;

            var pNode = root.TryGetProperty("position_m", out var p) ? p :
                root.TryGetProperty("position", out var p2) ? p2 : default;
            var qNode = root.TryGetProperty("quaternion_wxyz", out var q) ? q :
                root.TryGetProperty("quaternion", out var q2) ? q2 : default;

            if (pNode.ValueKind != JsonValueKind.Array || qNode.ValueKind != JsonValueKind.Array)
            {
                sample = default;
                return false;
            }

            if (pNode.GetArrayLength() != 3 || qNode.GetArrayLength() != 4)
            {
                sample = default;
                return false;
            }

            var x = pNode[0].GetDouble();
            var y = pNode[1].GetDouble();
            var z = pNode[2].GetDouble();
            var qw = qNode[0].GetDouble();
            var qx = qNode[1].GetDouble();
            var qy = qNode[2].GetDouble();
            var qz = qNode[3].GetDouble();

            if (!AllFinite(x, y, z, qw, qx, qy, qz))
            {
                sample = default;
                return false;
            }

            var norm = Math.Sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
            if (norm <= 1e-12)
            {
                qw = 1.0;
                qx = 0.0;
                qy = 0.0;
                qz = 0.0;
            }
            else
            {
                var inv = 1.0 / norm;
                qw *= inv;
                qx *= inv;
                qy *= inv;
                qz *= inv;
            }

            sample = new PoseSample(tracked, x, y, z, qw, qx, qy, qz);
            return true;
        }
        catch (JsonException)
        {
            sample = default;
            return false;
        }
        catch (FormatException)
        {
            sample = default;
            return false;
        }
    }

    private static bool AllFinite(params double[] values)
    {
        foreach (var v in values)
        {
            if (double.IsNaN(v) || double.IsInfinity(v))
            {
                return false;
            }
        }

        return true;
    }

    public void Dispose()
    {
        _localCts.Cancel();
        try
        {
            _readerTask.Wait(TimeSpan.FromSeconds(1));
        }
        catch (AggregateException)
        {
        }
        _localCts.Dispose();
    }
}
