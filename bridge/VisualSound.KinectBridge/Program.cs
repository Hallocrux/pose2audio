using System.Diagnostics;

namespace VisualSound.KinectBridge;

internal sealed class CliStatusPrinter
{
    private readonly bool _live;
    private int _lastLen;

    public CliStatusPrinter(string mode)
    {
        _live = string.Equals(mode, "live", StringComparison.OrdinalIgnoreCase);
    }

    public void Print(string line)
    {
        if (!_live)
        {
            Console.WriteLine(line);
            return;
        }

        var padded = line;
        if (line.Length < _lastLen)
        {
            padded += new string(' ', _lastLen - line.Length);
        }
        _lastLen = line.Length;
        Console.Write("\r" + padded);
    }

    public void EndLiveLine()
    {
        if (_live && _lastLen > 0)
        {
            Console.WriteLine();
            _lastLen = 0;
        }
    }
}

internal static class Program
{
    public static async Task<int> Main(string[] args)
    {
        if (!BridgeOptions.TryParse(args, out var options, out var error))
        {
            if (string.Equals(error, "Help requested.", StringComparison.Ordinal))
            {
                Console.WriteLine(BridgeOptions.HelpText);
                return 0;
            }

            Console.Error.WriteLine(error);
            Console.Error.WriteLine();
            Console.Error.WriteLine(BridgeOptions.HelpText);
            return 2;
        }

        using var cts = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) =>
        {
            e.Cancel = true;
            cts.Cancel();
        };

        IPoseSource source = options.Mode switch
        {
            BridgeMode.Mock => new MockPoseSource(),
            BridgeMode.StdinJson => new StdinJsonPoseSource(cts.Token),
            BridgeMode.KinectSdk => CreateKinectSourceOrThrow(),
            _ => throw new InvalidOperationException($"Unsupported mode: {options.Mode}")
        };

        await using var sink = new UdpPoseSink(options.Host, options.Port);
        using (source)
        {
            Console.WriteLine(
                $"[{DateTimeOffset.Now:O}] bridge start mode={options.Mode} udp={options.Host}:{options.Port} hz={options.Hz}");
            Console.WriteLine("Press Ctrl+C to stop.");
            Console.WriteLine($"CLI output mode: {options.CliOutput}");

            var timer = new PeriodicTimer(TimeSpan.FromSeconds(1.0 / options.Hz));
            var sw = Stopwatch.StartNew();
            long sent = 0;
            PoseSample last = PoseSample.Identity;
            var stopAt = options.DurationSec > 0.0 ? DateTime.UtcNow.AddSeconds(options.DurationSec) : (DateTime?)null;
            var statusPrinter = new CliStatusPrinter(options.CliOutput);

            try
            {
                while (await timer.WaitForNextTickAsync(cts.Token))
                {
                    if (stopAt is not null && DateTime.UtcNow >= stopAt.Value)
                    {
                        break;
                    }

                    if (source.TryGetLatest(out var sample))
                    {
                        last = sample;
                    }

                    await sink.SendAsync(last, cts.Token);
                    sent++;

                    if (sent % options.Hz == 0)
                    {
                        var elapsed = sw.Elapsed.TotalSeconds;
                        if (elapsed > 0)
                        {
                            var rate = sent / elapsed;
                            var hrtf = TryComputeHrtfAzEl(last, options, out var az, out var el)
                                ? $"hrtf_az_el=({az:F2},{el:F2})"
                                : "hrtf_az_el=(n/a)";
                            statusPrinter.Print(
                                $"[{DateTimeOffset.Now:O}] sent={sent} avg_rate={rate:F1}Hz tracked={last.Tracked} " +
                                $"head_xyz=({last.X:F3},{last.Y:F3},{last.Z:F3}) {hrtf}");
                        }
                    }
                }
            }
            catch (OperationCanceledException)
            {
                // Graceful shutdown.
            }
            finally
            {
                timer.Dispose();
                statusPrinter.EndLiveLine();
            }

            Console.WriteLine($"[{DateTimeOffset.Now:O}] bridge stopped.");
        }

        return 0;
    }

    private static IPoseSource CreateKinectSourceOrThrow()
    {
        throw new InvalidOperationException(
            "kinect-sdk mode lives in dedicated project: bridge/VisualSound.KinectBridge.KinectSdk. " +
            "Run: dotnet run --project bridge\\VisualSound.KinectBridge.KinectSdk -- --host 127.0.0.1 --port 24567 --hz 60");
    }

    private static bool TryComputeHrtfAzEl(PoseSample pose, BridgeOptions options, out double az, out double el)
    {
        az = 0.0;
        el = 0.0;
        if (!pose.Tracked)
        {
            return false;
        }

        var relX = options.SourceX - pose.X;
        var relY = options.SourceY - pose.Y;
        var relZ = options.SourceZ - pose.Z;
        var relN2 = (relX * relX) + (relY * relY) + (relZ * relZ);
        if (relN2 < 1e-12)
        {
            return false;
        }

        var qw = pose.Qw;
        var qx = pose.Qx;
        var qy = pose.Qy;
        var qz = pose.Qz;
        var qn = Math.Sqrt((qw * qw) + (qx * qx) + (qy * qy) + (qz * qz));
        if (qn < 1e-12)
        {
            qw = 1.0;
            qx = 0.0;
            qy = 0.0;
            qz = 0.0;
        }
        else
        {
            qw /= qn;
            qx /= qn;
            qy /= qn;
            qz /= qn;
        }

        // Convert world vector to head frame by rotating with conjugate(q).
        RotateByQuaternion(qw, -qx, -qy, -qz, relX, relY, relZ, out var hx, out var hy, out var hz);

        az = Math.Atan2(hx, hz) * (180.0 / Math.PI);
        el = Math.Atan2(hy, Math.Sqrt((hx * hx) + (hz * hz)) + 1e-12) * (180.0 / Math.PI);
        return true;
    }

    private static void RotateByQuaternion(
        double w,
        double x,
        double y,
        double z,
        double vx,
        double vy,
        double vz,
        out double rx,
        out double ry,
        out double rz)
    {
        var dotUv = (x * vx) + (y * vy) + (z * vz);
        var dotUu = (x * x) + (y * y) + (z * z);
        var cx = (y * vz) - (z * vy);
        var cy = (z * vx) - (x * vz);
        var cz = (x * vy) - (y * vx);

        rx = (2.0 * dotUv * x) + (((w * w) - dotUu) * vx) + (2.0 * w * cx);
        ry = (2.0 * dotUv * y) + (((w * w) - dotUu) * vy) + (2.0 * w * cy);
        rz = (2.0 * dotUv * z) + (((w * w) - dotUu) * vz) + (2.0 * w * cz);
    }
}
