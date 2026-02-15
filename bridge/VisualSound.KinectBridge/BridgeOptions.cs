namespace VisualSound.KinectBridge;

internal enum BridgeMode
{
    Mock,
    StdinJson,
    KinectSdk,
}

internal sealed record BridgeOptions(
    string Host,
    int Port,
    int Hz,
    BridgeMode Mode,
    double DurationSec,
    double SourceX,
    double SourceY,
    double SourceZ,
    string CliOutput)
{
    public const string HelpText = """
Usage:
  VisualSound.KinectBridge [--host 127.0.0.1] [--port 24567] [--hz 60] [--mode mock|stdin-json|kinect-sdk] [--duration-sec 0] [--source-x 0] [--source-y 0] [--source-z 1] [--cli-output live|scroll]

Modes:
  mock        Generate synthetic 6DoF data for end-to-end validation.
  stdin-json  Read one JSON pose packet per line from stdin and forward to UDP.
  kinect-sdk  Read Kinect V2 pose using Microsoft.Kinect SDK (requires net48 build/runtime).

JSON packet schema:
  {"tracked":true,"position_m":[x,y,z],"quaternion_wxyz":[w,x,y,z]}
""";

    public static bool TryParse(string[] args, out BridgeOptions options, out string error)
    {
        var host = "127.0.0.1";
        var port = 24567;
        var hz = 60;
        var mode = BridgeMode.Mock;
        var durationSec = 0.0;
        var sourceX = 0.0;
        var sourceY = 0.0;
        var sourceZ = 1.0;
        var cliOutput = "live";

        for (var i = 0; i < args.Length; i++)
        {
            var arg = args[i];
            if (arg is "--help" or "-h")
            {
                options = default!;
                error = "Help requested.";
                return false;
            }

            if (!arg.StartsWith("--", StringComparison.Ordinal))
            {
                options = default!;
                error = $"Unknown argument: {arg}";
                return false;
            }

            if (i + 1 >= args.Length)
            {
                options = default!;
                error = $"Missing value for option: {arg}";
                return false;
            }

            var value = args[++i];
            switch (arg)
            {
                case "--host":
                    host = value.Trim();
                    break;
                case "--port":
                    if (!int.TryParse(value, out port))
                    {
                        options = default!;
                        error = $"Invalid --port: {value}";
                        return false;
                    }
                    break;
                case "--hz":
                    if (!int.TryParse(value, out hz))
                    {
                        options = default!;
                        error = $"Invalid --hz: {value}";
                        return false;
                    }
                    break;
                case "--mode":
                    mode = value switch
                    {
                        "mock" => BridgeMode.Mock,
                        "stdin-json" => BridgeMode.StdinJson,
                        "kinect-sdk" => BridgeMode.KinectSdk,
                        _ => (BridgeMode)(-1),
                    };
                    if (mode is not BridgeMode.Mock and not BridgeMode.StdinJson and not BridgeMode.KinectSdk)
                    {
                        options = default!;
                        error = $"Invalid --mode: {value}. Expected mock|stdin-json|kinect-sdk.";
                        return false;
                    }
                    break;
                case "--duration-sec":
                    if (!double.TryParse(value, out durationSec))
                    {
                        options = default!;
                        error = $"Invalid --duration-sec: {value}";
                        return false;
                    }
                    break;
                case "--source-x":
                    if (!double.TryParse(value, out sourceX))
                    {
                        options = default!;
                        error = $"Invalid --source-x: {value}";
                        return false;
                    }
                    break;
                case "--source-y":
                    if (!double.TryParse(value, out sourceY))
                    {
                        options = default!;
                        error = $"Invalid --source-y: {value}";
                        return false;
                    }
                    break;
                case "--source-z":
                    if (!double.TryParse(value, out sourceZ))
                    {
                        options = default!;
                        error = $"Invalid --source-z: {value}";
                        return false;
                    }
                    break;
                case "--cli-output":
                    cliOutput = value.Trim().ToLowerInvariant();
                    if (cliOutput is not "live" and not "scroll")
                    {
                        options = default!;
                        error = $"Invalid --cli-output: {value}. Expected live|scroll.";
                        return false;
                    }
                    break;
                default:
                    options = default!;
                    error = $"Unknown option: {arg}";
                    return false;
            }
        }

        if (string.IsNullOrWhiteSpace(host))
        {
            options = default!;
            error = "--host must be non-empty.";
            return false;
        }
        if (port is <= 0 or > 65535)
        {
            options = default!;
            error = $"--port must be in [1,65535], got {port}.";
            return false;
        }
        if (hz is <= 0 or > 1000)
        {
            options = default!;
            error = $"--hz must be in [1,1000], got {hz}.";
            return false;
        }
        if (durationSec < 0.0)
        {
            options = default!;
            error = $"--duration-sec must be >= 0, got {durationSec}.";
            return false;
        }
        if (!double.IsFinite(sourceX) || !double.IsFinite(sourceY) || !double.IsFinite(sourceZ))
        {
            options = default!;
            error = "--source-x/--source-y/--source-z must be finite.";
            return false;
        }

        options = new BridgeOptions(host, port, hz, mode, durationSec, sourceX, sourceY, sourceZ, cliOutput);
        error = string.Empty;
        return true;
    }
}
