using System;
using System.Diagnostics;
using System.Globalization;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Microsoft.Kinect;
using Microsoft.Kinect.Face;

namespace VisualSound.KinectBridge.KinectSdk.HD
{
    internal sealed class Program
    {
        private static int Main(string[] args)
        {
            BridgeOptions options;
            string error;
            if (!BridgeOptions.TryParse(args, out options, out error))
            {
                if (string.Equals(error, "help", StringComparison.OrdinalIgnoreCase))
                {
                    Console.WriteLine(BridgeOptions.HelpText);
                    return 0;
                }

                Console.Error.WriteLine(error);
                Console.Error.WriteLine();
                Console.Error.WriteLine(BridgeOptions.HelpText);
                return 2;
            }

            try
            {
                using (var bridge = new KinectSdkBridge(options))
                {
                    Console.CancelKeyPress += delegate(object sender, ConsoleCancelEventArgs e)
                    {
                        e.Cancel = true;
                        bridge.RequestStop();
                    };
                    return bridge.Run();
                }
            }
            catch (Exception ex)
            {
                string message;
                try
                {
                    message = ex.Message;
                }
                catch
                {
                    message = "<unavailable>";
                }
                Console.Error.WriteLine("Kinect bridge failed: " + message);
                return 1;
            }
        }
    }

    internal sealed class KinectSdkBridge : IDisposable
    {
        private readonly BridgeOptions _options;
        private readonly UdpClient _udp;
        private readonly IPEndPoint _endpoint;
        private readonly KinectSensor _sensor;
        private readonly BodyFrameReader _bodyReader;
        private readonly HighDefinitionFaceFrameSource _hdFaceSource;
        private readonly HighDefinitionFaceFrameReader _hdFaceReader;
        private readonly FaceAlignment _faceAlignment;
        private readonly Body[] _bodies;
        private readonly StringBuilder _jsonBuilder;
        private readonly bool _liveCli;
        private int _statusLastLen;

        private bool _stopRequested;
        private ulong _trackedBodyId;
        private bool _hasHead;
        private bool _hasBodyQ;
        private bool _hasFaceQ;
        private CameraSpacePoint _head;
        private Vector4 _bodyQ;
        private Vector4 _faceQ;
        private PoseSample _last;

        public KinectSdkBridge(BridgeOptions options)
        {
            _options = options;
            _endpoint = new IPEndPoint(ResolveIPv4(options.Host), options.Port);
            _udp = new UdpClient();
            _jsonBuilder = new StringBuilder(256);
            _liveCli = string.Equals(options.CliOutput, "live", StringComparison.OrdinalIgnoreCase);

            _sensor = KinectSensor.GetDefault();
            if (_sensor == null)
            {
                throw new InvalidOperationException("Kinect V2 sensor not found.");
            }

            _bodyReader = _sensor.BodyFrameSource.OpenReader();
            if (_bodyReader == null)
            {
                throw new InvalidOperationException("Failed to open BodyFrameReader.");
            }

            _hdFaceSource = new HighDefinitionFaceFrameSource(_sensor);
            _hdFaceReader = _hdFaceSource.OpenReader();
            if (_hdFaceReader == null)
            {
                throw new InvalidOperationException("Failed to open HighDefinitionFaceFrameReader.");
            }
            _faceAlignment = new FaceAlignment();

            _bodies = new Body[_sensor.BodyFrameSource.BodyCount];
            _last = PoseSample.Identity();
            _sensor.Open();
        }

        public int Run()
        {
            Console.WriteLine(
                "[{0}] kinect-sdk-hd bridge start udp={1}:{2} hz={3} duration={4}",
                DateTimeOffset.Now.ToString("O"),
                _options.Host,
                _options.Port,
                _options.Hz,
                _options.DurationSec.ToString("0.###", CultureInfo.InvariantCulture));
            Console.WriteLine("CLI output mode: {0}", _options.CliOutput);
            Console.WriteLine("Press Ctrl+C to stop.");

            var sw = Stopwatch.StartNew();
            var intervalMs = Math.Max(1, (int)Math.Round(1000.0 / _options.Hz));
            var nextTickMs = sw.ElapsedMilliseconds;
            var stopAt = _options.DurationSec > 0.0
                ? DateTime.UtcNow.AddSeconds(_options.DurationSec)
                : (DateTime?)null;
            var sent = 0L;

            while (!_stopRequested)
            {
                if (stopAt.HasValue && DateTime.UtcNow >= stopAt.Value)
                {
                    break;
                }

                PollKinect();
                ComposeLatestSample();
                SendSample(_last);
                sent++;

                if (sent % _options.Hz == 0)
                {
                    var elapsed = sw.Elapsed.TotalSeconds;
                    var rate = elapsed > 0.0 ? sent / elapsed : 0.0;
                    double az;
                    double el;
                    var hasHrtf = TryComputeHrtfAzEl(_last, out az, out el);
                    var line = string.Format(
                        CultureInfo.InvariantCulture,
                        "[{0}] sent={1} avg_rate={2:F1}Hz tracked={3} head_xyz=({4:F3},{5:F3},{6:F3}) hrtf_az_el=({7})",
                        DateTimeOffset.Now.ToString("O"),
                        sent,
                        rate,
                        _last.Tracked,
                        _last.X,
                        _last.Y,
                        _last.Z,
                        hasHrtf
                            ? string.Format(CultureInfo.InvariantCulture, "{0:F2},{1:F2}", az, el)
                            : "n/a");
                    PrintStatus(line);
                }

                nextTickMs += intervalMs;
                var sleep = (int)(nextTickMs - sw.ElapsedMilliseconds);
                if (sleep > 0)
                {
                    Thread.Sleep(sleep);
                }
                else if (sleep < -intervalMs)
                {
                    nextTickMs = sw.ElapsedMilliseconds;
                }
            }

            EndStatusLine();
            Console.WriteLine("[{0}] bridge stopped.", DateTimeOffset.Now.ToString("O"));
            return 0;
        }

        public void RequestStop()
        {
            _stopRequested = true;
        }

        private void PollKinect()
        {
            using (var bodyFrame = _bodyReader.AcquireLatestFrame())
            {
                if (bodyFrame != null)
                {
                    bodyFrame.GetAndRefreshBodyData(_bodies);
                    var body = SelectNearestTrackedBody(_bodies);
                    if (body == null)
                    {
                        _trackedBodyId = 0;
                        _hasHead = false;
                        _hasBodyQ = false;
                        _hasFaceQ = false;
                        _hdFaceSource.TrackingId = 0;
                    }
                    else
                    {
                        _trackedBodyId = body.TrackingId;
                        if (!_hdFaceSource.IsTrackingIdValid || _hdFaceSource.TrackingId != _trackedBodyId)
                        {
                            _hdFaceSource.TrackingId = _trackedBodyId;
                            _hasFaceQ = false;
                        }

                        var head = body.Joints[JointType.Head];
                        _hasHead = head.TrackingState != TrackingState.NotTracked;
                        if (_hasHead)
                        {
                            _head = head.Position;
                        }

                        var q = body.JointOrientations[JointType.Head].Orientation;
                        _hasBodyQ = IsFinite(q);
                        if (_hasBodyQ)
                        {
                            _bodyQ = Normalize(q);
                        }
                    }
                }
            }

            if (_trackedBodyId == 0 || !_hdFaceSource.IsTrackingIdValid)
            {
                _hasFaceQ = false;
                return;
            }

            using (var faceFrame = _hdFaceReader.AcquireLatestFrame())
            {
                if (faceFrame == null)
                {
                    _hasFaceQ = false;
                    return;
                }

                faceFrame.GetAndRefreshFaceAlignmentResult(_faceAlignment);
                if (faceFrame.FaceAlignmentQuality != FaceAlignmentQuality.High)
                {
                    _hasFaceQ = false;
                    return;
                }

                var fq = _faceAlignment.FaceOrientation;
                _hasFaceQ = IsFinite(fq);
                if (_hasFaceQ)
                {
                    _faceQ = Normalize(fq);
                }
            }
        }

        private void ComposeLatestSample()
        {
            if (!_hasHead || !_hasFaceQ)
            {
                if (_last.Tracked)
                {
                    _last = new PoseSample(
                        false,
                        _last.X,
                        _last.Y,
                        _last.Z,
                        _last.Qw,
                        _last.Qx,
                        _last.Qy,
                        _last.Qz);
                }
                else
                {
                    _last = PoseSample.Identity();
                }
                return;
            }

            _last = new PoseSample(true, _head.X, _head.Y, _head.Z, _faceQ.W, _faceQ.X, _faceQ.Y, _faceQ.Z);
        }

        private void SendSample(PoseSample s)
        {
            _jsonBuilder.Clear();
            _jsonBuilder.Append("{\"tracked\":");
            _jsonBuilder.Append(s.Tracked ? "true" : "false");
            _jsonBuilder.Append(",\"position_m\":[");
            AppendNumber(s.X);
            _jsonBuilder.Append(',');
            AppendNumber(s.Y);
            _jsonBuilder.Append(',');
            AppendNumber(s.Z);
            _jsonBuilder.Append("],\"quaternion_wxyz\":[");
            AppendNumber(s.Qw);
            _jsonBuilder.Append(',');
            AppendNumber(s.Qx);
            _jsonBuilder.Append(',');
            AppendNumber(s.Qy);
            _jsonBuilder.Append(',');
            AppendNumber(s.Qz);
            _jsonBuilder.Append("]}");

            var bytes = Encoding.UTF8.GetBytes(_jsonBuilder.ToString());
            _udp.Send(bytes, bytes.Length, _endpoint);
        }

        private void AppendNumber(double v)
        {
            _jsonBuilder.Append(v.ToString("R", CultureInfo.InvariantCulture));
        }

        private bool TryComputeHrtfAzEl(PoseSample pose, out double az, out double el)
        {
            az = 0.0;
            el = 0.0;
            if (!pose.Tracked)
            {
                return false;
            }

            var relX = _options.SourceX - pose.X;
            var relY = _options.SourceY - pose.Y;
            var relZ = _options.SourceZ - pose.Z;
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

        private static Body SelectNearestTrackedBody(Body[] bodies)
        {
            Body best = null;
            var bestZ = double.MaxValue;
            for (var i = 0; i < bodies.Length; i++)
            {
                var body = bodies[i];
                if (body == null || !body.IsTracked)
                {
                    continue;
                }

                var head = body.Joints[JointType.Head];
                if (head.TrackingState == TrackingState.NotTracked)
                {
                    continue;
                }

                var z = head.Position.Z;
                if (z < bestZ)
                {
                    bestZ = z;
                    best = body;
                }
            }

            return best;
        }

        private static bool IsFinite(Vector4 q)
        {
            return IsFinite(q.W) && IsFinite(q.X) && IsFinite(q.Y) && IsFinite(q.Z);
        }

        private static bool IsFinite(float x)
        {
            return !float.IsNaN(x) && !float.IsInfinity(x);
        }

        private void PrintStatus(string line)
        {
            if (!_liveCli)
            {
                Console.WriteLine(line);
                return;
            }

            if (line.Length < _statusLastLen)
            {
                line = line + new string(' ', _statusLastLen - line.Length);
            }
            _statusLastLen = line.Length;
            Console.Write("\r" + line);
        }

        private void EndStatusLine()
        {
            if (_liveCli && _statusLastLen > 0)
            {
                Console.WriteLine();
                _statusLastLen = 0;
            }
        }

        private static Vector4 Normalize(Vector4 q)
        {
            var n = Math.Sqrt((q.W * q.W) + (q.X * q.X) + (q.Y * q.Y) + (q.Z * q.Z));
            if (n <= 1e-12)
            {
                return IdentityQuaternion();
            }

            var inv = (float)(1.0 / n);
            return new Vector4
            {
                W = q.W * inv,
                X = q.X * inv,
                Y = q.Y * inv,
                Z = q.Z * inv
            };
        }

        private static Vector4 IdentityQuaternion()
        {
            return new Vector4 { W = 1f, X = 0f, Y = 0f, Z = 0f };
        }

        private static IPAddress ResolveIPv4(string host)
        {
            IPAddress ip;
            if (IPAddress.TryParse(host, out ip))
            {
                return ip;
            }

            var addresses = Dns.GetHostAddresses(host);
            for (var i = 0; i < addresses.Length; i++)
            {
                if (addresses[i].AddressFamily == AddressFamily.InterNetwork)
                {
                    return addresses[i];
                }
            }

            throw new InvalidOperationException("Unable to resolve IPv4 host: " + host);
        }

        public void Dispose()
        {
            _hdFaceReader.Dispose();
            _bodyReader.Dispose();
            if (_sensor.IsOpen)
            {
                _sensor.Close();
            }
            _udp.Dispose();
        }
    }

    internal sealed class BridgeOptions
    {
        public string Host;
        public int Port;
        public int Hz;
        public double DurationSec;
        public double SourceX;
        public double SourceY;
        public double SourceZ;
        public string CliOutput;

        public static string HelpText =
@"Usage:
  VisualSound.KinectBridge.KinectSdk.HD [--host 127.0.0.1] [--port 24567] [--hz 60] [--duration-sec 0] [--source-x 0] [--source-y 0] [--source-z 1] [--cli-output live|scroll]

Notes:
  - Requires Kinect for Windows SDK v2 and Kinect V2 sensor/runtime.
  - Sends UDP JSON packets:
    {""tracked"":true,""position_m"":[x,y,z],""quaternion_wxyz"":[w,x,y,z]}
";

        public static bool TryParse(string[] args, out BridgeOptions options, out string error)
        {
            options = new BridgeOptions
            {
                Host = "127.0.0.1",
                Port = 24567,
                Hz = 60,
                DurationSec = 0.0,
                SourceX = 0.0,
                SourceY = 0.0,
                SourceZ = 1.0,
                CliOutput = "live"
            };

            for (var i = 0; i < args.Length; i++)
            {
                var arg = args[i];
                if (arg == "--help" || arg == "-h")
                {
                    error = "help";
                    return false;
                }

                if (!arg.StartsWith("--", StringComparison.Ordinal))
                {
                    error = "Unknown argument: " + arg;
                    return false;
                }

                if (i + 1 >= args.Length)
                {
                    error = "Missing value for option: " + arg;
                    return false;
                }

                var value = args[++i];
                switch (arg)
                {
                    case "--host":
                        options.Host = value.Trim();
                        break;
                    case "--port":
                        if (!int.TryParse(value, out options.Port))
                        {
                            error = "Invalid --port: " + value;
                            return false;
                        }
                        break;
                    case "--hz":
                        if (!int.TryParse(value, out options.Hz))
                        {
                            error = "Invalid --hz: " + value;
                            return false;
                        }
                        break;
                    case "--duration-sec":
                        if (!double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out options.DurationSec))
                        {
                            error = "Invalid --duration-sec: " + value;
                            return false;
                        }
                        break;
                    case "--source-x":
                        if (!double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out options.SourceX))
                        {
                            error = "Invalid --source-x: " + value;
                            return false;
                        }
                        break;
                    case "--source-y":
                        if (!double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out options.SourceY))
                        {
                            error = "Invalid --source-y: " + value;
                            return false;
                        }
                        break;
                    case "--source-z":
                        if (!double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out options.SourceZ))
                        {
                            error = "Invalid --source-z: " + value;
                            return false;
                        }
                        break;
                    case "--cli-output":
                        options.CliOutput = value.Trim().ToLowerInvariant();
                        if (options.CliOutput != "live" && options.CliOutput != "scroll")
                        {
                            error = "Invalid --cli-output: " + value + ". Expected live|scroll.";
                            return false;
                        }
                        break;
                    default:
                        error = "Unknown option: " + arg;
                        return false;
                }
            }

            if (string.IsNullOrWhiteSpace(options.Host))
            {
                error = "--host must be non-empty.";
                return false;
            }
            if (options.Port <= 0 || options.Port > 65535)
            {
                error = "--port must be in [1,65535], got " + options.Port;
                return false;
            }
            if (options.Hz <= 0 || options.Hz > 1000)
            {
                error = "--hz must be in [1,1000], got " + options.Hz;
                return false;
            }
            if (options.DurationSec < 0.0)
            {
                error = "--duration-sec must be >= 0.";
                return false;
            }
            if (double.IsNaN(options.SourceX) || double.IsInfinity(options.SourceX) ||
                double.IsNaN(options.SourceY) || double.IsInfinity(options.SourceY) ||
                double.IsNaN(options.SourceZ) || double.IsInfinity(options.SourceZ))
            {
                error = "--source-x/--source-y/--source-z must be finite.";
                return false;
            }
            if (options.CliOutput != "live" && options.CliOutput != "scroll")
            {
                error = "--cli-output must be live or scroll.";
                return false;
            }

            error = null;
            return true;
        }
    }

    internal struct PoseSample
    {
        public bool Tracked;
        public double X;
        public double Y;
        public double Z;
        public double Qw;
        public double Qx;
        public double Qy;
        public double Qz;

        public PoseSample(bool tracked, double x, double y, double z, double qw, double qx, double qy, double qz)
        {
            Tracked = tracked;
            X = x;
            Y = y;
            Z = z;
            Qw = qw;
            Qx = qx;
            Qy = qy;
            Qz = qz;
        }

        public static PoseSample Identity()
        {
            return new PoseSample(false, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
        }
    }
}
