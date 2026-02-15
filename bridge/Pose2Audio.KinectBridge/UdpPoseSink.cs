using System.Net;
using System.Net.Sockets;

namespace Pose2Audio.KinectBridge;

internal sealed class UdpPoseSink : IPoseSink
{
    private readonly Socket _socket;
    private readonly EndPoint _endpoint;
    private readonly PosePacketEncoder _encoder = new();

    public UdpPoseSink(string host, int port)
    {
        _socket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
        var ip = IPAddress.TryParse(host, out var parsed)
            ? parsed
            : Dns.GetHostAddresses(host).First(ip0 => ip0.AddressFamily == AddressFamily.InterNetwork);
        _endpoint = new IPEndPoint(ip, port);
    }

    public async ValueTask SendAsync(PoseSample sample, CancellationToken cancellationToken)
    {
        var payload = _encoder.Encode(sample);
        await _socket.SendToAsync(payload, SocketFlags.None, _endpoint, cancellationToken);
    }

    public ValueTask DisposeAsync()
    {
        _socket.Dispose();
        return ValueTask.CompletedTask;
    }
}

