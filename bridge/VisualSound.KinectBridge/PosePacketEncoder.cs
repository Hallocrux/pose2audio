using System.Buffers;
using System.Text.Json;

namespace VisualSound.KinectBridge;

internal sealed class PosePacketEncoder
{
    private readonly ArrayBufferWriter<byte> _buffer = new(256);

    public ReadOnlyMemory<byte> Encode(PoseSample sample)
    {
        _buffer.Clear();
        using (var writer = new Utf8JsonWriter(_buffer))
        {
            writer.WriteStartObject();
            writer.WriteBoolean("tracked", sample.Tracked);
            writer.WritePropertyName("position_m");
            writer.WriteStartArray();
            writer.WriteNumberValue(sample.X);
            writer.WriteNumberValue(sample.Y);
            writer.WriteNumberValue(sample.Z);
            writer.WriteEndArray();

            writer.WritePropertyName("quaternion_wxyz");
            writer.WriteStartArray();
            writer.WriteNumberValue(sample.Qw);
            writer.WriteNumberValue(sample.Qx);
            writer.WriteNumberValue(sample.Qy);
            writer.WriteNumberValue(sample.Qz);
            writer.WriteEndArray();
            writer.WriteEndObject();
        }

        return _buffer.WrittenMemory;
    }
}

