namespace VisualSound.KinectBridge;

internal readonly record struct PoseSample(
    bool Tracked,
    double X,
    double Y,
    double Z,
    double Qw,
    double Qx,
    double Qy,
    double Qz)
{
    public static PoseSample Identity => new(
        Tracked: false,
        X: 0.0,
        Y: 0.0,
        Z: 0.0,
        Qw: 1.0,
        Qx: 0.0,
        Qy: 0.0,
        Qz: 0.0);
}

