import numpy as np

from math3d.coords import az_el_to_vec, vec_to_az_el_deg


def _angle_diff_deg(a: float, b: float) -> float:
    d = (a - b) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


def test_az_el_roundtrip():
    cases = [
        (0.0, 0.0),
        (30.0, 10.0),
        (-60.0, -20.0),
        (179.0, 45.0),
    ]
    for az, el in cases:
        v = az_el_to_vec(az, el)
        az2, el2 = vec_to_az_el_deg(v)
        assert abs(_angle_diff_deg(az2, az)) < 1e-6
        assert abs(el2 - el) < 1e-6


def test_vec_to_az_el_forward():
    az, el = vec_to_az_el_deg(np.array([0.0, 0.0, 1.0], dtype=np.float64))
    assert abs(az) < 1e-6
    assert abs(el) < 1e-6
