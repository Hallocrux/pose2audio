"""Coordinate helpers for azimuth/elevation in degrees."""

from __future__ import annotations

import math

import numpy as np


def vec_to_az_el_deg(v: np.ndarray) -> tuple[float, float]:
    """
    Acoustic angle convention (single source of truth):
    - input vector is in acoustic head frame
    - acoustic axes are (x right, y up, z front)

    For vector v in (x right, y up, z front):
      az: atan2(x, z)  => 0=forward, +90=right
      el: atan2(y, sqrt(x^2+z^2))
    """
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    az = math.degrees(math.atan2(x, z))
    el = math.degrees(math.atan2(y, math.sqrt(x * x + z * z) + 1e-12))
    return az, el


def az_el_to_vec(az_deg: float, el_deg: float) -> np.ndarray:
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    x = math.sin(az) * math.cos(el)
    y = math.sin(el)
    z = math.cos(az) * math.cos(el)
    return np.array([x, y, z], dtype=np.float64)
