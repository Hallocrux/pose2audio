import numpy as np

from hrtf.sofa_db import SofaHrirDB, _angle_diff_deg


def test_angle_diff_wraps_correctly():
    assert _angle_diff_deg(350.0, 10.0) == -20.0
    assert _angle_diff_deg(10.0, 350.0) == 20.0
    assert _angle_diff_deg(0.0, 180.0) == 180.0


def test_nearest_index_prefers_close_azimuth_and_elevation():
    db = SofaHrirDB.__new__(SofaHrirDB)
    db.az_list = np.array([0.0, 45.0, 90.0], dtype=np.float32)
    db.el_list = np.array([0.0, 0.0, 10.0], dtype=np.float32)
    idx = db._nearest_index(50.0, 2.0)
    assert idx == 1

