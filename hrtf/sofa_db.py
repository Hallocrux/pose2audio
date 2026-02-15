"""Minimal SOFA HRIR database loader (nearest neighbor)."""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import numpy as np
from netCDF4 import Dataset

logger = logging.getLogger(__name__)


def _pick_var(ds: Dataset, candidates: Sequence[str]):
    for name in candidates:
        if name in ds.variables:
            return ds.variables[name]
    raise KeyError(f"Cannot find any of variables: {candidates}")


def _angle_diff_deg(a: float, b: float) -> float:
    d = (a - b) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


class SofaHrirDB:
    def __init__(self, path: str, truncate_ms: Optional[float] = None):
        self.path = path
        self.ds = Dataset(path, "r")

        v_ir = _pick_var(self.ds, ["Data.IR", "Data_IR", "DataIR"])
        v_sp = _pick_var(
            self.ds, ["SourcePosition", "Source_Position", "SourcePosition_"]
        )

        sr = None
        if "Data.SamplingRate" in self.ds.variables:
            sr = float(self.ds.variables["Data.SamplingRate"][:].squeeze())
        elif "Data_SamplingRate" in self.ds.variables:
            sr = float(self.ds.variables["Data_SamplingRate"][:].squeeze())
        elif hasattr(self.ds, "Data_SamplingRate"):
            sr = float(getattr(self.ds, "Data_SamplingRate"))
        for k in ["DataSamplingRate", "SamplingRate", "Fs"]:
            if sr is None and hasattr(self.ds, k):
                sr = float(getattr(self.ds, k))
        if sr is None:
            raise KeyError(
                "Cannot find sampling rate in SOFA file (Data.SamplingRate)."
            )

        self.sr = int(round(sr))
        self.source_pos = np.array(v_sp[:], dtype=np.float32)  # [az, el, r]
        self.ir = np.array(v_ir[:], dtype=np.float32)  # (M, R, N)

        if self.ir.ndim != 3:
            raise ValueError(f"Unexpected Data.IR shape: {self.ir.shape} (need M,R,N)")
        if self.source_pos.ndim != 2 or self.source_pos.shape[0] != self.ir.shape[0]:
            raise ValueError("SourcePosition mismatch with Data.IR M dimension")
        self.M, self.R, self.N = self.ir.shape
        if self.R < 2:
            raise ValueError("Need at least 2 receivers (ears)")

        if truncate_ms is not None:
            max_n = int(round((truncate_ms / 1000.0) * self.sr))
            max_n = max(8, min(max_n, self.N))
            self.ir = self.ir[:, :, :max_n]
            self.N = max_n

        self.az_list = self.source_pos[:, 0].astype(np.float32)
        self.el_list = self.source_pos[:, 1].astype(np.float32)

        logger.info("[SOFA] loaded: %s", path)
        logger.info(
            "[SOFA] sr=%s, IR shape=(M=%s, R=%s, N=%s)",
            self.sr,
            self.M,
            self.R,
            self.N,
        )

    def close(self):
        try:
            self.ds.close()
        except (AttributeError, RuntimeError, OSError):
            pass

    def _nearest_index(self, az_deg: float, el_deg: float) -> int:
        az_deg = float(az_deg)
        el_deg = float(el_deg)
        daz = np.array(
            [_angle_diff_deg(float(a), az_deg) for a in self.az_list], dtype=np.float32
        )
        delv = (self.el_list - el_deg).astype(np.float32)
        dist2 = daz * daz + 2.0 * delv * delv
        return int(np.argmin(dist2))

    def get_hrir(
        self, az_deg: float, el_deg: float, swap_lr: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        idx = self._nearest_index(az_deg, el_deg)
        h0 = self.ir[idx, 0, :].copy()
        h1 = self.ir[idx, 1, :].copy()
        if swap_lr:
            return h1, h0, idx
        return h0, h1, idx
