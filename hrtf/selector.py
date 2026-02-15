"""HRIR selector helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sofa_db import SofaHrirDB


@dataclass(frozen=True)
class HrtfSelection:
    hL: np.ndarray
    hR: np.ndarray
    idx: int
    dataset_az: float
    dataset_el: float


class HrtfSelector:
    def __init__(self, db: SofaHrirDB, swap_lr: bool = False):
        self.db = db
        self.swap_lr = swap_lr

    def select(self, az_deg: float, el_deg: float) -> HrtfSelection:
        hL, hR, idx = self.db.get_hrir(az_deg, el_deg, swap_lr=self.swap_lr)
        sp = self.db.source_pos[idx]
        return HrtfSelection(
            hL=hL,
            hR=hR,
            idx=idx,
            dataset_az=float(sp[0]),
            dataset_el=float(sp[1]),
        )
