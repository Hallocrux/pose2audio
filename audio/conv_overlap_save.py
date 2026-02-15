"""FFT overlap-save convolver."""

from __future__ import annotations

from typing import Optional

import numpy as np


def next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


class OverlapSaveConvolver:
    def __init__(
        self,
        h: np.ndarray,
        blocksize: int,
        initial_overlap: Optional[np.ndarray] = None,
    ):
        self.h = np.asarray(h, dtype=np.float32)
        self.M = self.h.size
        self.B = int(blocksize)
        if self.B <= 0:
            raise ValueError(f"blocksize must be > 0, got {self.B}")
        if self.M <= 0:
            raise ValueError("Impulse response must have at least one sample")

        self.nfft = next_pow2(self.B + self.M - 1)
        self.H = np.fft.rfft(self.h, self.nfft)

        if self.M - 1 > 0:
            if initial_overlap is not None and initial_overlap.size == self.M - 1:
                self.overlap = initial_overlap.astype(np.float32, copy=True)
            else:
                self.overlap = np.zeros(self.M - 1, dtype=np.float32)
            self._x_ext = np.zeros(self.M - 1 + self.B, dtype=np.float32)
        else:
            self.overlap = np.zeros(0, dtype=np.float32)
            self._x_ext = np.zeros(self.B, dtype=np.float32)

    def process_block(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.size != self.B:
            raise ValueError(f"Expected block size {self.B}, got {x.size}")

        if self.M == 1:
            return (x * self.h[0]).astype(np.float32, copy=False)

        x_ext = self._x_ext
        x_ext[: self.M - 1] = self.overlap
        x_ext[self.M - 1 :] = x
        X = np.fft.rfft(x_ext, self.nfft)
        y_ext = np.fft.irfft(X * self.H, self.nfft).astype(np.float32)
        y = y_ext[self.M - 1 : self.M - 1 + self.B].copy()
        self.overlap[:] = x_ext[-(self.M - 1) :]
        return y
