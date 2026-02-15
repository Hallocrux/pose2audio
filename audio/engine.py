"""Real-time binaural engine with crossfaded convolution."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .conv_overlap_save import OverlapSaveConvolver


class BinauralEngine:
    """
    No locks in callback:
      - Control plane prepares new convolvers, assigns next_prepared (atomic pointer swap).
      - Audio thread consumes next_prepared at block boundary and starts crossfade.
    """

    def __init__(
        self,
        sr: int,
        block: int,
        fade_ms: float,
        initL: OverlapSaveConvolver,
        initR: OverlapSaveConvolver,
    ):
        self.sr = sr
        self.block = block
        self.fade_len = max(1, int(round((fade_ms / 1000.0) * sr)))
        self.fade_pos = 0

        self.convL = initL
        self.convR = initR

        self.pending: Optional[Tuple[OverlapSaveConvolver, OverlapSaveConvolver]] = None
        self.next_prepared: Optional[
            Tuple[OverlapSaveConvolver, OverlapSaveConvolver]
        ] = None

        # Reused buffers to avoid per-block allocations in the audio callback path.
        self._out = np.zeros((self.block, 2), dtype=np.float32)
        self._idx = np.arange(self.block, dtype=np.float32)
        self._alpha = np.zeros(self.block, dtype=np.float32)
        self._tmpL = np.zeros(self.block, dtype=np.float32)
        self._tmpR = np.zeros(self.block, dtype=np.float32)

    def submit_prepared(
        self, newL: OverlapSaveConvolver, newR: OverlapSaveConvolver
    ) -> None:
        self.next_prepared = (newL, newR)

    def _consume_next_if_any(self):
        nxt = self.next_prepared
        if nxt is None:
            return
        self.next_prepared = None
        self.pending = nxt
        self.fade_pos = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        self._consume_next_if_any()
        n = x.size

        if self.pending is None:
            yL = self.convL.process_block(x)
            yR = self.convR.process_block(x)
            self._out[:n, 0] = yL
            self._out[:n, 1] = yR
            return self._out[:n]

        newL, newR = self.pending
        yL_a = self.convL.process_block(x)
        yR_a = self.convR.process_block(x)
        yL_b = newL.process_block(x)
        yR_b = newR.process_block(x)

        alpha = self._alpha[:n]
        np.add(self._idx[:n], float(self.fade_pos), out=alpha)
        np.divide(alpha, float(self.fade_len), out=alpha)
        np.clip(alpha, 0.0, 1.0, out=alpha)

        np.subtract(yL_b, yL_a, out=self._tmpL[:n])
        np.multiply(self._tmpL[:n], alpha, out=self._tmpL[:n])
        np.add(yL_a, self._tmpL[:n], out=self._out[:n, 0])

        np.subtract(yR_b, yR_a, out=self._tmpR[:n])
        np.multiply(self._tmpR[:n], alpha, out=self._tmpR[:n])
        np.add(yR_a, self._tmpR[:n], out=self._out[:n, 1])

        self.fade_pos += n
        if self.fade_pos >= self.fade_len:
            self.convL, self.convR = newL, newR
            self.pending = None

        return self._out[:n]
