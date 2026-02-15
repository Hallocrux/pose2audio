"""Audio sources (file or synthetic)."""

from __future__ import annotations

import logging
import math

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def linear_resample_mono(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Simple linear resample (OK for MVP; not high-fidelity)."""
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    x = x.astype(np.float32, copy=False)
    ratio = sr_out / sr_in
    n_out = int(round(x.shape[0] * ratio))
    if n_out <= 1:
        return np.zeros(1, dtype=np.float32)
    t_in = np.linspace(0.0, 1.0, num=x.shape[0], endpoint=False, dtype=np.float32)
    t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float32)
    y = np.interp(t_out, t_in, x).astype(np.float32)
    return y


class AudioFileSource:
    """
    Preload audio file into memory. Callback pulls frames by slicing.
    Output is mono float32.
    """

    def __init__(
        self,
        path: str,
        target_sr: int,
        mono: str = "avg",
        loop: bool = True,
        gain: float = 1.0,
    ):
        data, sr_in = sf.read(path, always_2d=True, dtype="float32")  # shape (T, C)
        if data.shape[1] == 1:
            mono_data = data[:, 0]
        else:
            if mono == "left":
                mono_data = data[:, 0]
            elif mono == "right":
                mono_data = data[:, 1]
            else:
                mono_data = np.mean(data[:, :2], axis=1)

        mono_data = np.clip(mono_data * float(gain), -1.0, 1.0).astype(np.float32)

        if sr_in != target_sr:
            mono_data = linear_resample_mono(mono_data, sr_in, target_sr)
            logger.info("[AUDIO] resampled %s Hz -> %s Hz (linear)", sr_in, target_sr)

        self.x = mono_data
        self.sr = target_sr
        self.loop = loop
        self.pos = 0

        dur = len(self.x) / self.sr
        logger.info(
            "[AUDIO] loaded: %s | len=%s samples | sr=%s | dur=%.2fs | loop=%s",
            path,
            len(self.x),
            self.sr,
            dur,
            self.loop,
        )

    def next(self, frames: int) -> np.ndarray:
        # Return exactly 'frames' samples; if EOF and not looping, pad zeros.
        out = np.zeros(frames, dtype=np.float32)
        if len(self.x) == 0:
            return out

        n = frames
        i = 0
        while i < n:
            remain = len(self.x) - self.pos
            if remain <= 0:
                if self.loop:
                    self.pos = 0
                    continue
                else:
                    break
            take = min(n - i, remain)
            out[i : i + take] = self.x[self.pos : self.pos + take]
            self.pos += take
            i += take
        return out


class SineSource:
    def __init__(self, sr: int, freq: float, amp: float = 0.2):
        self.sr = sr
        self.freq = float(freq)
        self.amp = float(amp)
        self.phase = 0.0

    def next(self, frames: int) -> np.ndarray:
        t = np.arange(frames, dtype=np.float32) / self.sr
        y = np.sin(2.0 * math.pi * self.freq * t + self.phase).astype(np.float32)
        self.phase = float(
            (self.phase + 2.0 * math.pi * self.freq * (frames / self.sr))
            % (2.0 * math.pi)
        )
        return self.amp * y
