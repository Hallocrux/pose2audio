import numpy as np
import pytest

from audio.conv_overlap_save import OverlapSaveConvolver


def _run_stream(convolver: OverlapSaveConvolver, x: np.ndarray, block: int) -> np.ndarray:
    outputs = []
    for i in range(0, len(x), block):
        chunk = x[i : i + block]
        outputs.append(convolver.process_block(chunk))
    return np.concatenate(outputs)


def test_overlap_save_matches_linear_convolution_prefix():
    rng = np.random.default_rng(0)
    block = 64
    x = rng.normal(size=block * 8).astype(np.float32)
    h = rng.normal(size=31).astype(np.float32)

    conv = OverlapSaveConvolver(h, block)
    y = _run_stream(conv, x, block)
    y_ref = np.convolve(x, h)[: x.size].astype(np.float32)
    np.testing.assert_allclose(y, y_ref, atol=2e-5, rtol=1e-4)


def test_single_tap_impulse_response_is_scaling():
    block = 32
    x = np.linspace(-1.0, 1.0, block, dtype=np.float32)
    conv = OverlapSaveConvolver(np.array([0.25], dtype=np.float32), block)
    y = conv.process_block(x)
    np.testing.assert_allclose(y, 0.25 * x)


def test_blocksize_mismatch_raises_value_error():
    conv = OverlapSaveConvolver(np.array([1.0, 0.0, 0.0], dtype=np.float32), 16)
    with pytest.raises(ValueError):
        conv.process_block(np.zeros(8, dtype=np.float32))

