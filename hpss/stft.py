from __future__ import annotations

import numpy as np
from scipy.signal import get_window

from .config import STFTConfig


class STFT:
    """Centered short-time Fourier transform with exact least-squares inversion.

    Analysis and synthesis use the same window. Synthesis divides the
    overlap-added frames by the overlap-added squared window, so complementary
    masks reconstruct the input exactly (up to floating point) because the
    transform is linear and the inverse is a single linear operator.
    """

    def __init__(self, config: STFTConfig | None = None) -> None:
        self.config = config or STFTConfig()
        self.n_fft = self.config.n_fft
        self.hop = self.config.hop
        self.pad = self.n_fft // 2 if self.config.center else 0
        wl = self.config.win_length
        win = get_window(self.config.window, wl, fftbins=True).astype(np.float64)
        if wl < self.n_fft:
            left = (self.n_fft - wl) // 2
            padded = np.zeros(self.n_fft, dtype=np.float64)
            padded[left : left + wl] = win
            win = padded
        self.window = win
        self._window_sq = win**2

    def frame_times(self, n_frames: int, sample_rate: int) -> np.ndarray:
        return np.arange(n_frames, dtype=np.float64) * self.hop / sample_rate

    def frequencies(self, sample_rate: int) -> np.ndarray:
        return np.fft.rfftfreq(self.n_fft, d=1.0 / sample_rate)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("STFT.forward expects a mono signal")
        if x.size == 0:
            return np.zeros((self.n_fft // 2 + 1, 0), dtype=np.complex128)
        pad = self.pad
        mode = self.config.pad_mode
        if mode != "constant" and x.size <= pad:
            mode = "constant"
        if pad:
            xp = np.pad(x, (pad, pad), mode=mode)
        else:
            xp = x.copy()
        if len(xp) < self.n_fft:
            xp = np.pad(xp, (0, self.n_fft - len(xp)))
        n_frames = 1 + int(np.ceil((len(xp) - self.n_fft) / self.hop))
        needed = (n_frames - 1) * self.hop + self.n_fft
        if len(xp) < needed:
            xp = np.pad(xp, (0, needed - len(xp)))
        frames = np.lib.stride_tricks.sliding_window_view(xp, self.n_fft)[:: self.hop]
        frames = frames[:n_frames]
        return np.fft.rfft(frames * self.window, axis=1).T

    def inverse(self, spec: np.ndarray, length: int) -> np.ndarray:
        spec = np.asarray(spec)
        if spec.ndim != 2:
            raise ValueError("STFT.inverse expects a (freq, frames) spectrum")
        n_frames = spec.shape[1]
        frames = np.fft.irfft(spec.T, n=self.n_fft, axis=1) * self.window
        out_len = (n_frames - 1) * self.hop + self.n_fft
        y = np.zeros(out_len, dtype=np.float64)
        wsum = np.zeros(out_len, dtype=np.float64)
        for i in range(n_frames):
            start = i * self.hop
            y[start : start + self.n_fft] += frames[i]
            wsum[start : start + self.n_fft] += self._window_sq
        # The trimmed region is fully overlapped for hop <= n_fft/4.  The
        # maximum() only guards the outermost padded samples.
        y /= np.maximum(wsum, 1e-12)
        if self.pad:
            y = y[self.pad : self.pad + length]
        else:
            y = y[:length]
        if len(y) < length:
            y = np.pad(y, (0, length - len(y)))
        return y

    def roundtrip(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return self.inverse(self.forward(x), len(x))
