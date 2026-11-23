"""Audio file I/O for stems and reports.

Stems are written as 32-bit float and are **never renormalized**.  Float WAV can
legitimately store samples outside [-1, 1], so an intentional policy overshoot
(see ``shared`` masks) survives to disk where it can be measured and heard.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import soundfile as sf


def write_wav(path: str | Path, x: np.ndarray, sr: int, subtype: str = "FLOAT") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak > 1.0:
        warnings.warn(
            f"{path.name}: peak {peak:.3f} > 1.0; writing as float32 without "
            "normalization so the level relation is preserved.",
            stacklevel=2,
        )
    sf.write(str(path), x.astype(np.float32), sr, subtype=subtype)
    return path


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), dtype="float64", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return np.asarray(x, dtype=np.float64), int(sr)


def mix_stems(stems: dict[str, np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(v, dtype=np.float64).reshape(-1) for v in stems.values()]
    n = max(len(a) for a in arrays)
    out = np.zeros(n)
    for a in arrays:
        out[: len(a)] += a
    return out
