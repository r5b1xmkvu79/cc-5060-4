"""Matplotlib figures: spectrograms, supports, masks, ambiguity, summaries.

Uses the non-interactive Agg backend so figures can be produced on a server.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from .decompose import HPSSResult

_MASK_CMAP = LinearSegmentedColormap.from_list(
    "mask", ["#0b0b12", "#1f4e79", "#2ea3c9", "#8ff0d0", "#f7f7d0"]
)
_DIV_CMAP = LinearSegmentedColormap.from_list(
    "div", ["#3b6fd4", "#12141c", "#e0a33b"]
)


def _db(x: np.ndarray, floor_db: float = -90.0) -> np.ndarray:
    ref = np.max(x) if x.size else 1.0
    if ref <= 0:
        return np.full_like(x, floor_db, dtype=np.float64)
    return np.maximum(20.0 * np.log10(np.maximum(x, 1e-12) / ref), floor_db)


def _imshow(ax, data, sr, hop, extent_time, vmin=None, vmax=None, cmap="magma", label=""):
    ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=(0.0, extent_time, 0.0, sr / 2.0),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax.set_ylabel("Hz")
    ax.set_xlabel("s")
    if label:
        ax.set_title(label, fontsize=9, loc="left")


def plot_decomposition(
    result: HPSSResult,
    path: str | Path,
    title: str = "",
    true_spectra: tuple[np.ndarray, np.ndarray] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mag_db = _db(result.magnitude)
    dur = result.input_length / result.sample_rate
    hop = result.stft.hop
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), constrained_layout=True)

    _imshow(axes[0, 0], mag_db, result.sample_rate, hop, dur, vmin=-80, vmax=0,
            label="mixture |STFT| (dB)")
    _imshow(axes[0, 1], _db(result.harmonic_support), result.sample_rate, hop, dur,
            vmin=-80, vmax=0, label="harmonic support (time median)")
    _imshow(axes[1, 0], _db(result.percussive_support), result.sample_rate, hop, dur,
            vmin=-80, vmax=0, label="percussive support (freq median)")
    _imshow(axes[1, 1], result.ambiguity, result.sample_rate, hop, dur, vmin=0, vmax=1,
            cmap=_DIV_CMAP, label="ambiguity (both supports strong)")

    _imshow(axes[2, 0], result.mask_harmonic, result.sample_rate, hop, dur, vmin=0, vmax=1,
            cmap=_MASK_CMAP, label=f"mask harmonic ({result.config.shared_policy})")
    _imshow(axes[2, 1], result.mask_percussive, result.sample_rate, hop, dur, vmin=0, vmax=1,
            cmap=_MASK_CMAP, label=f"mask percussive ({result.config.shared_policy})")

    _imshow(axes[3, 0], result.mask_residual, result.sample_rate, hop, dur, vmin=0, vmax=1,
            cmap=_MASK_CMAP, label="mask residual (always delivered)")
    ax = axes[3, 1]
    ax.plot(result.times, np.sum(result.magnitude**2, axis=0), lw=0.8, color="#222")
    ax.set_title("frame energy / s", fontsize=9, loc="left")
    ax.set_xlabel("s")
    ax.set_xlim(0, dur)
    ax.grid(alpha=0.2)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_masks_only(result: HPSSResult, path: str | Path, title: str = "") -> Path:
    """Compact 3-panel mask figure for dense comparison."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dur = result.input_length / result.sample_rate
    hop = result.stft.hop
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), constrained_layout=True)
    for ax, (m, name) in zip(
        axes,
        [
            (result.mask_harmonic, "M_h"),
            (result.mask_percussive, "M_p"),
            (result.mask_residual, "M_r"),
        ],
    ):
        _imshow(ax, m, result.sample_rate, hop, dur, vmin=0, vmax=1, cmap=_MASK_CMAP, label=name)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_scenario_waveforms(
    scenario, result: HPSSResult, path: str | Path
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sr = scenario.sample_rate
    t = np.arange(len(scenario.mixture)) / sr
    fig, axes = plt.subplots(3, 1, figsize=(13, 7), constrained_layout=True, sharex=True)
    axes[0].plot(t, scenario.mixture, lw=0.5, color="#111")
    axes[0].set_title("mixture", loc="left", fontsize=9)
    axes[1].plot(t, scenario.true_harmonic, lw=0.5, color="#2b6cb0", label="true harmonic")
    axes[1].plot(t, scenario.true_percussive, lw=0.5, color="#c05621", label="true percussive")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_title("ground truth", loc="left", fontsize=9)
    axes[2].plot(t, result.harmonic, lw=0.5, color="#2b6cb0", label="est harmonic")
    axes[2].plot(t, result.percussive, lw=0.5, color="#c05621", label="est percussive")
    axes[2].plot(t, result.residual, lw=0.5, color="#6b7280", label="est residual")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_title("estimate", loc="left", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_summary(rows: list[dict], path: str | Path) -> Path:
    """Bar charts: mean SDR, recomposition error, residual fraction, ambiguity."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def sdr_of(r):
        v = r.get("bss", {}).get("mean_sdr_db")
        return float(v) if v is not None and np.isfinite(v) else np.nan

    names = [r["name"] for r in rows]
    sdr = [sdr_of(r) for r in rows]
    rec = [r["energy"]["recomposition_error_db"] for r in rows]
    resid = [r["energy"]["residual_energy_fraction"] for r in rows]
    amb = [r["diagnostic"]["ambiguous_energy_fraction"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    panels = [
        (axes[0, 0], sdr, "mean SDR (dB, residual excluded)", "#2b6cb0"),
        (axes[0, 1], rec, "recomposition error (dB, lower better)", "#2f855a"),
        (axes[1, 0], resid, "residual energy fraction", "#6b7280"),
        (axes[1, 1], amb, "ambiguous energy fraction", "#b7791f"),
    ]
    for ax, vals, label, color in panels:
        y = np.arange(len(names))
        ax.barh(y, vals, color=color, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(label, fontsize=9)
        ax.grid(axis="x", alpha=0.25)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
