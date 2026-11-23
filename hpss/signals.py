"""Synthetic signal generators with known ground truth.

All functions return float64 mono signals in [-1, 1] and are deterministic
given the RNG seed.  Variants deliberately break the morphology assumptions
of HPSS (vibrato, noisy sustain, glissando, clipped attack) so the report can
show *where* the model fails instead of asserting it always works.
"""

from __future__ import annotations

import numpy as np

_TWO_PI = 2.0 * np.pi


def _t(duration: float, sr: int) -> np.ndarray:
    return np.arange(int(round(duration * sr)), dtype=np.float64) / sr


def adsr(
    n: int,
    sr: int,
    attack: float = 0.01,
    decay: float = 0.1,
    sustain: float = 0.7,
    release: float = 0.1,
) -> np.ndarray:
    env = np.zeros(n, dtype=np.float64)
    a = max(1, int(attack * sr))
    d = max(1, int(decay * sr))
    r = max(1, int(release * sr))
    if a + d + r > n:
        a = max(1, n // 10)
        d = max(1, n // 10)
        r = max(1, n - a - d)
    env[:a] = np.linspace(0.0, 1.0, a, endpoint=False)
    env[a : a + d] = np.linspace(1.0, sustain, d, endpoint=False)
    env[a + d : n - r] = sustain
    env[n - r :] = np.linspace(sustain, 0.0, r)
    return env


def harmonic_tone(
    freq: float,
    duration: float,
    sr: int,
    amplitude: float = 0.3,
    n_harmonics: int = 12,
    vibrato_hz: float = 0.0,
    vibrato_cents: float = 0.0,
    noise: float = 0.0,
    glide_to: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Additive harmonic tone.

    ``noise`` adds inharmonic broadband energy (breaks spectral-line
    assumption).  ``vibrato_cents`` / ``glide_to`` break the stationary-pitch
    assumption.  These are the intended counterexample controls.
    """
    t = _t(duration, sr)
    n = len(t)
    rng = np.random.default_rng(seed)
    if glide_to is not None:
        base = freq * (glide_to / freq) ** (t / max(t[-1], 1e-9))
    else:
        base = np.full(n, float(freq))
    if vibrato_hz > 0 and vibrato_cents > 0:
        cents = vibrato_cents * np.sin(_TWO_PI * vibrato_hz * t)
        base = base * 2.0 ** (cents / 1200.0)
    phase = _TWO_PI * np.cumsum(base) / sr
    out = np.zeros(n, dtype=np.float64)
    for k in range(1, n_harmonics + 1):
        if freq * k >= sr / 2:
            break
        out += (1.0 / k**1.2) * np.sin(k * phase)
        if vibrato_cents > 0:
            # vibrato smears each partial: add a small sideband pair
            out += 0.02 / k * np.sin((k + 0.01) * phase)
    out *= adsr(n, sr, attack=min(0.02, duration / 10))
    out /= max(np.max(np.abs(out)), 1e-12)
    if noise > 0:
        out = out * (1.0 - noise) + noise * rng.standard_normal(n) / 3.0
        out /= max(np.max(np.abs(out)), 1e-12)
    return amplitude * out


def percussive_hit(
    duration: float,
    sr: int,
    kind: str = "snare",
    amplitude: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Short broadband transient: kick / snare / hat / click."""
    t = _t(duration, sr)
    n = len(t)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    if kind == "kick":
        f = 120.0 * np.exp(-t / 0.025) + 45.0
        body = np.sin(_TWO_PI * np.cumsum(f) / sr) * np.exp(-t / 0.08)
        click = noise * np.exp(-t / 0.004) * 0.2
        out = body + click
    elif kind == "snare":
        body = np.sin(_TWO_PI * 190.0 * t) * np.exp(-t / 0.05) * 0.4
        out = noise * np.exp(-t / 0.06) + body
    elif kind == "hat":
        out = noise * np.exp(-t / 0.012)
    elif kind == "click":
        out = noise * np.exp(-t / 0.002)
    else:
        raise ValueError(f"unknown hit kind {kind!r}")
    # crude high-pass so hats/snares are less spectrally flat at low end
    out = np.diff(np.concatenate([[0.0], out]))
    out /= max(np.max(np.abs(out)), 1e-12)
    return amplitude * out


def click_train(times: list[float], duration: float, sr: int, amplitude: float = 0.5, seed: int = 0) -> np.ndarray:
    out = np.zeros(int(round(duration * sr)), dtype=np.float64)
    hit = percussive_hit(0.12, sr, "click", amplitude=amplitude, seed=seed)
    for i, t0 in enumerate(times):
        j = int(round(t0 * sr))
        if j >= len(out):
            continue
        k = min(len(hit), len(out) - j)
        out[j : j + k] += hit[:k]
    return out


def dense_drum_pattern(
    duration: float,
    sr: int,
    bpm: float = 168.0,
    amplitude: float = 0.5,
    seed: int = 0,
) -> tuple[np.ndarray, list[float]]:
    """Fast 16th-note pattern (dense transients overlapping a sustain)."""
    beat = 60.0 / bpm
    step = beat / 4.0
    n_steps = int(duration / step)
    out = np.zeros(int(round(duration * sr)), dtype=np.float64)
    times: list[float] = []
    rng = np.random.default_rng(seed)
    for i in range(n_steps):
        t0 = i * step
        if t0 >= duration:
            break
        which = i % 8
        if which in (0, 3):
            kind = "kick"
        elif which in (2, 6):
            kind = "snare"
        elif which in (1, 5, 7):
            kind = "hat"
        else:
            kind = "click"
        amp = amplitude * float(rng.uniform(0.7, 1.0))
        hit = percussive_hit(min(0.25, duration - t0), sr, kind, amplitude=amp, seed=seed + i)
        j = int(round(t0 * sr))
        out[j : j + len(hit)] += hit
        times.append(t0)
    return out, times


def mix(*signals: np.ndarray, gains: list[float] | None = None) -> np.ndarray:
    length = max(len(s) for s in signals)
    out = np.zeros(length, dtype=np.float64)
    if gains is None:
        gains = [1.0] * len(signals)
    for g, s in zip(gains, signals):
        out[: len(s)] += g * np.asarray(s, dtype=np.float64)
    return out


def peak_normalize(x: np.ndarray, target: float = 0.95) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak <= 0:
        return np.asarray(x, dtype=np.float64)
    return np.asarray(x, dtype=np.float64) * (target / peak)


def align_to(x: np.ndarray, length: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) >= length:
        return x[:length]
    return np.pad(x, (0, length - len(x)))
