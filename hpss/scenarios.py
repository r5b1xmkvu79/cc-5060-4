"""Known-ground-truth scenarios.

Each scenario returns the true harmonic component, true percussive component,
the mixture, and metadata.  Standard scenarios satisfy the HPSS morphology
assumption (smooth tonal lines + broadband transients).  ``counterexample``
scenarios intentionally violate it so the report can show failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import signals


@dataclass
class Scenario:
    name: str
    sample_rate: int
    mixture: np.ndarray
    true_harmonic: np.ndarray
    true_percussive: np.ndarray
    harmonic_gain: float = 1.0
    percussive_gain: float = 1.0
    notes: str = ""
    tags: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.mixture) / self.sample_rate


def _finalize(
    name: str,
    sr: int,
    h: np.ndarray,
    p: np.ndarray,
    h_gain: float,
    p_gain: float,
    notes: str,
    tags: list[str],
    meta: dict | None = None,
) -> Scenario:
    n = max(len(h), len(p))
    h = signals.align_to(h * h_gain, n)
    p = signals.align_to(p * p_gain, n)
    mix = signals.peak_normalize(h + p, 0.95)
    scale = float(np.max(np.abs(h + p))) or 1.0
    return Scenario(
        name=name,
        sample_rate=sr,
        mixture=mix,
        true_harmonic=h / scale * 0.95,
        true_percussive=p / scale * 0.95,
        harmonic_gain=h_gain,
        percussive_gain=p_gain,
        notes=notes,
        tags=tags,
        meta=meta or {},
    )


def sustained_plus_pulse(sr: int = 22050, duration: float = 4.0, seed: int = 0) -> Scenario:
    """Sustain (A3+E4 dyad) under sparse snare/kick pulses.  Clean baseline."""
    h = (
        signals.harmonic_tone(220.0, duration, sr, amplitude=0.30, n_harmonics=14, seed=seed)
        + signals.harmonic_tone(329.63, duration, sr, amplitude=0.22, n_harmonics=10, seed=seed + 1)
    )
    pulse_times = [0.5, 1.25, 2.0, 2.75, 3.5]
    p = signals.click_train(pulse_times, duration, sr, amplitude=0.55, seed=seed + 7)
    return _finalize(
        "sustain_plus_pulse",
        sr,
        h,
        p,
        1.0,
        1.0,
        "Dyad A3/E4 with five isolated broadband clicks; sparse transients.",
        ["clean", "known-truth"],
        {"pulse_times_s": pulse_times},
    )


def dense_drums_with_sustain(sr: int = 22050, duration: float = 5.0, seed: int = 0) -> Scenario:
    """Sustained pad under a dense 16th-note drum pattern.  Key overlap case."""
    h = (
        signals.harmonic_tone(196.0, duration, sr, amplitude=0.22, n_harmonics=16, seed=seed)
        + signals.harmonic_tone(293.66, duration, sr, amplitude=0.20, n_harmonics=12, seed=seed + 1)
        + signals.harmonic_tone(392.0, duration, sr, amplitude=0.12, n_harmonics=9, seed=seed + 2)
    )
    p, times = signals.dense_drum_pattern(duration, sr, bpm=168.0, amplitude=0.55, seed=seed + 3)
    return _finalize(
        "dense_drums_with_sustain",
        sr,
        h,
        p,
        1.0,
        1.0,
        "Three-note pad under 168 BPM 16th-note drums; sustained tones overlap hits.",
        ["clean", "dense", "known-truth", "overlap"],
        {"drum_times_s": times, "bpm": 168.0},
    )


def quiet_sustain_loud_pulses(sr: int = 22050, duration: float = 4.0, seed: int = 0) -> Scenario:
    """Imbalanced levels: sustain 20 dB below transients."""
    h = signals.harmonic_tone(261.63, duration, sr, amplitude=0.05, n_harmonics=12, seed=seed)
    p = signals.click_train([0.4, 1.1, 1.8, 2.5, 3.2], duration, sr, amplitude=0.8, seed=seed + 4)
    return _finalize(
        "quiet_sustain_loud_pulses",
        sr,
        h,
        p,
        1.0,
        1.0,
        "Sine-ish sustain at -20 dB relative to loud clicks; tests level sensitivity.",
        ["clean", "imbalanced", "known-truth"],
    )


def vibrato_sustain(sr: int = 22050, duration: float = 4.0, seed: int = 0) -> Scenario:
    """Counterexample: 6 Hz vibrato smears partials in frequency."""
    h = signals.harmonic_tone(
        330.0, duration, sr, amplitude=0.4, n_harmonics=10,
        vibrato_hz=6.0, vibrato_cents=60.0, seed=seed,
    )
    p = signals.click_train([0.6, 1.6, 2.6, 3.4], duration, sr, amplitude=0.5, seed=seed + 5)
    return _finalize(
        "vibrato_sustain",
        sr,
        h,
        p,
        1.0,
        1.0,
        "Counterexample: 6 Hz / +/-60 cent vibrato violates stationary-pitch assumption.",
        ["counterexample", "vibrato", "known-truth"],
    )


def noisy_sustain(sr: int = 22050, duration: float = 4.0, seed: int = 0) -> Scenario:
    """Counterexample: breathy noise in the sustain is morphologically percussive."""
    h = signals.harmonic_tone(
        220.0, duration, sr, amplitude=0.35, n_harmonics=10, noise=0.35, seed=seed
    )
    p = signals.click_train([0.5, 1.5, 2.5, 3.5], duration, sr, amplitude=0.4, seed=seed + 6)
    return _finalize(
        "noisy_sustain",
        sr,
        h,
        p,
        1.0,
        1.0,
        "Counterexample: 35% broadband noise inside the sustain is claimed as percussive.",
        ["counterexample", "noisy-sustain", "known-truth"],
    )


def overlapping_attacks(sr: int = 22050, duration: float = 4.0, seed: int = 0) -> Scenario:
    """Counterexample: tones re-attack *during* drum hits (overlapping onsets)."""
    h = np.zeros(int(duration * sr))
    for i, (t0, f) in enumerate([(0.0, 220.0), (1.0, 261.63), (2.0, 196.0), (3.0, 293.66)]):
        seg = signals.harmonic_tone(
            f, 1.0, sr, amplitude=0.28, n_harmonics=12, seed=seed + i
        )
        j = int(t0 * sr)
        h[j : j + len(seg)] += seg[: len(h) - j]
    p = signals.click_train([1.0, 1.03, 2.0, 2.02, 3.0, 3.03], duration, sr, amplitude=0.5, seed=seed + 8)
    return _finalize(
        "overlapping_attacks",
        sr,
        h,
        p,
        1.0,
        1.0,
        "Counterexample: note onsets coincide with transients, so onsets are ambiguous.",
        ["counterexample", "overlapping-onsets", "known-truth"],
    )


def glissando_sustain(sr: int = 22050, duration: float = 4.0, seed: int = 0) -> Scenario:
    """Counterexample: continuous glide is not a stationary line."""
    h = signals.harmonic_tone(
        200.0, duration, sr, amplitude=0.35, n_harmonics=10, glide_to=400.0, seed=seed
    )
    p = signals.click_train([0.7, 1.7, 2.7, 3.6], duration, sr, amplitude=0.45, seed=seed + 9)
    return _finalize(
        "glissando_sustain",
        sr,
        h,
        p,
        1.0,
        1.0,
        "Counterexample: octave glide violates stationarity; median line tracking drifts.",
        ["counterexample", "glissando", "known-truth"],
    )


def percussive_drone(sr: int = 22050, duration: float = 4.0, seed: int = 0) -> Scenario:
    """Counterexample: amplitude-modulated drone at 8 Hz looks periodic/drum-like."""
    t = np.arange(int(duration * sr)) / sr
    tone = signals.harmonic_tone(110.0, duration, sr, amplitude=0.5, n_harmonics=8, seed=seed)
    am = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 8.0 * t))
    h = tone * (0.4 + 0.6 * am)
    p = signals.click_train([1.5], duration, sr, amplitude=0.3, seed=seed + 10)
    return _finalize(
        "percussive_drone",
        sr,
        h,
        p,
        1.0,
        1.0,
        "Counterexample: hard-gated 8 Hz drone creates periodic transients in a tonal source.",
        ["counterexample", "modulated-drone", "known-truth"],
    )


BUILDERS = {
    "sustain_plus_pulse": sustained_plus_pulse,
    "dense_drums_with_sustain": dense_drums_with_sustain,
    "quiet_sustain_loud_pulses": quiet_sustain_loud_pulses,
    "vibrato_sustain": vibrato_sustain,
    "noisy_sustain": noisy_sustain,
    "overlapping_attacks": overlapping_attacks,
    "glissando_sustain": glissando_sustain,
    "percussive_drone": percussive_drone,
}

CLEAN = ["sustain_plus_pulse", "dense_drums_with_sustain", "quiet_sustain_loud_pulses"]
COUNTEREXAMPLES = [
    "vibrato_sustain",
    "noisy_sustain",
    "overlapping_attacks",
    "glissando_sustain",
    "percussive_drone",
]


def build(name: str, sr: int = 22050, duration: float = 4.0, seed: int = 0) -> Scenario:
    if name not in BUILDERS:
        raise KeyError(f"unknown scenario {name!r}; have {sorted(BUILDERS)}")
    return BUILDERS[name](sr=sr, duration=duration, seed=seed)
