"""Sample-pack material: known-source mixtures for harmonic/percussive work.

Every entry is built by **forward synthesis** from explicit stems,
``mixture = scale * (g_h * H + g_p * P)``, with the coefficients stored in the
manifest.  The alternative construction -- deriving one stem by **phase
cancellation** (``mixture - scaled other stem``) -- is computed as well and
its null depth against the true stem is reported.  In exact arithmetic the
two paths are identical; after the float32 WAV round-trip the null depth
quantifies how indistinguishable they remain.  The pack ships the
forward-synthesized stems (exact ground truth) plus the null-derived
percussive as the comparison reference, so both constructions are auditable.

Entries with an ``assignable`` layer (the brush sweep) support **segment
edits**: the editor reassigns time segments of that layer between the
harmonic and percussive stems.  :func:`verify_edit` rebuilds sources, mixture
and reference masks from an edit and measures whether they still agree
(remix error, phase-cancel null depth, mask partition), which is what makes
an attribution change checkable instead of cosmetic.

Estimated stems shipped in the pack are always produced by
:func:`hpss.decompose.decompose` on the mixture -- never by copying the true
sources -- so the pack honestly shows the decomposer's limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import signals
from .config import HPSSConfig, STFTConfig
from .stft import STFT

OWNERS = ("harmonic", "percussive")


@dataclass(frozen=True)
class Segment:
    """One editable time region of the assignable layer."""

    index: int
    start_s: float
    end_s: float
    owner: str = "percussive"

    def __post_init__(self) -> None:
        if self.owner not in OWNERS:
            raise ValueError(f"segment owner must be one of {OWNERS}")
        if not self.end_s > self.start_s >= 0.0:
            raise ValueError("segment must satisfy 0 <= start < end")

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "owner": self.owner,
        }


@dataclass
class PackEntry:
    """A known-source mixture built from named layers."""

    name: str
    sample_rate: int
    seed: int
    layers: dict[str, np.ndarray]
    base_owner: dict[str, str]
    assignable: str | None
    segments: list[Segment]
    harmonic_gain: float
    percussive_gain: float
    peak_target: float
    params: dict
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return max(len(x) for x in self.layers.values())

    @property
    def duration(self) -> float:
        return self.n_samples / self.sample_rate


def routing_mask(
    segments: list[Segment],
    n: int,
    sr: int,
    default_owner: str,
    fade_s: float = 0.01,
) -> np.ndarray:
    """Per-sample harmonic ownership of the assignable layer, in [0, 1].

    Piecewise-constant per segment with short raised-cosine crossfades at
    ownership changes so re-routed stems have no clicks.
    """
    m = np.full(n, 1.0 if default_owner == "harmonic" else 0.0, dtype=np.float64)
    for seg in segments:
        a = max(0, min(n, int(round(seg.start_s * sr))))
        b = max(0, min(n, int(round(seg.end_s * sr))))
        m[a:b] = 1.0 if seg.owner == "harmonic" else 0.0
    f = int(round(fade_s * sr))
    if f > 0 and n > 1:
        edges = np.flatnonzero(np.diff(m) != 0.0)
        for i in edges:
            a = max(0, i + 1 - f)
            b = min(n, i + 1 + f)
            v0, v1 = m[i], m[i + 1]
            w = 0.5 + 0.5 * np.cos(np.linspace(0.0, np.pi, b - a))
            m[a:b] = v0 * w + v1 * (1.0 - w)
    return m


def compose(
    entry: PackEntry, segments: list[Segment] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Forward-synthesize ``(true_harmonic, true_percussive, mixture, scale)``.

    Layers are summed into stems by ownership, scaled by the per-stem gains,
    then the sum is peak-normalized to ``peak_target``; the same scale is
    applied to both stems so ``true_h + true_p == mixture`` exactly.
    """
    segs = entry.segments if segments is None else list(segments)
    n = entry.n_samples
    h = np.zeros(n, dtype=np.float64)
    p = np.zeros(n, dtype=np.float64)
    for name, layer in entry.layers.items():
        if name == entry.assignable:
            continue
        dst = h if entry.base_owner[name] == "harmonic" else p
        dst[: len(layer)] += layer
    if entry.assignable is not None:
        layer = entry.layers[entry.assignable]
        mask_h = routing_mask(
            segs, n, entry.sample_rate, entry.base_owner[entry.assignable]
        )
        h[: len(layer)] += layer * mask_h[: len(layer)]
        p[: len(layer)] += layer * (1.0 - mask_h[: len(layer)])
    h *= entry.harmonic_gain
    p *= entry.percussive_gain
    raw = h + p
    peak = float(np.max(np.abs(raw))) if n else 0.0
    scale = entry.peak_target / peak if peak > 0 else 1.0
    return h * scale, p * scale, raw * scale, scale


def null_depth_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    """10*log10(||estimate - reference||^2 / ||reference||^2); -inf means exact."""
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    den = float(np.dot(ref, ref))
    num = float(np.dot(est - ref, est - ref))
    if den <= 0:
        return float("nan")
    if num <= 0:
        return -np.inf
    return float(10.0 * np.log10(num / den))


def phase_cancel_percussive(mixture: np.ndarray, true_harmonic: np.ndarray) -> np.ndarray:
    """The other construction path: percussive stem as ``mixture - harmonic``."""
    return np.asarray(mixture, dtype=np.float64) - np.asarray(true_harmonic, dtype=np.float64)


def reference_masks(
    true_h: np.ndarray, true_p: np.ndarray, stft_cfg: STFTConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Oracle ideal-ratio masks from the true stems (float32, freq x frames)."""
    stft = STFT(stft_cfg or STFTConfig())
    h2 = np.abs(stft.forward(true_h)) ** 2
    p2 = np.abs(stft.forward(true_p)) ** 2
    denom = h2 + p2
    m_h = np.divide(h2, denom, out=np.full_like(h2, 0.5), where=denom > 1e-18)
    m_p = 1.0 - m_h
    return m_h.astype(np.float32), m_p.astype(np.float32)


def edited_segments(entry: PackEntry, owners: list[str]) -> list[Segment]:
    """Return the entry's segments with owners replaced by ``owners``."""
    if len(owners) != len(entry.segments):
        raise ValueError(
            f"expected {len(entry.segments)} owners, got {len(owners)}"
        )
    out = []
    for seg, owner in zip(entry.segments, owners):
        if owner not in OWNERS:
            raise ValueError(f"unknown owner {owner!r}")
        out.append(replace(seg, owner=owner))
    return out


def verify_edit(
    entry: PackEntry, owners: list[str], stft_cfg: STFTConfig | None = None
) -> dict:
    """Rebuild sources/mixture/reference masks after a segment edit and check
    that they still agree with each other and with the shipped mixture.

    Returns the measured evidence; ``ok`` is True only when the rebuilt
    mixture matches the original (remix error), the phase-cancelled harmonic
    reproduces the rebuilt percussive stem (null depth), and the reference
    masks form an exact partition.
    """
    segs = edited_segments(entry, owners)
    eh, ep, emix, scale = compose(entry, segs)
    oh, op, omix, _ = compose(entry, entry.segments)
    remix_err = null_depth_db(omix, eh + ep)
    null_err = null_depth_db(ep, phase_cancel_percussive(omix, eh))
    m_h, m_p = reference_masks(eh, ep, stft_cfg)
    mask_sum_err = float(np.max(np.abs(m_h.astype(np.float64) + m_p.astype(np.float64) - 1.0)))
    om_h, _ = reference_masks(oh, op, stft_cfg)
    changed = [
        s.index for s, e in zip(entry.segments, segs) if s.owner != e.owner
    ]
    sr = entry.sample_rate
    hop = (stft_cfg or STFTConfig()).hop
    delta = np.abs(m_h.astype(np.float64) - om_h.astype(np.float64))
    per_segment = []
    for seg in segs:
        a = int(seg.start_s * sr / hop)
        b = max(a + 1, int(seg.end_s * sr / hop))
        per_segment.append(
            {
                "index": seg.index,
                "owner": seg.owner,
                "mask_change": float(np.mean(delta[:, a:b])),
            }
        )
    ok = bool(
        remix_err < -120.0 and null_err < -120.0 and mask_sum_err < 1e-6
    )
    return {
        "entry": entry.name,
        "owners": [s.owner for s in segs],
        "changed_segments": changed,
        "remix_error_db": remix_err,
        "phase_cancel_null_db": null_err,
        "mask_sum_max_abs_error": mask_sum_err,
        "scale": scale,
        "per_segment_mask_change": per_segment,
        "ok": ok,
    }


def strength_to_config(strength: float) -> HPSSConfig:
    """Map the UI separation-strength slider in [0, 1] to a kernel config.

    Higher strength sharpens the masks (``mask_power``) and lengthens the
    median supports, i.e. a more decisive, more smoothed split.
    """
    s = min(max(float(strength), 0.0), 1.0)
    kernel = 15 + 2 * round(20 * s)  # odd-ish, 15 .. 55
    return HPSSConfig(kernel_time=kernel, kernel_freq=kernel, mask_power=1.0 + 5.0 * s)


# --------------------------------------------------------------------------
# Entry builders
# --------------------------------------------------------------------------


def sustained_chords_sparse_drums(sr: int = 22050, seed: int = 0) -> PackEntry:
    """Sustained triad progression under sparse kick/snare/hat hits."""
    duration = 5.0
    n = int(round(duration * sr))
    chords = signals.align_to(np.zeros(1), n)
    chord_plan = [
        (0.0, 1.7, [130.81, 164.81, 196.00]),   # C
        (1.7, 1.7, [174.61, 220.00, 261.63]),   # F
        (3.4, 1.6, [196.00, 246.94, 293.66]),   # G
    ]
    for ci, (t0, dur, freqs) in enumerate(chord_plan):
        seg = np.zeros(int(round(dur * sr)))
        for ni, f in enumerate(freqs):
            seg += signals.harmonic_tone(
                f, dur, sr, amplitude=0.30, n_harmonics=12, seed=seed + 10 * ci + ni
            )
        j = int(round(t0 * sr))
        chords[j : j + len(seg)] += seg[: len(chords) - j]
    drums = signals.place_hits(
        [
            (0.45, "kick", 0.60),
            (1.75, "snare", 0.50),
            (2.60, "hat", 0.35),
            (3.30, "kick", 0.55),
            (4.20, "snare", 0.45),
        ],
        duration,
        sr,
        seed=seed + 100,
    )
    return PackEntry(
        name="sustained_chords_sparse_drums",
        sample_rate=sr,
        seed=seed,
        layers={"chords": chords, "drums": drums},
        base_owner={"chords": "harmonic", "drums": "percussive"},
        assignable=None,
        segments=[],
        harmonic_gain=1.0,
        percussive_gain=0.9,
        peak_target=0.95,
        params={
            "duration_s": duration,
            "chord_plan": [
                {"start_s": t0, "duration_s": d, "freqs_hz": f} for t0, d, f in chord_plan
            ],
            "hits": [
                {"time_s": t, "kind": k, "amplitude": a}
                for t, k, a in [
                    (0.45, "kick", 0.60),
                    (1.75, "snare", 0.50),
                    (2.60, "hat", 0.35),
                    (3.30, "kick", 0.55),
                    (4.20, "snare", 0.45),
                ]
            ],
        },
        notes="Sustained C/F/G triads under five sparse drum hits; clean morphology case.",
        tags=["known-truth", "sustained-chords", "sparse-drums"],
    )


def brush_sweep_over_sustain(sr: int = 22050, seed: int = 0) -> PackEntry:
    """Drum-brush sweeps over a sustained dyad; sweep attribution is editable.

    Gains are 1.0 so re-routing sweep segments between stems never changes
    the mixture -- only the attribution, and with it the reference masks.
    """
    duration = 5.5
    n = int(round(duration * sr))
    sustain = signals.harmonic_tone(
        220.0, duration, sr, amplitude=0.28, n_harmonics=12, seed=seed
    ) + signals.harmonic_tone(
        277.18, duration, sr, amplitude=0.20, n_harmonics=10, seed=seed + 1
    )
    sweep = np.zeros(n)
    s1 = signals.brush_sweep(
        1.6, sr, amplitude=0.50, low_hz=1800.0, high_hz=6500.0, sweep_up=True, seed=seed + 2
    )
    s2 = signals.brush_sweep(
        1.4, sr, amplitude=0.45, low_hz=1600.0, high_hz=5500.0, sweep_up=False, seed=seed + 3
    )
    j = int(round(0.8 * sr))
    sweep[j : j + len(s1)] += s1
    j = int(round(3.2 * sr))
    sweep[j : j + len(s2)] += s2
    hits = signals.place_hits(
        [(0.40, "kick", 0.45), (5.00, "snare", 0.40)], duration, sr, seed=seed + 4
    )
    segments = [
        Segment(index=i, start_s=0.75 + 0.5 * i, end_s=0.75 + 0.5 * (i + 1))
        for i in range(8)
    ]
    return PackEntry(
        name="brush_sweep_over_sustain",
        sample_rate=sr,
        seed=seed,
        layers={"sustain": sustain, "sweep": sweep, "hits": hits},
        base_owner={"sustain": "harmonic", "sweep": "percussive", "hits": "percussive"},
        assignable="sweep",
        segments=segments,
        harmonic_gain=1.0,
        percussive_gain=1.0,
        peak_target=0.95,
        params={
            "duration_s": duration,
            "sustain_freqs_hz": [220.0, 277.18],
            "sweeps": [
                {"start_s": 0.8, "duration_s": 1.6, "low_hz": 1800.0, "high_hz": 6500.0, "sweep_up": True},
                {"start_s": 3.2, "duration_s": 1.4, "low_hz": 1600.0, "high_hz": 5500.0, "sweep_up": False},
            ],
            "hits": [
                {"time_s": 0.40, "kind": "kick", "amplitude": 0.45},
                {"time_s": 5.00, "kind": "snare", "amplitude": 0.40},
            ],
            "segment_grid": {"start_s": 0.75, "step_s": 0.5, "count": 8},
        },
        notes="Two brush sweeps over an A3/C#4 dyad. The sweep layer is morphologically "
        "ambiguous (sustained yet noisy); its 8 segments can be re-owned per stem.",
        tags=["known-truth", "brush-sweep", "editable-attribution", "ambiguous"],
    )


BUILDERS = {
    "sustained_chords_sparse_drums": sustained_chords_sparse_drums,
    "brush_sweep_over_sustain": brush_sweep_over_sustain,
}

ENTRY_NAMES = list(BUILDERS)


def build_entry(name: str, sr: int = 22050, seed: int = 0) -> PackEntry:
    if name not in BUILDERS:
        raise KeyError(f"unknown pack entry {name!r}; have {sorted(BUILDERS)}")
    return BUILDERS[name](sr=sr, seed=seed)
