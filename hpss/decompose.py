from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from .config import HPSSConfig, STFTConfig
from .stft import STFT


@dataclass
class HPSSResult:
    sample_rate: int
    stft: STFT
    spectrum: np.ndarray          # complex STFT of the mixture (freq, frames)
    magnitude: np.ndarray         # |spectrum|
    harmonic_support: np.ndarray  # time-smoothed power
    percussive_support: np.ndarray  # freq-smoothed power
    ambiguity: np.ndarray         # in [0,1], high where both supports are strong
    mask_harmonic: np.ndarray
    mask_percussive: np.ndarray
    mask_residual: np.ndarray
    harmonic: np.ndarray          # time-domain component
    percussive: np.ndarray
    residual: np.ndarray
    input_length: int
    config: HPSSConfig
    diagnostic: dict

    @property
    def frequencies(self) -> np.ndarray:
        return self.stft.frequencies(self.sample_rate)

    @property
    def times(self) -> np.ndarray:
        return self.stft.frame_times(self.spectrum.shape[1], self.sample_rate)

    @property
    def residual_energy_fraction(self) -> float:
        p = self._energy(self.residual)
        total = self._energy(self.harmonic) + self._energy(self.percussive) + p
        return float(p / total) if total > 0 else 0.0

    @staticmethod
    def _energy(x: np.ndarray) -> float:
        return float(np.sum(np.asarray(x, dtype=np.float64) ** 2))


def _unit_power(x: np.ndarray) -> np.ndarray:
    peak = np.max(x) if x.size else 0.0
    if peak <= 0:
        return np.zeros_like(x)
    return x / peak


def median_supports(
    magnitude: np.ndarray, kernel_time: int, kernel_freq: int, use_power: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth the spectrogram along each axis.

    A sustained tone is smooth in time -> large time-median.
    A drum hit is broadband -> large frequency-median.
    """
    src = magnitude**2 if use_power else magnitude
    kt = min(kernel_time, src.shape[1]) if src.shape[1] else max(1, kernel_time)
    kf = min(kernel_freq, src.shape[0]) if src.shape[0] else max(1, kernel_freq)
    harmonic = median_filter(src, size=(1, max(1, kt)), mode="reflect")
    percussive = median_filter(src, size=(max(1, kf), 1), mode="reflect")
    return harmonic, percussive


def soft_share(harmonic: np.ndarray, percussive: np.ndarray, power: float) -> np.ndarray:
    h = np.maximum(harmonic, 0.0) ** (power / 2.0)
    p = np.maximum(percussive, 0.0) ** (power / 2.0)
    denom = h + p
    share = np.zeros_like(denom)
    np.divide(h, denom, out=share, where=denom > 0)
    # no support at all -> split evenly rather than dropping the bin
    share = np.where(denom > 0, share, 0.5)
    return share


def ambiguous_fraction(
    harmonic: np.ndarray, percussive: np.ndarray, magnitude: np.ndarray, floor_db: float
) -> np.ndarray:
    """Energy fraction of each bin that *both* structures can claim.

    ``min(H, P) / |X|^2`` is the fraction of the observed bin power that the
    weaker directional median also retains, i.e. how much of this bin both a
    time-persistent and a frequency-broadband structure can explain.  It is 0
    when only one structure explains the bin and approaches 1 when both explain
    all of it.  Bins more than ``floor_db`` below the peak power are treated as
    silence (numerically unstable ratio, negligible energy) and score 0.
    """
    if magnitude.size == 0:
        return np.zeros_like(magnitude)
    power = magnitude**2
    both = np.minimum(harmonic, percussive)
    frac = np.divide(both, power, out=np.zeros_like(power), where=power > 0)
    frac = np.clip(frac, 0.0, 1.0)
    ref = float(np.max(power)) if power.size else 0.0
    quiet = power < ref * 10.0 ** (floor_db / 10.0)
    frac[quiet] = 0.0
    return frac


def build_masks(
    share: np.ndarray,
    ambiguity: np.ndarray,
    mask_kind: str,
    policy: str,
    duplicate_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition or share the mixture according to an explicit rule.

    Returns ``(m_h, m_p, m_r)``.  For every policy except ``shared`` the three
    masks sum to one at every bin, so the delivered stems remix to the input
    exactly.  ``shared`` deliberately assigns the ambiguous fraction to *both*
    components, so the masks sum to ``1 + 2*duplicate_weight*ambiguity``: the
    remix overshoots and the overshoot is reported, not hidden.
    """
    if mask_kind == "soft":
        base_h = share
    else:
        base_h = (share >= 0.5).astype(np.float64)
    base_p = 1.0 - base_h
    a = np.clip(ambiguity, 0.0, 1.0)

    if policy == "complementary":
        m_h, m_p, m_r = base_h, base_p, np.zeros_like(a)
    elif policy == "shared":
        # Duplicate only the fraction the base split assigns to each side, so
        # the overshoot is bounded by 2*min(base_h, base_p) and the mask sum is
        # 1 + 2*min(base_h, base_p) <= 2.
        w = duplicate_weight * np.minimum(base_h, base_p)
        m_h = base_h + w
        m_p = base_p + w
        m_r = np.zeros_like(a)
    elif policy == "residual":
        m_h = (1.0 - a) * base_h
        m_p = (1.0 - a) * base_p
        m_r = a
    elif policy == "split":
        m_h = (1.0 - a) * base_h + 0.5 * a
        m_p = (1.0 - a) * base_p + 0.5 * a
        m_r = np.zeros_like(a)
    elif policy == "harmonic":
        m_h = (1.0 - a) * base_h + a
        m_p = (1.0 - a) * base_p
        m_r = np.zeros_like(a)
    elif policy == "percussive":
        m_h = (1.0 - a) * base_h
        m_p = (1.0 - a) * base_p + a
        m_r = np.zeros_like(a)
    else:  # validated by HPSSConfig, defensive only
        raise ValueError(f"unknown shared_policy {policy!r}")
    return m_h, m_p, m_r


def _apply_mask(stft: STFT, spectrum: np.ndarray, mask: np.ndarray, length: int) -> np.ndarray:
    return stft.inverse(mask * spectrum, length)


def decompose(signal: np.ndarray, sample_rate: int, config: HPSSConfig | None = None) -> HPSSResult:
    config = config or HPSSConfig()
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    length = len(x)
    stft = STFT(config.stft)
    spectrum = stft.forward(x)
    magnitude = np.abs(spectrum)
    harmonic_sup, percussive_sup = median_supports(
        magnitude, config.kernel_time, config.kernel_freq, use_power=True
    )
    share = soft_share(harmonic_sup, percussive_sup, config.mask_power)
    ambiguity = ambiguous_fraction(
        harmonic_sup, percussive_sup, magnitude, config.ambiguity_floor_db
    )
    m_h, m_p, m_r = build_masks(
        share, ambiguity, config.mask_kind, config.shared_policy, config.duplicate_weight
    )

    harmonic = _apply_mask(stft, spectrum, m_h, length)
    percussive = _apply_mask(stft, spectrum, m_p, length)
    residual = _apply_mask(stft, spectrum, m_r, length)

    # Exact-by-construction checks (magnitude masks keep the complex phase, so
    # the masked spectra add back to the input spectrum).
    two_sum = harmonic + percussive
    three_sum = harmonic + percussive + residual
    x_energy = float(np.sum(x**2))
    flux_db = _db(np.sum((three_sum - x) ** 2), x_energy)
    two_flux_db = _db(np.sum((two_sum - x) ** 2), x_energy)
    mask_sum = m_h + m_p + m_r
    if mask_sum.size:
        mask_sum_err = float(np.max(np.abs(mask_sum - 1.0)))
        mask_over = float(np.max(np.maximum(mask_sum - 1.0, 0.0)))
    else:
        mask_sum_err = 0.0
        mask_over = 0.0

    # Fraction of total *energy* landing in each ambiguous allocation.
    power = magnitude**2
    total_power = float(np.sum(power)) or 1.0
    amb_frac = float(np.sum(ambiguity * power) / total_power)

    diagnostic = {
        "config": config.to_dict(),
        "sample_rate": sample_rate,
        "duration_s": length / sample_rate,
        "n_frames": int(spectrum.shape[1]),
        "n_bins": int(spectrum.shape[0]),
        "mask_sum_max_error": mask_sum_err,
        "mask_max_overshoot": mask_over,
        "shared_energy_duplicated": float(
            np.sum(2.0 * config.duplicate_weight * np.minimum(share, 1.0 - share) * ambiguity * power)
            / total_power
        ),
        "reconstruction_error_db": flux_db,
        "two_component_error_db": two_flux_db,
        "input_energy": x_energy,
        "harmonic_energy": float(np.sum(harmonic**2)),
        "percussive_energy": float(np.sum(percussive**2)),
        "residual_energy": float(np.sum(residual**2)),
        "remix_energy_ratio": float(
            np.sum(three_sum**2) / x_energy if x_energy > 0 else 0.0
        ),
        "ambiguous_energy_fraction": amb_frac,
        "ambiguous_bin_fraction": float(np.mean(ambiguity > 0.5)),
        "residual_energy_fraction": float(np.sum(residual**2) / x_energy) if x_energy else 0.0,
        "frac_ambiguous_to_residual": float(
            np.sum(ambiguity * (m_r > 0) * power) / (np.sum(ambiguity * power) or 1.0)
        ),
    }
    return HPSSResult(
        sample_rate=sample_rate,
        stft=stft,
        spectrum=spectrum,
        magnitude=magnitude,
        harmonic_support=harmonic_sup,
        percussive_support=percussive_sup,
        ambiguity=ambiguity,
        mask_harmonic=m_h,
        mask_percussive=m_p,
        mask_residual=m_r,
        harmonic=harmonic,
        percussive=percussive,
        residual=residual,
        input_length=length,
        config=config,
        diagnostic=diagnostic,
    )


def _db(num: float, ref: float) -> float:
    if ref <= 0:
        return float("nan")
    if num <= 0:
        return -np.inf
    return float(10.0 * np.log10(num / ref))
