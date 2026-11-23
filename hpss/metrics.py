"""Evaluation metrics: separation, leakage, recomposition, mask fidelity.

No single number is trusted on its own.  Three independent views are reported:

1. ``bss_scores``      BSS SDR / SIR / SAR (scale-invariant).  SIR is the
                       standard interference ("cross-talk") measure.
2. ``composition``     time-domain attribution of an estimate onto the
                       orthogonalized ground-truth source directions.  Shows
                       self / other-source / unexplained energy fractions.
3. ``mask_quality``    agreement of the delivered soft masks with an oracle
                       ideal-ratio mask built from the true source spectra.

Energy is reported *separately* (``energy_report``) because BSS rescales each
estimate, hiding level errors.  Recomposition is checked twice: the mask sum
must be a partition, and the time-domain remix is compared with the input.
"""

from __future__ import annotations

import numpy as np


def _fit(length: int, *signals: np.ndarray) -> list[np.ndarray]:
    out = []
    for s in signals:
        s = np.asarray(s, dtype=np.float64).reshape(-1)
        if len(s) < length:
            s = np.pad(s, (0, length - len(s)))
        out.append(s[:length])
    return out


def _energy(x: np.ndarray) -> float:
    return float(np.dot(x, x))


def _ratio_db(num: float, den: float) -> float:
    if den <= 0:
        return float("nan")
    if num <= 0:
        return -np.inf
    return float(10.0 * np.log10(num / den))


def bss_scores(
    true_h: np.ndarray, true_p: np.ndarray, est_h: np.ndarray, est_p: np.ndarray
) -> dict:
    """SDR/SIR/SAR for the two delivered components (residual excluded).

    ``mir_eval`` scales estimates internally, so these describe shape quality
    only; pair with :func:`energy_report` for absolute levels.
    """
    import mir_eval

    n = min(len(true_h), len(true_p), len(est_h), len(est_p))
    th, tp, eh, ep = _fit(n, true_h, true_p, est_h, est_p)
    ref = np.vstack([th, tp])
    est = np.vstack([eh, ep])
    try:
        with np.errstate(invalid="ignore", divide="ignore"):
            sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
                ref, est, compute_permutation=False
            )
    except Exception as exc:  # degenerate/silent estimate
        return {"error": f"{type(exc).__name__}: {exc}"}
    out = {}
    for i, name in enumerate(("harmonic", "percussive")):
        out[name] = {
            "sdr_db": _finite(sdr[i]),
            "sir_db": _finite(sir[i]),
            "sar_db": _finite(sar[i]),
        }
    out["mean_sdr_db"] = _finite(np.mean(sdr))
    out["mean_sir_db"] = _finite(np.mean(sir))
    out["mean_sar_db"] = _finite(np.mean(sar))
    return out


def _finite(v) -> float:
    v = float(v)
    return v if np.isfinite(v) else float(np.sign(v) * np.inf)


def _orthonormal_basis(refs: list[np.ndarray]) -> list[tuple[int, np.ndarray]]:
    basis: list[tuple[int, np.ndarray]] = []
    for idx, r in enumerate(refs):
        v = r.astype(np.float64).copy()
        for _, u in basis:
            v -= np.dot(v, u) * u
        nrm = float(np.linalg.norm(v))
        if nrm > 1e-12:
            basis.append((idx, v / nrm))
    return basis


def composition(estimate: np.ndarray, refs: dict[str, np.ndarray]) -> dict:
    """Energy split of ``estimate`` across orthogonalized ground-truth directions.

    Each direction is attributed to the reference that created it.  A reference
    fully explained by earlier ones contributes ~0.  Fractions are non-negative
    and sum to 1; ``other`` is energy not proportional to any reference (e.g. a
    time-varying mask applied to a source, or a genuinely new component).

    Interpretation for stems: ``other_source`` is cross-talk, ``other`` is
    residual/nonlinear remainder.  Because a single Gabor mask of a source is
    not generally a scalar multiple of that source, ``other`` is not zero even
    for a good separation; compare it across methods rather than expecting 0.
    """
    est = np.asarray(estimate, dtype=np.float64).reshape(-1)
    names = list(refs)
    mat = np.vstack(_fit(len(est), *[refs[k] for k in names]))
    basis = _orthonormal_basis([mat[i] for i in range(mat.shape[0])])
    total = _energy(est)
    fractions = {k: 0.0 for k in names}
    if total <= 0:
        fractions["other"] = 0.0
        return fractions
    used = np.zeros_like(est)
    for idx, u in basis:
        proj = np.dot(est, u) * u
        used += proj
        fractions[names[idx]] += _energy(proj) / total
    fractions["other"] = max(0.0, 1.0 - _energy(used) / total)
    return fractions


def leakage_summary(
    true_h: np.ndarray, true_p: np.ndarray, est_h: np.ndarray, est_p: np.ndarray
) -> dict:
    """Cross-talk and unexplained remainder for each delivered stem."""
    ch = composition(est_h, {"harmonic": true_h, "percussive": true_p})
    cp = composition(est_p, {"harmonic": true_h, "percussive": true_p})
    n = min(len(true_h), len(true_p), len(est_h), len(est_p))
    th, tp, eh, ep = _fit(n, true_h, true_p, est_h, est_p)
    return {
        "harmonic": {
            "self_fraction": ch["harmonic"],
            "crosstalk_fraction": ch["percussive"],
            "unexplained_fraction": ch["other"],
            "crosstalk_vs_true_percussive_db": _ratio_db(
                ch["percussive"] * _energy(eh), _energy(tp)
            ),
        },
        "percussive": {
            "self_fraction": cp["percussive"],
            "crosstalk_fraction": cp["harmonic"],
            "unexplained_fraction": cp["other"],
            "crosstalk_vs_true_harmonic_db": _ratio_db(
                cp["harmonic"] * _energy(ep), _energy(th)
            ),
        },
    }


def energy_report(
    x: np.ndarray,
    true_h: np.ndarray,
    true_p: np.ndarray,
    est_h: np.ndarray,
    est_p: np.ndarray,
    est_r: np.ndarray,
    mask_sum: np.ndarray | None = None,
) -> dict:
    """Absolute-level and recomposition report (no per-stem normalization)."""
    n = len(x)
    x, th, tp, eh, ep, er = _fit(n, x, true_h, true_p, est_h, est_p, est_r)
    two = eh + ep
    three = eh + ep + er
    e = {
        "input": _energy(x),
        "true_h": _energy(th),
        "true_p": _energy(tp),
        "est_h": _energy(eh),
        "est_p": _energy(ep),
        "est_r": _energy(er),
    }
    out = {
        **{f"{k}_energy": v for k, v in e.items()},
        "est_h_over_true_h_db": _ratio_db(e["est_h"], e["true_h"]),
        "est_p_over_true_p_db": _ratio_db(e["est_p"], e["true_p"]),
        "harmonic_plus_percussive_over_input_db": _ratio_db(_energy(two), e["input"]),
        "delivered_over_input_db": _ratio_db(_energy(three), e["input"]),
        "recomposition_error_db": _ratio_db(_energy(three - x), e["input"]),
        "two_component_recomposition_error_db": _ratio_db(_energy(two - x), e["input"]),
        "recomposition_correlation": float(np.corrcoef(three, x)[0, 1]) if e["input"] > 0 else 1.0,
        "residual_energy_fraction": e["est_r"] / (_energy(three) or 1.0),
        "residual_over_input_db": _ratio_db(e["est_r"], e["input"]),
    }
    if mask_sum is not None:
        out["mask_sum_max_abs_error"] = float(np.max(np.abs(mask_sum - 1.0)))
        out["mask_max_overshoot"] = float(np.max(np.maximum(mask_sum - 1.0, 0.0)))
    return out


def residual_composition(
    est_r: np.ndarray, true_h: np.ndarray, true_p: np.ndarray
) -> dict:
    return composition(est_r, {"harmonic": true_h, "percussive": true_p})


def mask_quality(
    est_mask: np.ndarray,
    true_h_spec: np.ndarray,
    true_p_spec: np.ndarray,
    harmonic_src: bool,
) -> dict:
    """Soft-mask agreement with an oracle ideal-ratio mask.

    The oracle IRM is the standard power ratio between the two true sources.
    For strongly level-imbalanced mixtures (loud broadband transients over a
    quiet sustain) the IRM labels most bins by the louder source, so treat the
    absolute F1 as a sparse-transient score, not a probability of correctness.
    """
    h2 = np.abs(true_h_spec) ** 2
    p2 = np.abs(true_p_spec) ** 2
    denom = h2 + p2
    oracle_h = np.divide(h2, denom, out=np.full_like(h2, 0.5), where=denom > 1e-18)
    oracle = oracle_h if harmonic_src else 1.0 - oracle_h
    m = np.clip(est_mask, 0.0, 1.0)
    peak = float(np.max(denom)) if denom.size else 0.0
    active = denom > peak * 1e-6
    if not np.any(active):
        return {}
    m_a, o_a = m[active], oracle[active]
    corr = (
        float(np.corrcoef(m_a, o_a)[0, 1])
        if np.std(m_a) > 1e-9 and np.std(o_a) > 1e-9
        else float("nan")
    )
    pred = m_a >= 0.5
    truth = o_a >= 0.5
    tp = float(np.sum(pred & truth))
    fp = float(np.sum(pred & ~truth))
    fn = float(np.sum(~pred & truth))
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {
        "corr": corr,
        "mae": float(np.mean(np.abs(m_a - o_a))),
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def evaluate(
    scenario,
    result,
) -> dict:
    """Assemble all metrics for one scenario / decomposition pair."""
    est_h, est_p, est_r = result.harmonic, result.percussive, result.residual
    x, th, tp = scenario.mixture, scenario.true_harmonic, scenario.true_percussive
    mask_sum = result.mask_harmonic + result.mask_percussive + result.mask_residual
    spec_h = result.stft.forward(th)
    spec_p = result.stft.forward(tp)
    return {
        "name": scenario.name,
        "tags": scenario.tags,
        "notes": scenario.notes,
        "duration_s": scenario.duration,
        "diagnostic": result.diagnostic,
        "bss": bss_scores(th, tp, est_h, est_p),
        "leakage": leakage_summary(th, tp, est_h, est_p),
        "energy": energy_report(x, th, tp, est_h, est_p, est_r, mask_sum),
        "residual_composition": residual_composition(est_r, th, tp),
        "mask_quality": {
            "harmonic": mask_quality(result.mask_harmonic, spec_h, spec_p, True),
            "percussive": mask_quality(result.mask_percussive, spec_h, spec_p, False),
        },
        "policy": result.config.shared_policy,
        "reference_is_estimate_only": True,
    }
