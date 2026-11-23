#!/usr/bin/env python3
"""Run the full HPSS study and write audio stems, mask figures, and reports.

Usage:
    PYTHONPATH=vendor/python python3 scripts/run_study.py
    PYTHONPATH=vendor/python python3 scripts/run_study.py --scenarios sustain_plus_pulse
    PYTHONPATH=vendor/python python3 scripts/run_study.py --input my.wav --sr 44100

Everything is written under ``results/``.  No output stem is rescaled: levels
are preserved so the remix can be compared with the input.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "python"))

from hpss import HPSSConfig, STFTConfig, decompose, metrics, scenarios, signals, viz  # noqa: E402
from hpss import io as hpss_io  # noqa: E402

RESULTS = ROOT / "results"
AUDIO = RESULTS / "audio"
FIGURES = RESULTS / "figures"
METRICS = RESULTS / "metrics"


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return str(o)
    raise TypeError(f"not serializable: {type(o)}")


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default) + "\n")


def _clean_float(v):
    if isinstance(v, float) and not np.isfinite(v):
        return v
    return v


def write_stems(prefix: str, scenario, result) -> dict:
    paths = {}
    sr = scenario.sample_rate
    stems = {
        "mixture": scenario.mixture,
        "true_harmonic": scenario.true_harmonic,
        "true_percussive": scenario.true_percussive,
        "est_harmonic": result.harmonic,
        "est_percussive": result.percussive,
        "est_residual": result.residual,
        "remix": result.harmonic + result.percussive + result.residual,
        "recomposition_diff": result.harmonic + result.percussive + result.residual - scenario.mixture,
        "harmonic_plus_percussive": result.harmonic + result.percussive,
    }
    for name, x in stems.items():
        p = hpss_io.write_wav(AUDIO / f"{prefix}__{name}.wav", x, sr)
        paths[name] = str(p.relative_to(RESULTS))
    return paths


def run_scenario(name: str, sr: int, duration: float, cfg: HPSSConfig, seed: int = 0) -> dict:
    sc = scenarios.build(name, sr=sr, duration=duration, seed=seed)
    result = decompose(sc.mixture, sc.sample_rate, cfg)
    ev = metrics.evaluate(sc, result)
    ev["audio"] = write_stems(name, sc, result)
    viz.plot_decomposition(result, FIGURES / f"{name}__decomposition.png",
                           title=f"{name}  [{cfg.shared_policy}]")
    viz.plot_masks_only(result, FIGURES / f"{name}__masks.png",
                        title=f"{name} masks ({cfg.shared_policy})")
    viz.plot_scenario_waveforms(sc, result, FIGURES / f"{name}__waveforms.png")
    ev["true_component_spectra"] = {
        "harmonic_energy_fraction_of_input": float(
            np.sum(sc.true_harmonic**2) / np.sum(sc.mixture**2)
        ),
        "percussive_energy_fraction_of_input": float(
            np.sum(sc.true_percussive**2) / np.sum(sc.mixture**2)
        ),
    }
    return ev


def run_policy_comparison(name: str, sr: int, duration: float) -> dict:
    out = {}
    for policy in ("complementary", "residual", "shared"):
        cfg = HPSSConfig(shared_policy=policy)
        sc = scenarios.build(name, sr=sr, duration=duration)
        result = decompose(sc.mixture, sc.sample_rate, cfg)
        ev = metrics.evaluate(sc, result)
        out[policy] = {
            "ambiguous_energy_fraction": ev["diagnostic"]["ambiguous_energy_fraction"],
            "residual_energy_fraction": ev["diagnostic"]["residual_energy_fraction"],
            "shared_energy_duplicated": ev["diagnostic"]["shared_energy_duplicated"],
            "mask_sum_max_abs_error": ev["energy"]["mask_sum_max_abs_error"],
            "delivered_over_input_db": ev["energy"]["delivered_over_input_db"],
            "recomposition_error_db": ev["energy"]["recomposition_error_db"],
            "mean_sdr_db": ev["bss"].get("mean_sdr_db"),
            "mean_sir_db": ev["bss"].get("mean_sir_db"),
        }
        viz.plot_masks_only(
            result, FIGURES / f"policy__{name}__{policy}.png",
            title=f"{name} | policy={policy} | amb={ev['diagnostic']['ambiguous_energy_fraction']:.3f}",
        )
    return out


def run_window_scales(name: str, sr: int, duration: float) -> dict:
    out = {}
    for n_fft, kernel in ((512, 15), (1024, 21), (2048, 31), (4096, 51)):
        cfg = HPSSConfig(stft=STFTConfig(n_fft=n_fft, hop_length=n_fft // 4),
                         kernel_time=kernel, kernel_freq=kernel)
        sc = scenarios.build(name, sr=sr, duration=duration)
        result = decompose(sc.mixture, sc.sample_rate, cfg)
        ev = metrics.evaluate(sc, result)
        out[str(n_fft)] = {
            "mean_sdr_db": ev["bss"].get("mean_sdr_db"),
            "mean_sir_db": ev["bss"].get("mean_sir_db"),
            "recomposition_error_db": ev["energy"]["recomposition_error_db"],
            "ambiguous_energy_fraction": ev["diagnostic"]["ambiguous_energy_fraction"],
            "mask_quality_harmonic_f1": ev["mask_quality"]["harmonic"].get("f1"),
            "mask_quality_percussive_f1": ev["mask_quality"]["percussive"].get("f1"),
        }
        viz.plot_masks_only(result, FIGURES / f"scale__{name}__nfft{n_fft}.png",
                            title=f"{name} n_fft={n_fft} kernel={kernel}")
    return out


def run_homemade_mix(sr: int, duration: float) -> dict:
    """A hand-assembled mix: sustained chord + spoken-click rhythm + noise bed."""
    rng = np.random.default_rng(42)
    n = int(duration * sr)
    t = np.arange(n) / sr
    # sustained dyad with slow drift
    h = (
        signals.harmonic_tone(174.61, duration, sr, amplitude=0.20, n_harmonics=16, seed=11)
        + signals.harmonic_tone(261.63, duration, sr, amplitude=0.15, n_harmonics=12, seed=12)
        + signals.harmonic_tone(349.23, duration, sr, amplitude=0.10, n_harmonics=9, seed=13)
    )
    # irregular "hand-played" transients, including a burst of overlapping hits
    times = [0.3, 0.72, 1.05, 1.4, 1.42, 1.45, 1.85, 2.3, 2.75, 2.78, 3.2, 3.6, 3.95]
    p = signals.click_train(times, duration, sr, amplitude=0.55, seed=21)
    p += signals.percussive_hit(duration, sr, "kick", amplitude=0.4, seed=22)
    # a low-level noise bed that is neither tonal nor transient
    bed = 0.01 * rng.standard_normal(n) * np.hanning(n)
    scale = float(np.max(np.abs(h + p))) or 1.0
    h, p, bed = h / scale * 0.8, p / scale * 0.8, bed / scale
    mix = signals.peak_normalize(h + p + bed, 0.95)
    from hpss.scenarios import Scenario

    sc = Scenario(
        name="homemade_mix",
        sample_rate=sr,
        mixture=mix,
        true_harmonic=h,
        true_percussive=p,
        notes="Hand-built: 3-note pad + irregular/overlapping clicks + noise bed. "
              "The bed has no ground-truth class; it is expected to split across residual.",
        tags=["homemade", "known-truth-partial", "overlap", "noise-bed"],
        meta={"hit_times_s": times},
    )
    result = decompose(sc.mixture, sr)
    ev = metrics.evaluate(sc, result)
    ev["audio"] = write_stems("homemade_mix", sc, result)
    ev["audio"]["noise_bed"] = str(
        hpss_io.write_wav(AUDIO / "homemade_mix__noise_bed.wav", bed, sr).relative_to(RESULTS)
    )
    viz.plot_decomposition(result, FIGURES / "homemade_mix__decomposition.png",
                           title="homemade_mix (residual policy)")
    viz.plot_scenario_waveforms(sc, result, FIGURES / "homemade_mix__waveforms.png")
    ev["bed_composition"] = metrics.composition(bed, {"harmonic": h, "percussive": p})
    ev["residual_composition_vs_bed"] = metrics.composition(result.residual, {"bed": bed})
    return ev


def run_external(path: Path, sr_override: int | None) -> dict | None:
    """Decompose a user-supplied recording.  No ground truth -> metrics limited."""
    if not path.exists():
        return None
    x, sr = hpss_io.read_wav(path)
    if sr_override:
        if sr_override != sr:
            import scipy.signal as ss

            g = np.gcd(int(sr_override), int(sr))
            x = ss.resample_poly(x, sr_override // g, sr // g)
            sr = sr_override
    result = decompose(x, sr)
    stem = path.stem
    audio = write_stems(f"external_{stem}", _ExternalScenario(x, sr), result)
    viz.plot_decomposition(result, FIGURES / f"external_{stem}__decomposition.png",
                           title=f"{path.name} (no ground truth)")
    viz.plot_masks_only(result, FIGURES / f"external_{stem}__masks.png",
                        title=f"{path.name} masks")
    en = metrics.energy_report(x, np.zeros_like(x), np.zeros_like(x),
                               result.harmonic, result.percussive, result.residual,
                               result.mask_harmonic + result.mask_percussive + result.mask_residual)
    return {
        "name": f"external_{stem}",
        "path": str(path),
        "sample_rate": sr,
        "duration_s": len(x) / sr,
        "diagnostic": result.diagnostic,
        "energy": {k: v for k, v in en.items() if not k.startswith("true_")
                   and "over_true" not in k},
        "audio": audio,
        "notes": "External recording: no ground truth, so SDR/SIR/leakage vs truth are unavailable. "
                 "Recomposition and mask/ambiguity summaries still apply.",
        "tags": ["external", "no-ground-truth"],
        "policy": result.config.shared_policy,
    }


class _ExternalScenario:
    def __init__(self, mixture, sr):
        self.mixture = mixture
        self.sample_rate = sr
        self.true_harmonic = np.zeros_like(mixture)
        self.true_percussive = np.zeros_like(mixture)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name", "policy", "duration_s", "mean_sdr_db", "mean_sir_db",
        "h_sir_db", "p_sir_db", "h_crosstalk", "p_crosstalk",
        "recomposition_error_db", "delivered_over_input_db",
        "residual_energy_fraction", "ambiguous_energy_fraction",
        "mask_h_f1", "mask_p_f1",
    ]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            b = r.get("bss", {})
            lk = r.get("leakage", {})
            en = r.get("energy", {})
            mq = r.get("mask_quality", {})
            w.writerow({
                "name": r["name"],
                "policy": r.get("policy", ""),
                "duration_s": f"{r.get('duration_s', float('nan')):.2f}",
                "mean_sdr_db": _fmt(b.get("mean_sdr_db")),
                "mean_sir_db": _fmt(b.get("mean_sir_db")),
                "h_sir_db": _fmt(b.get("harmonic", {}).get("sir_db")),
                "p_sir_db": _fmt(b.get("percussive", {}).get("sir_db")),
                "h_crosstalk": _fmt(lk.get("harmonic", {}).get("crosstalk_fraction")),
                "p_crosstalk": _fmt(lk.get("percussive", {}).get("crosstalk_fraction")),
                "recomposition_error_db": _fmt(en.get("recomposition_error_db")),
                "delivered_over_input_db": _fmt(en.get("delivered_over_input_db")),
                "residual_energy_fraction": _fmt(en.get("residual_energy_fraction")),
                "ambiguous_energy_fraction": _fmt(r.get("diagnostic", {}).get("ambiguous_energy_fraction")),
                "mask_h_f1": _fmt(mq.get("harmonic", {}).get("f1")),
                "mask_p_f1": _fmt(mq.get("percussive", {}).get("f1")),
            })


def _fmt(v):
    if v is None:
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return "inf" if f > 0 else "-inf"
    return f"{f:.3f}"


def write_html(report: dict, path: Path) -> None:
    def esc(s):
        return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    rows = []
    for r in report["scenarios"]:
        b = r.get("bss", {})
        en = r["energy"]
        lk = r.get("leakage", {})
        mq = r.get("mask_quality", {})
        d = r.get("diagnostic", {})
        audio = r.get("audio", {})
        links = " ".join(
            f'<a href="{esc(v)}">{esc(k)}</a>'
            for k, v in audio.items()
            if k in ("mixture", "true_harmonic", "true_percussive", "est_harmonic",
                     "est_percussive", "est_residual", "remix", "recomposition_diff",
                     "harmonic_plus_percussive", "noise_bed")
        )
        figs = f'<a href="figures/{esc(r["name"])}__decomposition.png">masks</a> ' \
               f'<a href="figures/{esc(r["name"])}__waveforms.png">waveforms</a>'
        rows.append(
            "<tr>"
            f"<td>{esc(r['name'])}<br><small>{esc(r.get('notes',''))}</small></td>"
            f"<td>{esc(', '.join(r.get('tags', [])))}</td>"
            f"<td>{_fmt(b.get('mean_sdr_db'))}</td>"
            f"<td>{_fmt(b.get('harmonic',{}).get('sir_db'))}</td>"
            f"<td>{_fmt(b.get('percussive',{}).get('sir_db'))}</td>"
            f"<td>{_fmt(lk.get('harmonic',{}).get('crosstalk_fraction'))}</td>"
            f"<td>{_fmt(lk.get('percussive',{}).get('crosstalk_fraction'))}</td>"
            f"<td>{_fmt(en.get('recomposition_error_db'))}</td>"
            f"<td>{_fmt(en.get('delivered_over_input_db'))}</td>"
            f"<td>{_fmt(en.get('residual_energy_fraction'))}</td>"
            f"<td>{_fmt(d.get('ambiguous_energy_fraction'))}</td>"
            f"<td>{_fmt(mq.get('harmonic',{}).get('f1'))} / {_fmt(mq.get('percussive',{}).get('f1'))}</td>"
            f"<td>{links}<br><small>{figs}</small></td>"
            "</tr>"
        )

    pol = report.get("policy_comparison", {})
    pol_rows = []
    for scen, per_policy in pol.items():
        for policy, v in per_policy.items():
            pol_rows.append(
                "<tr>"
                f"<td>{esc(scen)}</td><td>{esc(policy)}</td>"
                f"<td>{_fmt(v['ambiguous_energy_fraction'])}</td>"
                f"<td>{_fmt(v['residual_energy_fraction'])}</td>"
                f"<td>{_fmt(v['shared_energy_duplicated'])}</td>"
                f"<td>{_fmt(v['mask_sum_max_abs_error'])}</td>"
                f"<td>{_fmt(v['delivered_over_input_db'])}</td>"
                f"<td>{_fmt(v['recomposition_error_db'])}</td>"
                f"<td>{_fmt(v['mean_sdr_db'])}</td>"
                f"<td>{_fmt(v['mean_sir_db'])}</td>"
                "</tr>"
            )

    scale = report.get("window_scales", {})
    scale_rows = []
    for scen, per_scale in scale.items():
        for n_fft, v in per_scale.items():
            scale_rows.append(
                "<tr>"
                f"<td>{esc(scen)}</td><td>{esc(n_fft)}</td>"
                f"<td>{_fmt(v['mean_sdr_db'])}</td>"
                f"<td>{_fmt(v['mean_sir_db'])}</td>"
                f"<td>{_fmt(v['recomposition_error_db'])}</td>"
                f"<td>{_fmt(v['ambiguous_energy_fraction'])}</td>"
                f"<td>{_fmt(v['mask_quality_harmonic_f1'])} / {_fmt(v['mask_quality_percussive_f1'])}</td>"
                "</tr>"
            )

    def table(header, body):
        h = "".join(f"<th>{esc(x)}</th>" for x in header)
        return f"<table><thead><tr>{h}</tr></thead><tbody>{''.join(body)}</tbody></table>"

    scalable = ["n_fft"]
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Harmonic / Percussive Decomposition Study</title>
<style>
body {{ font-family: -apple-system, Segoe UI, sans-serif; margin: 2rem auto; max-width: 1500px;
        padding: 0 1.5rem; color: #1a202c; line-height: 1.45; }}
h1, h2 {{ border-bottom: 1px solid #cbd5e0; padding-bottom: .3rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .86rem; }}
th, td {{ border: 1px solid #e2e8f0; padding: .35rem .45rem; text-align: left; vertical-align: top; }}
th {{ background: #edf2f7; position: sticky; top: 0; }}
tr:nth-child(even) td {{ background: #f7fafc; }}
small {{ color: #4a5568; }}
code, .mono {{ font-family: ui-monospace, monospace; }}
.note {{ background: #fffaf0; border-left: 4px solid #d69e2e; padding: .75rem 1rem; margin: 1rem 0; }}
.ok {{ color: #276749; font-weight: 600; }}
.bad {{ color: #9b2c2c; font-weight: 600; }}
img {{ max-width: 100%; border: 1px solid #e2e8f0; margin: .5rem 0; }}
</style></head><body>
<h1>Harmonic / Percussive Decomposition with Explicit Uncertainty</h1>
<p>Generated by <code>scripts/run_study.py</code>.  Sample rate {report['sample_rate']} Hz,
durations up to {report.get('duration_s')} s.  Audio stems and mask figures are linked from each row.</p>

<div class="note">
<strong>What is and is not claimed.</strong> This is a structural
harmonic/percussive separator driven by time- and frequency-direction median
supports, <em>not</em> a source classifier.  It does not label instruments and
does not assume that "high band = one instrument, low band = another"; two
parts are separated only when their short-time morphology differs.  The
"clean" scenarios satisfy that assumption.  The "counterexample" scenarios
deliberately violate it (vibrato, breathy noise, overlapping onsets, glissando,
modulated drone) and are expected to produce large cross-talk; their rows show
that rather than hiding it.  Ambiguous bins are routed by an explicit rule and
are never deleted: the delivered harmonic + percussive + residual stems always
remix to the input (see recomposition-error column), and no stem is
renormalized.
</div>

<h2>1. Scenarios vs known ground truth</h2>
<p>Audio: <code>results/audio/&lt;name&gt;__*.wav</code>.  Each estimate stem is also the
isolation ("solo") file for listening; <code>recomposition_diff.wav</code> is
remix minus input.  Mask figures show mixture |STFT|, each directional support,
the ambiguity map, and the three masks.</p>
{table(["scenario", "tags", "mean SDR dB", "H SIR dB", "P SIR dB", "H crosstalk", "P crosstalk",
        "recomposition err dB", "delivered vs input dB", "residual frac", "ambiguous frac",
        "mask F1 H / P", "audio + figures"], rows)}

<h2>2. Ambiguity policy comparison</h2>
<p>Identical supports, three rules for bins claimed by both structures.
<code>residual</code> (default) moves the ambiguous fraction to the residual stem, so
harmonic+percussive+residual still equals the input.
<code>shared</code> deliberately assigns the ambiguous fraction to <em>both</em> stems;
its mask sum then exceeds 1 and the remix overshoots by the reported amount — this
is shown on purpose so the duplication is measurable rather than implicit.
<code>complementary</code> ignores ambiguity and is the usual soft-mask baseline.</p>
{table(["scenario", "policy", "ambiguous frac", "residual frac", "duplicated frac",
        "mask-sum max |err|", "delivered vs input dB", "recomposition err dB",
        "mean SDR dB", "mean SIR dB"], pol_rows)}

<h2>3. Window / kernel scale sweep</h2>
<p>STFT size and median kernel change what counts as "smooth": short windows resolve
transients but blur sustained partials; long windows do the reverse.  Reported for the
dense-drum case where the two structures overlap most.</p>
{table(["scenario", "n_fft", "mean SDR dB", "mean SIR dB", "recomposition err dB",
        "ambiguous frac", "mask F1 H / P"], scale_rows)}

<h2>4. Hand-built mix and external audio</h2>
<p>See <code>results/metrics/report.json</code> for full detail.  The hand-built mix
contains a noise bed with no ground-truth class; its composition is reported, and the
decomposer is expected to split it (partly into the residual) rather than assign it to
an instrument.</p>

<h2>5. How to read the numbers</h2>
<ul>
<li><b>mean SDR</b>: mir_eval BSS SDR, which rescales estimates, so it measures shape
only.  <b>delivered vs input dB</b> is the unscaled level check.</li>
<li><b>H/P SIR</b>: interference rejection; the direct cross-talk measure.</li>
<li><b>crosstalk fraction</b>: time-domain projection of a stem onto the other true
source.  <b>unexplained</b> energy is not attributed to either source (a masking
nonlinearity or a source that is not a scaled version of the truth).</li>
<li><b>recomposition err</b>: −∞ means the three stems summed exactly to the input
(magnitude masks with phase preserved).  A finite value with policy
<code>shared</code> is the intentional duplication, not a bug.</li>
<li><b>mask F1</b>: agreement with an oracle ideal-ratio mask.  For level-imbalanced
mixtures this is dominated by the loud source and is not a correctness probability.</li>
</ul>
<p>Machine-readable metrics: <code>results/metrics/report.json</code>,
<code>results/metrics/summary.csv</code>.</p>
</body></html>
"""
    path.write_text(html)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="*", default=None,
                    help="subset of scenario names (default: all)")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--input", type=Path, default=None, help="optional user recording (.wav)")
    ap.add_argument("--no-sweeps", action="store_true", help="skip policy and window sweeps")
    args = ap.parse_args()

    AUDIO.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)

    names = args.scenarios or list(scenarios.BUILDERS)
    cfg = HPSSConfig()
    report = {
        "sample_rate": args.sr,
        "duration_s": args.duration,
        "default_config": cfg.to_dict(),
        "scenarios": [],
    }

    for name in names:
        print(f"[scenario] {name}")
        report["scenarios"].append(run_scenario(name, args.sr, args.duration, cfg))

    if not args.no_sweeps:
        sweep_targets = [n for n in ("dense_drums_with_sustain",) if n in scenarios.BUILDERS]
        report["policy_comparison"] = {
            n: run_policy_comparison(n, args.sr, args.duration) for n in sweep_targets
        }
        report["window_scales"] = {
            n: run_window_scales(n, args.sr, args.duration) for n in sweep_targets
        }
        print("[homemade] building hand-assembled mix")
        report["homemade"] = run_homemade_mix(args.sr, args.duration)
        report["scenarios"].append(report["homemade"])

    if args.input is not None:
        print(f"[external] {args.input}")
        ext = run_external(args.input, args.sr)
        if ext is None:
            print(f"  input not found: {args.input}", file=sys.stderr)
        else:
            report["external"] = ext

    _dump(METRICS / "report.json", report)
    write_csv(report["scenarios"], METRICS / "summary.csv")
    write_html(report, RESULTS / "report.html")
    viz.plot_summary(report["scenarios"], FIGURES / "summary.png")

    print(f"\nWrote {RESULTS/'report.html'}")
    print(f"Wrote {METRICS/'report.json'} and {METRICS/'summary.csv'}")
    print(f"Audio in {AUDIO}  ({len(list(AUDIO.glob('*.wav')))} files)")
    print(f"Figures in {FIGURES}  ({len(list(FIGURES.glob('*.png')))} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
