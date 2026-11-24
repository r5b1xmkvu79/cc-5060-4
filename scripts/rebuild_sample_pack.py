#!/usr/bin/env python3
"""Rebuild the sample pack from saved parameters and verify it, then run the
legacy scenarios to completion as a regression.

Checks (any failure exits non-zero -- no silent pass):

1. The committed pack exists and every file listed in its manifest is present
   with a matching SHA-256 (the pack must really be there, not just claimed).
2. A fresh build into a temporary directory reproduces the committed files:
   WAV/NPY bytes must match exactly; evidence/verification JSON is compared
   by value with tight tolerance.
3. Entry metadata (coefficients, segments, null-test depths) rebuilds to the
   same values.
4. Every legacy scenario in ``hpss.scenarios.BUILDERS`` decomposes end-to-end
   with exact recomposition (< -200 dB).

All measurements are written to ``results/sample_pack_verify.json`` so the
component-leakage and recomposition numbers trace to executed evidence.

Usage:
    PYTHONPATH=vendor/python python3 scripts/rebuild_sample_pack.py [--pack sample_pack]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor" / "python"))

from hpss import HPSSConfig, decompose, metrics, scenarios  # noqa: E402
from make_sample_pack import build_pack, file_hash  # noqa: E402

VERIFY_JSON = ROOT / "results" / "sample_pack_verify.json"
VALUE_COMPARE_SUFFIXES = ("evidence.json", "verification.json")


def _sha256(path: Path) -> str:
    return file_hash(path)


def _close(a, b, rtol=1e-9, atol=1e-12, path=""):
    """Deep-compare two JSON-like structures; returns list of mismatch paths."""
    bad = []
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            bad.append(f"{path}: keys {sorted(set(a) ^ set(b))}")
        for k in a.keys() & b.keys():
            bad.extend(_close(a[k], b[k], rtol, atol, f"{path}.{k}"))
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            bad.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            bad.extend(_close(x, y, rtol, atol, f"{path}[{i}]"))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        fa, fb = float(a), float(b)
        if math.isinf(fa) or math.isinf(fb) or math.isnan(fa) or math.isnan(fb):
            if str(a) != str(b):
                bad.append(f"{path}: {a!r} != {b!r}")
        elif not math.isclose(fa, fb, rel_tol=rtol, abs_tol=atol):
            bad.append(f"{path}: {a!r} != {b!r}")
    elif a != b:
        bad.append(f"{path}: {a!r} != {b!r}")
    return bad


def verify_pack(pack_dir: Path, sr: int, seed: int) -> dict:
    pack_dir = Path(pack_dir)
    report: dict = {"pack_dir": str(pack_dir), "checks": {}, "ok": False}
    manifest_path = pack_dir / "manifest.json"
    if not manifest_path.exists():
        report["checks"]["manifest"] = {"ok": False, "error": f"missing {manifest_path}"}
        return report
    committed = json.loads(manifest_path.read_text())

    # 1. committed files exist and match their recorded hashes
    missing, sha_bad = [], []
    for rel, want in committed["files"].items():
        p = pack_dir / rel
        if not p.exists():
            missing.append(rel)
        elif _sha256(p) != want:
            sha_bad.append(rel)
    report["checks"]["committed_files"] = {
        "ok": not missing and not sha_bad,
        "n_files": len(committed["files"]),
        "missing": missing,
        "sha_mismatch": sha_bad,
    }

    # 2. fresh rebuild reproduces the pack
    with tempfile.TemporaryDirectory(prefix="hpss_pack_") as tmp:
        fresh = build_pack(Path(tmp), sr=sr, seed=seed)
        rebuilt = Path(tmp)
        sha_mismatch, value_mismatch = [], {}
        for rel, want in committed["files"].items():
            new_path = rebuilt / rel
            if not new_path.exists():
                sha_mismatch.append(f"{rel} (not rebuilt)")
                continue
            if rel.endswith(VALUE_COMPARE_SUFFIXES):
                old = json.loads((pack_dir / rel).read_text())
                new = json.loads(new_path.read_text())
                bad = _close(old, new, path=rel)
                if bad:
                    value_mismatch[rel] = bad[:20]
            elif _sha256(new_path) != want:
                sha_mismatch.append(rel)
    report["checks"]["rebuild"] = {
        "ok": not sha_mismatch and not value_mismatch,
        "sha_mismatch": sha_mismatch,
        "value_mismatch": value_mismatch,
    }

    # 3. entry metadata (coefficients, segments, null depths) rebuilds equal
    meta_bad = _close(committed["entries"], fresh["entries"], path="entries")
    report["checks"]["entry_metadata"] = {"ok": not meta_bad, "mismatches": meta_bad[:20]}

    # 4. leakage / recomposition evidence traced from the fresh estimates
    evidence_summary = {}
    for name in committed["entries"]:
        ev = json.loads((pack_dir / "evidence" / f"{name}__evidence.json").read_text())
        evidence_summary[name] = {
            "harmonic_crosstalk_fraction": ev["leakage"]["harmonic"]["crosstalk_fraction"],
            "percussive_crosstalk_fraction": ev["leakage"]["percussive"]["crosstalk_fraction"],
            "recomposition_error_db": ev["energy"]["recomposition_error_db"],
            "residual_energy_fraction": ev["energy"]["residual_energy_fraction"],
            "null_test_wav_db": ev["null_test"]["wav_roundtrip_db"],
        }
    report["evidence_summary"] = evidence_summary

    report["ok"] = all(c["ok"] for c in report["checks"].values())
    return report


def run_legacy_regression(sr: int = 22050, duration: float = 4.0) -> dict:
    """Decompose every legacy scenario end-to-end and record key metrics."""
    out = {}
    ok = True
    for name in scenarios.BUILDERS:
        sc = scenarios.build(name, sr=sr, duration=duration)
        result = decompose(sc.mixture, sc.sample_rate, HPSSConfig())
        ev = metrics.evaluate(sc, result)
        rec_db = ev["energy"]["recomposition_error_db"]
        ok &= bool(rec_db < -200.0)
        out[name] = {
            "mean_sdr_db": ev["bss"].get("mean_sdr_db"),
            "harmonic_crosstalk_fraction": ev["leakage"]["harmonic"]["crosstalk_fraction"],
            "percussive_crosstalk_fraction": ev["leakage"]["percussive"]["crosstalk_fraction"],
            "recomposition_error_db": rec_db,
            "residual_energy_fraction": ev["energy"]["residual_energy_fraction"],
            "ambiguous_energy_fraction": ev["diagnostic"]["ambiguous_energy_fraction"],
        }
    return {"ok": bool(ok), "scenarios": out}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", type=Path, default=ROOT / "sample_pack")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=VERIFY_JSON)
    args = ap.parse_args()

    report = verify_pack(args.pack, sr=args.sr, seed=args.seed)
    report["legacy_regression"] = run_legacy_regression(sr=args.sr)
    report["ok"] = bool(report["ok"] and report["legacy_regression"]["ok"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=float) + "\n")

    for check, res in report["checks"].items():
        print(f"[{'ok' if res['ok'] else 'FAIL'}] {check}")
    for name, s in report["evidence_summary"].items():
        print(f"  {name}: xtalk H {s['harmonic_crosstalk_fraction']:.4f} / "
              f"P {s['percussive_crosstalk_fraction']:.4f}, "
              f"recomposition {s['recomposition_error_db']:.1f} dB, "
              f"null {s['null_test_wav_db']:.1f} dB")
    leg = report["legacy_regression"]
    print(f"[{'ok' if leg['ok'] else 'FAIL'}] legacy regression "
          f"({len(leg['scenarios'])} scenarios run to completion)")
    print(f"Wrote {args.out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
