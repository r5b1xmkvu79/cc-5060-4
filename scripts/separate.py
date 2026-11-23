#!/usr/bin/env python3
"""Separate one audio file into harmonic, percussive and residual stems.

Example:
    PYTHONPATH=vendor/python python3 scripts/separate.py in.wav --out-dir out \
        --policy residual
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "python"))

from hpss import HPSSConfig, STFTConfig, decompose, viz  # noqa: E402
from hpss import io as hpss_io  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("out"))
    ap.add_argument("--policy", default="residual")
    ap.add_argument("--n-fft", type=int, default=2048)
    ap.add_argument("--hop", type=int, default=None)
    ap.add_argument("--kernel", type=int, default=31)
    ap.add_argument("--window", default="hann")
    ap.add_argument("--mask-power", type=float, default=2.0)
    ap.add_argument("--ambiguity-floor-db", type=float, default=-40.0)
    ap.add_argument("--mask-kind", default="soft", choices=("soft", "hard"))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    x, sr = hpss_io.read_wav(args.input)
    cfg = HPSSConfig(
        stft=STFTConfig(n_fft=args.n_fft, hop_length=args.hop, window=args.window),
        kernel_time=args.kernel,
        kernel_freq=args.kernel,
        shared_policy=args.policy,
        mask_power=args.mask_power,
        mask_kind=args.mask_kind,
        ambiguity_floor_db=args.ambiguity_floor_db,
    )
    result = decompose(x, sr, cfg)

    out = args.out_dir
    hpss_io.write_wav(out / "harmonic.wav", result.harmonic, sr)
    hpss_io.write_wav(out / "percussive.wav", result.percussive, sr)
    hpss_io.write_wav(out / "residual.wav", result.residual, sr)
    hpss_io.write_wav(out / "remix.wav", result.harmonic + result.percussive + result.residual, sr)
    if not args.no_figures:
        viz.plot_decomposition(result, out / "decomposition.png",
                               title=f"{args.input.name} [{args.policy}]")

    summary = {
        "input": str(args.input),
        "sample_rate": sr,
        "duration_s": len(x) / sr,
        "config": cfg.to_dict(),
        **{k: v for k, v in result.diagnostic.items() if k != "config"},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps(summary, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
