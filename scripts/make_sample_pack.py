#!/usr/bin/env python3
"""Build the audio decomposition sample pack into ``sample_pack/``.

For every entry the pack contains: the known-source stems, the forward-
synthesized mixture (coefficients stored in the manifest), a phase-cancelled
percussive reference with its measured null depth, oracle reference masks,
the *actual* decomposer output (never the true sources), leakage and
recomposition evidence, and -- for the brush-sweep entry -- a shipped segment
edit with its rebuild verification.  An audition project (self-contained HTML
+ project.json) rounds it out.

Usage:
    PYTHONPATH=vendor/python python3 scripts/make_sample_pack.py [--out sample_pack]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "python"))

from hpss import HPSSConfig, decompose, metrics, samplepack, scenarios  # noqa: E402
from hpss import io as hpss_io  # noqa: E402

PACK_FORMAT = 1
DEMO_EDIT_OWNERS = {2: "harmonic", 5: "harmonic"}  # brush-sweep segment indices


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


def wav_data_bytes(path: Path) -> bytes:
    """The audio payload of a WAV file (its ``data`` chunk).

    libsndfile writes a ``PEAK`` chunk containing a wall-clock timestamp, so
    full-file hashes are not reproducible; the audio payload is.
    """
    b = Path(path).read_bytes()
    if b[:4] != b"RIFF" or b[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a RIFF/WAVE file")
    off = 12
    while off + 8 <= len(b):
        cid = b[off : off + 4]
        size = int.from_bytes(b[off + 4 : off + 8], "little")
        if cid == b"data":
            return b[off + 8 : off + 8 + size]
        off += 8 + size + (size & 1)
    raise ValueError(f"{path} has no data chunk")


def file_hash(path: Path) -> str:
    """Content hash: WAV files are hashed on their audio payload only.

    Falls back to the raw file bytes when a ``.wav`` does not parse (e.g. a
    corrupted or tampered file), so integrity checks still report a mismatch
    instead of crashing.
    """
    path = Path(path)
    if path.suffix.lower() == ".wav":
        try:
            return hashlib.sha256(wav_data_bytes(path)).hexdigest()
        except ValueError:
            pass
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256(path: Path) -> str:
    return file_hash(path)


def _write_npy(path: Path, arr: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return path


def _scenario_of(entry: samplepack.PackEntry, h, p, mix) -> scenarios.Scenario:
    return scenarios.Scenario(
        name=entry.name,
        sample_rate=entry.sample_rate,
        mixture=mix,
        true_harmonic=h,
        true_percussive=p,
        harmonic_gain=entry.harmonic_gain,
        percussive_gain=entry.percussive_gain,
        notes=entry.notes,
        tags=list(entry.tags),
        meta=dict(entry.params),
    )


def build_pack(out_dir: Path, sr: int = 22050, seed: int = 0) -> dict:
    """Build every entry and return the manifest (also written to disk)."""
    out_dir = Path(out_dir)
    files: dict[str, str] = {}
    entries: dict[str, dict] = {}
    audition_entries = []

    def track(path: Path) -> str:
        rel = str(path.relative_to(out_dir))
        files[rel] = _sha256(path)
        return rel

    for name in samplepack.ENTRY_NAMES:
        entry = samplepack.build_entry(name, sr=sr, seed=seed)
        true_h, true_p, mix, scale = samplepack.compose(entry)

        # -- sources + forward-synthesized mixture -------------------------
        rel_h = track(hpss_io.write_wav(out_dir / "sources" / f"{name}__true_harmonic.wav", true_h, sr))
        rel_p = track(hpss_io.write_wav(out_dir / "sources" / f"{name}__true_percussive.wav", true_p, sr))
        rel_mix = track(hpss_io.write_wav(out_dir / "mixtures" / f"{name}__mixture.wav", mix, sr))

        # -- phase-cancellation reference ----------------------------------
        null_mem = samplepack.phase_cancel_percussive(mix, true_h)
        null_mem_db = samplepack.null_depth_db(true_p, null_mem)
        wav_mix, _ = hpss_io.read_wav(out_dir / "mixtures" / f"{name}__mixture.wav")
        wav_h, _ = hpss_io.read_wav(out_dir / "sources" / f"{name}__true_harmonic.wav")
        null_wav = samplepack.phase_cancel_percussive(wav_mix, wav_h)
        null_wav_db = samplepack.null_depth_db(true_p, null_wav)
        rel_null = track(hpss_io.write_wav(
            out_dir / "mixtures" / f"{name}__null_derived_percussive.wav", null_wav, sr
        ))

        # -- oracle reference masks ----------------------------------------
        m_h, m_p = samplepack.reference_masks(true_h, true_p)
        rel_mh = track(_write_npy(out_dir / "reference" / f"{name}__oracle_mask_harmonic.npy", m_h))
        rel_mp = track(_write_npy(out_dir / "reference" / f"{name}__oracle_mask_percussive.npy", m_p))

        # -- actual decomposer output (algorithm, not source substitution) --
        cfg = HPSSConfig()
        result = decompose(mix, sr, cfg)
        rel_eh = track(hpss_io.write_wav(out_dir / "estimates" / f"{name}__est_harmonic.wav", result.harmonic, sr))
        rel_ep = track(hpss_io.write_wav(out_dir / "estimates" / f"{name}__est_percussive.wav", result.percussive, sr))
        rel_er = track(hpss_io.write_wav(out_dir / "estimates" / f"{name}__est_residual.wav", result.residual, sr))

        sc = _scenario_of(entry, true_h, true_p, mix)
        ev = metrics.evaluate(sc, result)
        evidence = {
            "entry": name,
            "estimated_by": "hpss.decompose.decompose on the shipped mixture; "
                            "true sources are never substituted into the output",
            "decomposer_config": cfg.to_dict(),
            "diagnostic": result.diagnostic,
            "bss": ev["bss"],
            "leakage": ev["leakage"],
            "energy": ev["energy"],
            "residual_composition": ev["residual_composition"],
            "mask_quality": ev["mask_quality"],
            "null_test": {
                "construction": "null_percussive = mixture - true_harmonic",
                "in_memory_db": null_mem_db,
                "wav_roundtrip_db": null_wav_db,
            },
        }
        ev_path = out_dir / "evidence" / f"{name}__evidence.json"
        _dump(ev_path, evidence)
        rel_ev = track(ev_path)

        # -- shipped segment edit (editable entries only) -------------------
        edit_info = None
        if entry.assignable is not None and entry.segments:
            owners = [
                DEMO_EDIT_OWNERS.get(seg.index, seg.owner) for seg in entry.segments
            ]
            verification = samplepack.verify_edit(entry, owners)
            segs = samplepack.edited_segments(entry, owners)
            eh, ep, emix, _ = samplepack.compose(entry, segs)
            em_h, em_p = samplepack.reference_masks(eh, ep)
            edit_json = {
                "entry": name,
                "assignable_layer": entry.assignable,
                "owners": owners,
                "changed_segments": verification["changed_segments"],
            }
            edit_path = out_dir / "edits" / f"{name}__segment_edit.json"
            _dump(edit_path, edit_json)
            rel_edit = track(edit_path)
            rel_eh2 = track(hpss_io.write_wav(out_dir / "edits" / f"{name}__edited_true_harmonic.wav", eh, sr))
            rel_ep2 = track(hpss_io.write_wav(out_dir / "edits" / f"{name}__edited_true_percussive.wav", ep, sr))
            rel_emix = track(hpss_io.write_wav(out_dir / "edits" / f"{name}__edited_mixture.wav", emix, sr))
            rel_emh = track(_write_npy(out_dir / "edits" / f"{name}__edited_mask_harmonic.npy", em_h))
            rel_emp = track(_write_npy(out_dir / "edits" / f"{name}__edited_mask_percussive.npy", em_p))
            ver_path = out_dir / "edits" / f"{name}__verification.json"
            _dump(ver_path, verification)
            rel_ver = track(ver_path)
            edit_info = {
                "edit": rel_edit,
                "edited_true_harmonic": rel_eh2,
                "edited_true_percussive": rel_ep2,
                "edited_mixture": rel_emix,
                "edited_mask_harmonic": rel_emh,
                "edited_mask_percussive": rel_emp,
                "verification": rel_ver,
            }

        entries[name] = {
            "seed": seed,
            "sample_rate": sr,
            "duration_s": entry.duration,
            "params": entry.params,
            "coefficients": {
                "harmonic_gain": entry.harmonic_gain,
                "percussive_gain": entry.percussive_gain,
                "peak_target": entry.peak_target,
                "scale": scale,
                "construction": "mixture = scale * (harmonic_gain * H + percussive_gain * P)",
            },
            "base_owner": entry.base_owner,
            "assignable_layer": entry.assignable,
            "segments": [s.to_dict() for s in entry.segments],
            "decomposer_config": cfg.to_dict(),
            "null_test_db": {"in_memory": null_mem_db, "wav_roundtrip": null_wav_db},
            "files": {
                "true_harmonic": rel_h,
                "true_percussive": rel_p,
                "mixture": rel_mix,
                "null_derived_percussive": rel_null,
                "oracle_mask_harmonic": rel_mh,
                "oracle_mask_percussive": rel_mp,
                "est_harmonic": rel_eh,
                "est_percussive": rel_ep,
                "est_residual": rel_er,
                "evidence": rel_ev,
            },
            "edit": edit_info,
            "notes": entry.notes,
            "tags": entry.tags,
        }
        audition_entries.append((name, entries[name]))

    # -- audition project ----------------------------------------------------
    project = {
        "format": PACK_FORMAT,
        "title": "HPSS decomposition sample pack -- audition project",
        "entries": [
            {
                "name": name,
                "notes": meta["notes"],
                "stems": {
                    **{k: f"../{v}" for k, v in meta["files"].items() if k != "evidence"},
                    **(
                        {
                            f"edit_{k}": f"../{v}"
                            for k, v in meta["edit"].items()
                            if k.endswith(".wav") or "edited_" in k
                        }
                        if meta["edit"]
                        else {}
                    ),
                },
            }
            for name, meta in audition_entries
        ],
    }
    proj_path = out_dir / "audition" / "project.json"
    _dump(proj_path, project)
    track(proj_path)
    html_path = out_dir / "audition" / "index.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(_audition_html(project))
    track(html_path)

    manifest = {
        "format": PACK_FORMAT,
        "sample_rate": sr,
        "seed": seed,
        "hash_scheme": "sha256; WAV files hashed on their audio data chunk "
                       "(libsndfile PEAK chunks embed a wall-clock timestamp)",
        "generator": {
            "script": "scripts/make_sample_pack.py",
            "module": "hpss.samplepack",
            "rebuild": "scripts/rebuild_sample_pack.py",
        },
        "entries": entries,
        "audition_project": "audition/project.json",
        "files": files,
    }
    _dump(out_dir / "manifest.json", manifest)
    return manifest


def _audition_html(project: dict) -> str:
    rows = []
    for e in project["entries"]:
        players = []
        for stem, rel in e["stems"].items():
            if rel.endswith(".wav"):
                players.append(
                    f'<div class="stem"><span>{stem}</span>'
                    f'<audio controls preload="none" src="{rel}"></audio></div>'
                )
        rows.append(f"<h2>{e['name']}</h2><p>{e['notes']}</p>" + "".join(players))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{project['title']}</title>
<style>
body {{ font-family: -apple-system, Segoe UI, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; }}
.stem {{ display: flex; align-items: center; gap: .75rem; margin: .25rem 0; }}
.stem span {{ width: 240px; font-family: ui-monospace, monospace; font-size: .8rem; }}
h2 {{ border-bottom: 1px solid #cbd5e0; padding-bottom: .2rem; margin-top: 2rem; }}
</style></head><body>
<h1>{project['title']}</h1>
<p>Known-source mixtures for harmonic/percussive decomposition.  True stems,
forward-synthesized mixture, phase-cancelled percussive reference, and the
actual decomposer output side by side.  Open the spectral editor for
interactive separation (see README, <code>scripts/hpss_ui.py</code>).</p>
{''.join(rows)}
</body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "sample_pack")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    manifest = build_pack(args.out, sr=args.sr, seed=args.seed)
    n_files = len(manifest["files"])
    print(f"Wrote sample pack to {args.out} ({n_files} tracked files)")
    for name, meta in manifest["entries"].items():
        nt = meta["null_test_db"]
        print(f"  {name}: null depth {nt['wav_roundtrip']:.1f} dB (wav), "
              f"{nt['in_memory']:.1f} dB (in-memory)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
