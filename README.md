# Harmonic / Percussive Decomposition with Explicit Uncertainty

A structural separator that splits a mono recording into **harmonic**, **percussive**
and **residual** stems using short-time spectrogram morphology, with an explicit,
measurable rule for bins that both structures can claim.  It is not a source
classifier: it does not label instruments and does not equate frequency bands with
instrument classes.

## What it does

1. STFT with a centred Hann window; synthesis is the exact least-squares inverse
   (overlap-add divided by overlap-added squared window), verified to < -250 dB.
2. Estimate two **supports** in the short-time magnitude:
   - harmonic support = median filter along **time** (energy smooth over frames),
   - percussive support = median filter along **frequency** (energy broadband),
   applied to the power spectrogram with configurable kernel sizes.
3. Soft harmonic share `H^(p/2) / (H^(p/2) + P^(p/2))` gives the base split.
4. **Ambiguity** `min(H, P) / |X|^2` per bin = fraction of the bin power both
   structures can explain.  Bins below `ambiguity_floor_db` relative to the peak
   are treated as silence so near-zero ratios do not create false ambiguity.
5. An explicit **policy** routes the ambiguous fraction (see below).  Masks are
   magnitude masks, so phase is preserved and the frames add back.
6. Reconstruct all stems and report recomposition error, energy ratios, cross-talk
   and mask fidelity.

## Ambiguity policies (`shared_policy`)

| policy | ambiguous fraction | mask sum | remix |
| --- | --- | --- | --- |
| `residual` (default) | moved to the residual stem | exactly 1 | equals input |
| `complementary` | ignored (base split) | exactly 1 | equals input |
| `split` | split 50/50 between stems | exactly 1 | equals input |
| `harmonic` / `percussive` | given entirely to one stem | exactly 1 | equals input |
| `shared` | **duplicated into both** stems | `1 + 2·min(share,1−share)` ≤ 2 | overshoots input |

Only `shared` is non-conservative.  Its overshoot is intentional and is reported
(`mask_sum_max_abs_error`, `shared_energy_duplicated`, `delivered_over_input_db`,
`recomposition_error_db`).  It exists so the cost of "giving the ambiguous sound to
both" is visible rather than hidden.  All other policies preserve energy exactly:
`harmonic + percussive + residual == input` to numerical precision, and no stem is
renormalized.

## Why the residual is never deleted

When a bin is ambiguous the default rule puts it in the residual stem rather than
dropping it or guessing.  This keeps the delivered set lossless; the residual is
delivered as its own WAV and its composition (fraction attributable to each true
source vs neither) is measured.  A clean UI is not allowed to hide missing audio.

## Assumptions and counterexamples

The morphology assumption is: harmonic sounds are smooth along time, percussive
sounds are broadband along frequency.  It fails when:

- **vibrato** smears partials in frequency (`vibrato_sustain`),
- **breathy/noisy sustains** are broadband yet sustained (`noisy_sustain`),
- **overlapping onsets** put a tonal attack inside a drum hit (`overlapping_attacks`),
- **glissando** is not a stationary line (`glissando_sustain`),
- **modulated drones** create periodic transients inside a tonal source (`percussive_drone`).

Each is a built-in scenario with ground truth.  The report shows the resulting
cross-talk, negative SIR and mask-F1 drop instead of asserting the output is
correct.  On `noisy_sustain` the percussive SIR goes negative (-16 dB): the noise
inside the sustain is genuinely claimed by the percussive model, which is the
honest outcome for this assumption, not a bug to mask.

## Evaluation

Metrics are deliberately multi-view because none is sufficient alone:

- **BSS SDR/SIR/SAR** (mir_eval): scale-invariant, so shape only.  SIR is cross-talk.
- **Projection composition**: time-domain energy split of each stem onto the
  orthogonalized true sources, giving self / other-source (cross-talk) / unexplained
  fractions.  `unexplained` is non-zero even for good separation because a
  time-varying mask applied to a source is not a scalar multiple of it; compare
  across methods rather than expecting zero.
- **Energy report**: unscaled levels, `est/true` per stem, delivered/input dB,
  recomposition error, residual fraction.  Catches level errors that BSS normalizes
  away.
- **Mask F1**: agreement with an oracle ideal-ratio mask.  For strongly level-
  imbalanced mixes this is dominated by the loud source and is not a probability of
  correctness.

Ground truth is provided for synthetic scenarios (sustained tones + short pulses,
dense 16th-note drums, level imbalance) and a hand-built mix with irregular /
overlapping hits and a noise bed.  The noise bed has no ground-truth class; its
split is reported.

## Layout

```
hpss/            library (stft, decompose, signals, scenarios, metrics, viz, io)
scripts/run_study.py    full study -> results/report.html, CSV, JSON, audio, figures
scripts/separate.py     CLI: one WAV -> harmonic/percussive/residual WAVs
tests/test_hpss.py      unit tests (reconstruction, mask partition, energy, metrics)
results/
  audio/         mixture, true components, estimated stems, remix, recomposition diff
  figures/       decomposition and mask figures, summary
  metrics/       report.json, summary.csv
```

## Reproduce

Dependencies are installed locally under `vendor/python` (git-ignored); there is no
system pip in this environment.

```bash
# install (once)
python3 -m pip install --target vendor/python -r requirements.txt   # needs a pip
# this environment used an extracted pip wheel because ensurepip is unavailable

# full study
PYTHONPATH=vendor/python python3 scripts/run_study.py

# tests
PYTHONPATH=vendor/python python3 -m unittest discover -s tests -v

# single file
PYTHONPATH=vendor/python python3 scripts/separate.py my.wav --out-dir out --policy residual
```

Open `results/report.html` for the tables, links to every stem WAV for isolated
listening, and the mask figures.

## Tuning

- `n_fft` / `hop_length` / `window`: time-frequency resolution.  Short windows
  resolve transients but blur partials; long windows do the reverse (see the
  window-scale sweep in the report for the dense-drum case).
- `kernel_time` / `kernel_freq`: median kernel length along each axis, in frames /
  bins.  Too short loses the smoothness that defines each structure.
- `mask_power`, `mask_kind`, `ambiguity_floor_db`, `shared_policy`,
  `duplicate_weight`: mask sharpness, hard/soft, ambiguity gate and ownership rule.
