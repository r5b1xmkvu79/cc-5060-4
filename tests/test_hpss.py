"""Unit tests: STFT exactness, mask partition, attribution, amplitude rules.

Run with:  PYTHONPATH=vendor/python python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

import numpy as np

from hpss import HPSSConfig, STFTConfig, STFT, decompose, metrics, scenarios
from hpss.decompose import ambiguous_fraction, build_masks, soft_share


class TestSTFT(unittest.TestCase):
    def test_roundtrip_windows_and_hops(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(22050) * 0.1
        for window in ("hann", "hamming", "blackman"):
            for hop in (256, 512, 1024):
                cfg = STFTConfig(n_fft=2048, hop_length=hop, window=window)
                err = np.sum((x - STFT(cfg).roundtrip(x)) ** 2) / np.sum(x**2)
                self.assertLess(10 * np.log10(err), -250, f"{window} hop={hop}")

    def test_odd_and_empty_lengths(self):
        rng = np.random.default_rng(1)
        for n in (0, 1, 7, 100, 999, 12345):
            x = rng.standard_normal(n)
            y = STFT(STFTConfig(n_fft=512, hop_length=128)).roundtrip(x)
            self.assertEqual(y.shape, x.shape)

    def test_no_center(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(5000)
        cfg = STFTConfig(n_fft=512, hop_length=128, center=False, pad_mode="constant")
        y = STFT(cfg).roundtrip(x)
        # edge frames are not fully overlapped without centering; interior is exact
        interior = slice(1024, len(x) - 1024)
        self.assertLess(np.sum((x[interior] - y[interior]) ** 2) / np.sum(x**2), 1e-20)


class TestMasks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.share = rng.uniform(0.05, 0.95, size=(40, 25))
        self.amb = rng.uniform(0.0, 1.0, size=(40, 25)) * np.minimum(
            self.share, 1 - self.share
        ) * 2

    def test_partition_policies_sum_to_one(self):
        for policy in ("complementary", "residual", "split", "harmonic", "percussive"):
            for kind in ("soft", "hard"):
                mh, mp, mr = build_masks(self.share, self.amb, kind, policy)
                total = mh + mp + mr
                np.testing.assert_allclose(total, 1.0, atol=1e-12)
                self.assertGreaterEqual(mh.min(), 0.0)
                self.assertGreaterEqual(mp.min(), 0.0)
                self.assertGreaterEqual(mr.min(), 0.0)

    def test_shared_policy_duplicates_and_overshoots(self):
        mh, mp, mr = build_masks(self.share, self.amb, "soft", "shared")
        total = mh + mp + mr
        # overshoot is bounded by the minority share and is never > 2
        expected = 1.0 + 2.0 * np.minimum(self.share, 1 - self.share)
        np.testing.assert_allclose(total, expected, atol=1e-12)
        self.assertGreater(total.max(), 1.0)
        self.assertLessEqual(total.max(), 2.0 + 1e-12)

    def test_complementary_ignores_ambiguity(self):
        mh, mp, mr = build_masks(self.share, self.amb, "soft", "complementary")
        np.testing.assert_allclose(mh, self.share)
        np.testing.assert_allclose(mp, 1.0 - self.share)
        np.testing.assert_allclose(mr, 0.0)

    def test_residual_absorbs_ambiguous(self):
        mh, mp, mr = build_masks(self.share, self.amb, "soft", "residual")
        np.testing.assert_allclose(mh + mp, 1.0 - self.amb, atol=1e-12)
        np.testing.assert_allclose(mr, self.amb, atol=1e-12)


class TestDecompose(unittest.TestCase):
    def test_recomposition_exact_for_partition_policies(self):
        sc = scenarios.build("dense_drums_with_sustain")
        for policy in ("residual", "complementary", "split"):
            r = decompose(sc.mixture, sc.sample_rate, HPSSConfig(shared_policy=policy))
            remix = r.harmonic + r.percussive + r.residual
            err_db = 10 * np.log10(
                np.sum((remix - sc.mixture) ** 2) / np.sum(sc.mixture**2)
            )
            self.assertLess(err_db, -250)
            # no renormalization: delivered energy equals input energy
            ratio_db = 10 * np.log10(np.sum(remix**2) / np.sum(sc.mixture**2))
            self.assertAlmostEqual(ratio_db, 0.0, places=6)

    def test_shared_policy_overshoot_is_measured(self):
        sc = scenarios.build("dense_drums_with_sustain")
        r = decompose(sc.mixture, sc.sample_rate, HPSSConfig(shared_policy="shared"))
        remix = r.harmonic + r.percussive + r.residual
        ratio_db = 10 * np.log10(np.sum(remix**2) / np.sum(sc.mixture**2))
        self.assertGreater(ratio_db, 0.0)
        self.assertGreater(r.diagnostic["shared_energy_duplicated"], 0.0)

    def test_stems_are_time_aligned(self):
        sc = scenarios.build("sustain_plus_pulse")
        r = decompose(sc.mixture, sc.sample_rate)
        self.assertEqual(len(r.harmonic), len(sc.mixture))
        self.assertEqual(len(r.percussive), len(sc.mixture))
        self.assertEqual(len(r.residual), len(sc.mixture))

    def test_ambiguity_is_low_for_pure_transient(self):
        sr = 22050
        from hpss import signals

        p = signals.percussive_hit(0.5, sr, "snare", amplitude=0.8)
        r = decompose(p, sr)
        # a lone broadband hit has little temporal persistence
        self.assertLess(r.diagnostic["ambiguous_energy_fraction"], 0.10)

    def test_ambiguity_detects_overlap(self):
        sc = scenarios.build("noisy_sustain")
        r = decompose(sc.mixture, sc.sample_rate)
        self.assertGreater(r.diagnostic["ambiguous_energy_fraction"], 0.05)

    def test_empty_and_short_input(self):
        for x in (np.zeros(0), np.zeros(10), np.zeros(500)):
            r = decompose(x, 22050)
            self.assertEqual(len(r.harmonic), len(x))
            np.testing.assert_allclose(r.harmonic, 0.0)


class TestMetrics(unittest.TestCase):
    def test_composition_of_known_source_is_identity(self):
        sc = scenarios.build("sustain_plus_pulse")
        c = metrics.composition(sc.true_harmonic, {"h": sc.true_harmonic, "p": sc.true_percussive})
        self.assertAlmostEqual(c["h"], 1.0, places=6)
        self.assertAlmostEqual(c["p"], 0.0, places=6)

    def test_composition_of_sum(self):
        sc = scenarios.build("sustain_plus_pulse")
        mix = sc.true_harmonic + sc.true_percussive
        refs = {"h": sc.true_harmonic, "p": sc.true_percussive}
        # h-direction is attributed first: fraction_h = (||h||^2 + <h,p>) / ||h+p||^2
        e_h = np.dot(sc.true_harmonic, sc.true_harmonic)
        e_p = np.dot(sc.true_percussive, sc.true_percussive)
        cross = np.dot(sc.true_harmonic, sc.true_percussive)
        total = e_h + e_p + 2 * cross
        c = metrics.composition(mix, refs)
        self.assertAlmostEqual(c["h"], (e_h + cross) / total, places=4)
        self.assertAlmostEqual(c["p"], (e_p + cross) / total, places=4)

    def test_energy_report_detects_rescaling(self):
        sc = scenarios.build("sustain_plus_pulse")
        r = decompose(sc.mixture, sc.sample_rate)
        en = metrics.energy_report(
            sc.mixture, sc.true_harmonic, sc.true_percussive,
            r.harmonic * 2.0, r.percussive, r.residual,
        )
        # BSS would hide this; the energy report must not.
        self.assertGreater(en["est_h_over_true_h_db"], 5.0)


if __name__ == "__main__":
    unittest.main()
