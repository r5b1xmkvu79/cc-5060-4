"""Tests for the sample pack: synthesis, null test, segment edits, pack
build/verify round-trip, and the spectral-editor HTTP API.

Run with:  PYTHONPATH=vendor/python python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from hpss import samplepack, signals  # noqa: E402
from hpss.config import HPSSConfig  # noqa: E402


class TestSignals(unittest.TestCase):
    def test_brush_sweep_deterministic_and_bounded(self):
        a = signals.brush_sweep(1.0, 22050, amplitude=0.5, seed=7)
        b = signals.brush_sweep(1.0, 22050, amplitude=0.5, seed=7)
        np.testing.assert_array_equal(a, b)
        self.assertLessEqual(np.max(np.abs(a)), 0.5 + 1e-9)
        self.assertGreater(np.std(a), 0.01)  # not silence


class TestCompose(unittest.TestCase):
    def test_stems_sum_to_mixture_and_null_is_exact(self):
        for name in samplepack.ENTRY_NAMES:
            entry = samplepack.build_entry(name)
            h, p, mix, scale = samplepack.compose(entry)
            np.testing.assert_allclose(h + p, mix, atol=1e-12)
            null_p = samplepack.phase_cancel_percussive(mix, h)
            self.assertLess(samplepack.null_depth_db(p, null_p), -250.0)
            self.assertGreater(scale, 0.0)

    def test_coefficients_applied(self):
        entry = samplepack.build_entry("sustained_chords_sparse_drums")
        h, p, mix, scale = samplepack.compose(entry)
        self.assertAlmostEqual(entry.harmonic_gain, 1.0)
        self.assertAlmostEqual(entry.percussive_gain, 0.9)
        self.assertAlmostEqual(np.max(np.abs(mix)), entry.peak_target, places=6)


class TestSegmentEdit(unittest.TestCase):
    def setUp(self):
        self.entry = samplepack.build_entry("brush_sweep_over_sustain")
        self.owners = ["percussive"] * 8
        self.owners[2] = "harmonic"
        self.owners[5] = "harmonic"

    def test_edit_rebuilds_consistent_sources_mixture_and_masks(self):
        v = samplepack.verify_edit(self.entry, self.owners)
        self.assertTrue(v["ok"])
        self.assertEqual(v["changed_segments"], [2, 5])
        self.assertLess(v["remix_error_db"], -250.0)
        self.assertLess(v["phase_cancel_null_db"], -250.0)
        self.assertLess(v["mask_sum_max_abs_error"], 1e-6)

    def test_mask_change_concentrated_in_swapped_segments(self):
        v = samplepack.verify_edit(self.entry, self.owners)
        change = {s["index"]: s["mask_change"] for s in v["per_segment_mask_change"]}
        self.assertGreater(change[2], 0.5)
        self.assertGreater(change[5], 0.5)
        for far in (0, 4, 7):  # segments away from the sweeps stay put
            self.assertLess(change[far], 0.01)

    def test_wrong_owner_count_rejected(self):
        with self.assertRaises(ValueError):
            samplepack.verify_edit(self.entry, ["harmonic"])

    def test_routing_mask_crossfade_is_click_free(self):
        segs = samplepack.edited_segments(self.entry, self.owners)
        m = samplepack.routing_mask(segs, self.entry.n_samples, self.entry.sample_rate, "percussive")
        self.assertLessEqual(np.max(np.abs(np.diff(m))), 0.5)  # no unit jumps


class TestStrengthMapping(unittest.TestCase):
    def test_endpoints_and_clamping(self):
        lo = samplepack.strength_to_config(0.0)
        hi = samplepack.strength_to_config(1.0)
        self.assertIsInstance(lo, HPSSConfig)
        self.assertEqual((lo.kernel_time, lo.kernel_freq), (15, 15))
        self.assertEqual((hi.kernel_time, hi.kernel_freq), (55, 55))
        self.assertAlmostEqual(lo.mask_power, 1.0)
        self.assertAlmostEqual(hi.mask_power, 6.0)
        np.testing.assert_equal(
            samplepack.strength_to_config(-1).kernel_time, lo.kernel_time
        )
        np.testing.assert_equal(
            samplepack.strength_to_config(2).kernel_time, hi.kernel_time
        )


class TestPackBuildVerify(unittest.TestCase):
    def test_build_then_verify_roundtrip(self):
        from make_sample_pack import build_pack
        from rebuild_sample_pack import verify_pack

        with tempfile.TemporaryDirectory() as tmp:
            pack = Path(tmp) / "pack"
            manifest = build_pack(pack)
            self.assertEqual(len(manifest["entries"]), len(samplepack.ENTRY_NAMES))
            for rel, digest in manifest["files"].items():
                self.assertTrue((pack / rel).exists(), rel)
                self.assertEqual(len(digest), 64)
            report = verify_pack(pack, sr=manifest["sample_rate"], seed=manifest["seed"])
            self.assertTrue(report["checks"]["committed_files"]["ok"])
            self.assertTrue(report["checks"]["rebuild"]["ok"], report["checks"]["rebuild"])
            self.assertTrue(report["checks"]["entry_metadata"]["ok"])
            self.assertTrue(report["ok"])

    def test_verify_detects_tampered_file(self):
        from make_sample_pack import build_pack
        from rebuild_sample_pack import verify_pack

        with tempfile.TemporaryDirectory() as tmp:
            pack = Path(tmp) / "pack"
            manifest = build_pack(pack)
            victim = pack / next(iter(manifest["files"]))
            victim.write_bytes(b"tampered")
            report = verify_pack(pack, sr=manifest["sample_rate"], seed=manifest["seed"])
            self.assertFalse(report["checks"]["committed_files"]["ok"])
            self.assertFalse(report["ok"])


class TestUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from make_sample_pack import build_pack
        import hpss_ui

        cls.tmp = tempfile.TemporaryDirectory()
        pack = Path(cls.tmp.name) / "pack"
        build_pack(pack)
        kernel = hpss_ui.Kernel(pack, Path(cls.tmp.name) / "cache")
        hpss_ui.Handler.kernel = kernel
        cls.httpd = hpss_ui.ThreadingHTTPServer(("127.0.0.1", 0), hpss_ui.Handler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.tmp.cleanup()

    def _get(self, path, expect=200):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}") as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def test_index_page_served(self):
        status, body = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"separation strength", body)

    def test_entries_and_preview_and_solo_stem(self):
        status, body = self._get("/api/entries")
        self.assertEqual(status, 200)
        entries = json.loads(body)["entries"]
        self.assertEqual(len(entries), 2)
        status, body = self._get("/api/preview?name=brush_sweep_over_sustain&strength=0.6")
        self.assertEqual(status, 200)
        j = json.loads(body)
        self.assertEqual(j["config"]["kernel_time"], 39)
        for layer in ("mixture", "harmonic", "percussive", "residual"):
            self.assertIn(layer, j["images"])
            self.assertIn(layer, j["stems"])
        status, wav = self._get(j["stems"]["harmonic"])
        self.assertEqual(status, 200)
        self.assertEqual(wav[:4], b"RIFF")

    def test_edit_endpoint_rebuilds_and_verifies(self):
        owners = ["percussive"] * 8
        owners[1] = "harmonic"
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/edit?name=brush_sweep_over_sustain",
            data=json.dumps({"owners": owners}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as r:
            j = json.loads(r.read())
        self.assertTrue(j["verification"]["ok"])
        self.assertEqual(j["verification"]["changed_segments"], [1])
        self.assertIn("edited_harmonic", j["stems"])

    def test_kernel_failure_is_reported_not_hidden(self):
        status, body = self._get("/api/preview?name=does_not_exist&strength=0.5")
        self.assertEqual(status, 500)
        self.assertIn("error", json.loads(body))

    def test_path_traversal_blocked(self):
        status, _ = self._get("/pack/../README.md")
        self.assertEqual(status, 404)

    def test_audition_relative_stem_route(self):
        status, body = self._get("/audition/")
        self.assertEqual(status, 200)
        self.assertIn(b"../sources/", body)
        status, wav = self._get("/sources/brush_sweep_over_sustain__true_harmonic.wav")
        self.assertEqual(status, 200)
        self.assertEqual(wav[:4], b"RIFF")


if __name__ == "__main__":
    unittest.main()
