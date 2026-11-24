#!/usr/bin/env python3
"""Spectral editing window for the harmonic/percussive sample pack.

A local web UI driven by the hpss kernel: the separation-strength slider and
the solo buttons call :func:`hpss.decompose.decompose` on the pack mixture
live; the preview area shows the mixture, harmonic and percussive layers as
spectrograms; entries with an assignable layer expose the segment editor
whose rebuild is checked by :func:`hpss.samplepack.verify_edit`.  Kernel
failures are returned to the page and appended to
``results/ui_feedback.log`` -- they are never hidden.

Launch:
    PYTHONPATH=vendor/python python3 scripts/hpss_ui.py --port 8000
then open http://127.0.0.1:8000/ in a browser.  The pack audition page is at
http://127.0.0.1:8000/audition/ .
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "python"))

from hpss import STFTConfig, STFT, decompose, metrics, samplepack, viz  # noqa: E402
from hpss import io as hpss_io  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402  (hpss.viz already selected Agg)

CACHE = ROOT / ".cache" / "hpss_ui"
FEEDBACK_LOG = ROOT / "results" / "ui_feedback.log"
PACK_DIR = ROOT / "sample_pack"

MIME = {
    ".html": "text/html; charset=utf-8",
    ".json": "application/json",
    ".wav": "audio/wav",
    ".png": "image/png",
    ".npy": "application/octet-stream",
}


def log_feedback(route: str, message: str) -> None:
    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_LOG.open("a") as fh:
        fh.write(json.dumps({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "route": route,
            "error": message,
        }) + "\n")


def spectrogram_png(x: np.ndarray, sr: int, path: Path, n_fft: int = 1024) -> None:
    st = STFT(STFTConfig(n_fft=n_fft, hop_length=n_fft // 4))
    mag = np.abs(st.forward(np.asarray(x, dtype=np.float64)))
    peak = float(mag.max()) if mag.size else 0.0
    db = np.maximum(20.0 * np.log10(np.maximum(mag, 1e-12) / (peak or 1.0)), -90.0)
    fig, ax = plt.subplots(figsize=(8.0, 2.4), constrained_layout=True)
    ax.imshow(db, origin="lower", aspect="auto",
              extent=(0.0, len(x) / sr, 0.0, sr / 2.0), vmin=-90, vmax=0, cmap="magma")
    ax.set_xlabel("s")
    ax.set_ylabel("Hz")
    fig.savefig(path, dpi=110)
    plt.close(fig)


class Kernel:
    """Wraps the hpss kernel for the UI: decomposition, edits, evidence."""

    def __init__(self, pack_dir: Path, cache_dir: Path):
        self.pack_dir = Path(pack_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.pack_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"sample pack not found at {self.pack_dir}; "
                "run scripts/make_sample_pack.py first"
            )
        self.manifest = json.loads(manifest_path.read_text())

    def entries(self) -> dict:
        out = []
        for name, meta in self.manifest["entries"].items():
            out.append({
                "name": name,
                "notes": meta["notes"],
                "duration_s": meta["duration_s"],
                "assignable_layer": meta["assignable_layer"],
                "segments": meta["segments"],
                "null_test_db": meta["null_test_db"],
                "pack_stems": {k: f"/pack/{v}" for k, v in meta["files"].items()
                               if v.endswith(".wav")},
            })
        return {"entries": out}

    def _mixture(self, name: str) -> tuple[np.ndarray, int]:
        meta = self.manifest["entries"][name]
        return hpss_io.read_wav(self.pack_dir / meta["files"]["mixture"])

    def preview(self, name: str, strength: float) -> dict:
        if name not in self.manifest["entries"]:
            raise KeyError(f"unknown entry {name!r}")
        cfg = samplepack.strength_to_config(strength)
        x, sr = self._mixture(name)
        result = decompose(x, sr, cfg)
        key = f"{name}__s{int(round(strength * 100)):03d}"
        stems = {"harmonic": result.harmonic, "percussive": result.percussive,
                 "residual": result.residual}
        urls = {}
        for stem, sig in stems.items():
            hpss_io.write_wav(self.cache_dir / f"{key}__{stem}.wav", sig, sr)
            urls[stem] = f"/cache/{key}__{stem}.wav"
        images = {}
        for layer, sig in {**stems, "mixture": x}.items():
            png = self.cache_dir / f"{key}__spec_{layer}.png"
            spectrogram_png(sig, sr, png)
            images[layer] = f"/cache/{png.name}"
        meta = self.manifest["entries"][name]
        true_h, _ = hpss_io.read_wav(self.pack_dir / meta["files"]["true_harmonic"])
        true_p, _ = hpss_io.read_wav(self.pack_dir / meta["files"]["true_percussive"])
        leak = metrics.leakage_summary(true_h, true_p, result.harmonic, result.percussive)
        return {
            "entry": name,
            "strength": strength,
            "config": cfg.to_dict(),
            "images": images,
            "stems": {**urls, "mixture": f"/pack/{meta['files']['mixture']}"},
            "diagnostics": {
                "residual_energy_fraction": result.diagnostic["residual_energy_fraction"],
                "recomposition_error_db": result.diagnostic["reconstruction_error_db"],
                "ambiguous_energy_fraction": result.diagnostic["ambiguous_energy_fraction"],
                "harmonic_crosstalk_fraction": leak["harmonic"]["crosstalk_fraction"],
                "percussive_crosstalk_fraction": leak["percussive"]["crosstalk_fraction"],
                "null_test_wav_db": meta["null_test_db"]["wav_roundtrip"],
            },
        }

    def edit(self, name: str, owners: list[str]) -> dict:
        if name not in self.manifest["entries"]:
            raise KeyError(f"unknown entry {name!r}")
        entry = samplepack.build_entry(
            name, sr=self.manifest["sample_rate"], seed=self.manifest["seed"]
        )
        verification = samplepack.verify_edit(entry, owners)
        segs = samplepack.edited_segments(entry, owners)
        eh, ep, emix, _ = samplepack.compose(entry, segs)
        tag = hashlib.sha1(".".join(owners).encode()).hexdigest()[:8]
        key = f"{name}__edit{tag}"
        stems = {"edited_harmonic": eh, "edited_percussive": ep, "edited_mixture": emix}
        urls = {}
        for stem, sig in stems.items():
            hpss_io.write_wav(self.cache_dir / f"{key}__{stem}.wav", sig, entry.sample_rate)
            urls[stem] = f"/cache/{key}__{stem}.wav"
        return {"verification": verification, "stems": urls}


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>HPSS spectral editor</title>
<style>
body { font-family: -apple-system, Segoe UI, sans-serif; margin: 1.5rem auto; max-width: 1200px; padding: 0 1rem; color: #1a202c; }
h1 { font-size: 1.3rem; } h2 { font-size: 1.05rem; border-bottom: 1px solid #cbd5e0; padding-bottom: .2rem; }
.controls { display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; margin: .8rem 0; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: .6rem; }
figure { margin: 0; } figcaption { font-size: .8rem; color: #4a5568; }
img { width: 100%; border: 1px solid #e2e8f0; }
button { padding: .3rem .8rem; } button.active { background: #2b6cb0; color: #fff; }
#metrics { font-family: ui-monospace, monospace; font-size: .82rem; background: #f7fafc; border: 1px solid #e2e8f0; padding: .5rem .7rem; white-space: pre-wrap; }
#feedback { font-family: ui-monospace, monospace; font-size: .82rem; padding: .5rem .7rem; border: 1px solid #e2e8f0; background: #f0fff4; }
#feedback.error { background: #fff5f5; border-color: #fc8181; color: #9b2c2c; }
table { border-collapse: collapse; font-size: .85rem; } td, th { border: 1px solid #e2e8f0; padding: .25rem .5rem; }
</style></head><body>
<h1>HPSS spectral editor &mdash; harmonic / percussive sample pack</h1>
<div class="controls">
  <label>entry <select id="entry"></select></label>
  <label>separation strength <input id="strength" type="range" min="0" max="100" value="50"></label>
  <span id="sval">0.50</span>
  <span id="busy"></span>
</div>
<div class="controls" id="solo">
  solo:
  <button data-stem="mixture">mixture</button>
  <button data-stem="harmonic">harmonic</button>
  <button data-stem="percussive">percussive</button>
  <button data-stem="residual">residual</button>
  <audio id="player" controls></audio>
  <a id="audition" href="/audition/" target="_blank">audition project</a>
</div>
<div id="metrics">kernel idle</div>
<h2>preview</h2>
<div class="grid" id="preview"></div>
<div id="editor"></div>
<h2>kernel feedback</h2>
<div id="feedback">ok</div>
<script>
const $ = (id) => document.getElementById(id);
let current = null, stems = {}, debounce = null;

function feedback(msg, isErr) {
  const el = $("feedback");
  el.textContent = msg;
  el.className = isErr ? "error" : "";
}
async function api(url, opts) {
  const r = await fetch(url, opts);
  const j = await r.json();
  if (!r.ok) { feedback("kernel: " + (j.error || r.statusText), true); throw new Error(j.error); }
  feedback("kernel ok", false);
  return j;
}
async function loadEntries() {
  const j = await api("/api/entries");
  const sel = $("entry");
  sel.innerHTML = "";
  j.entries.forEach(e => {
    const o = document.createElement("option");
    o.value = e.name; o.textContent = e.name;
    sel.appendChild(o);
  });
  sel.onchange = runPreview;
  $("strength").oninput = () => {
    $("sval").textContent = ($("strength").value / 100).toFixed(2);
    clearTimeout(debounce); debounce = setTimeout(runPreview, 250);
  };
  document.querySelectorAll("#solo button").forEach(b => b.onclick = () => {
    const u = stems[b.dataset.stem];
    if (!u) return;
    document.querySelectorAll("#solo button").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    $("player").src = u; $("player").play();
  });
  runPreview();
}
async function runPreview() {
  const name = $("entry").value;
  const s = $("strength").value / 100;
  $("busy").textContent = "separating...";
  try {
    const j = await api(`/api/preview?name=${name}&strength=${s}`);
    current = j; stems = j.stems;
    const d = j.diagnostics, f = (v, n=4) => (typeof v === "number" ? v.toFixed(n) : v);
    $("metrics").textContent =
      `config: kernel=${j.config.kernel_time} mask_power=${j.config.mask_power.toFixed(2)} policy=${j.config.shared_policy}\n` +
      `residual energy fraction : ${f(d.residual_energy_fraction)}\n` +
      `recomposition error      : ${f(d.recomposition_error_db, 1)} dB\n` +
      `ambiguous energy fraction: ${f(d.ambiguous_energy_fraction)}\n` +
      `crosstalk  H<-P / P<-H   : ${f(d.harmonic_crosstalk_fraction)} / ${f(d.percussive_crosstalk_fraction)}\n` +
      `phase-cancel null depth  : ${f(d.null_test_wav_db, 1)} dB (wav round-trip)`;
    $("preview").innerHTML = ["mixture", "harmonic", "percussive", "residual"].map(l =>
      `<figure><img src="${j.images[l]}"><figcaption>${l} layer</figcaption></figure>`).join("");
    buildEditor(name);
  } catch (e) { /* feedback already shown */ }
  $("busy").textContent = "";
}
async function buildEditor(name) {
  const j = await api("/api/entries");
  const meta = j.entries.find(e => e.name === name);
  const ed = $("editor");
  if (!meta.segments.length) { ed.innerHTML = ""; return; }
  ed.innerHTML = `<h2>segment attribution (${meta.assignable_layer} layer)</h2>
    <table><thead><tr><th>segment</th><th>time (s)</th><th>owner</th></tr></thead>
    <tbody>${meta.segments.map(s => `<tr><td>${s.index}</td><td>${s.start_s.toFixed(2)}&ndash;${s.end_s.toFixed(2)}</td>
      <td><select data-seg="${s.index}">
        <option value="percussive"${s.owner === "percussive" ? " selected" : ""}>percussive</option>
        <option value="harmonic"${s.owner === "harmonic" ? " selected" : ""}>harmonic</option>
      </select></td></tr>`).join("")}</tbody></table>
    <div class="controls"><button id="rebuild">rebuild &amp; verify</button>
    <button data-estem="edited_harmonic">solo edited harmonic</button>
    <button data-estem="edited_percussive">solo edited percussive</button>
    <button data-estem="edited_mixture">solo edited mixture</button></div>
    <div id="editmetrics"></div>`;
  ed.querySelectorAll("[data-estem]").forEach(b => b.onclick = () => {
    if (stems[b.dataset.estem]) { $("player").src = stems[b.dataset.estem]; $("player").play(); }
  });
  $("rebuild").onclick = async () => {
    const owners = [...ed.querySelectorAll("[data-seg]")].map(s => s.value);
    try {
      const r = await api(`/api/edit?name=${name}`, {method: "POST",
        headers: {"Content-Type": "application/json"}, body: JSON.stringify({owners})});
      Object.assign(stems, r.stems);
      const v = r.verification, f = (x) => x.toFixed(1);
      $("editmetrics").innerHTML = `<pre>remix error        : ${f(v.remix_error_db)} dB
phase-cancel null  : ${f(v.phase_cancel_null_db)} dB
mask sum max |err| : ${v.mask_sum_max_abs_error}
changed segments   : ${JSON.stringify(v.changed_segments)}
consistent         : ${v.ok}</pre>`;
    } catch (e) { /* feedback already shown */ }
  };
}
loadEntries().catch(e => feedback("kernel: " + e.message, true));
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    kernel: Kernel = None  # set by serve()

    def log_message(self, fmt, *args):  # quieter logs
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- helpers ------------------------------------------------------------
    def _send(self, body: bytes, status: int = 200, ctype: str = "application/json"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200):
        self._send(json.dumps(obj, default=float).encode(), status)

    def _fail(self, status: int, msg: str):
        log_feedback(self.path, msg)
        self._json({"error": msg}, status)

    def _serve_file(self, base: Path, rel: str):
        base = base.resolve()
        target = (base / rel).resolve()
        if not str(target).startswith(str(base)) or not target.is_file():
            return self._fail(404, f"not found: {rel}")
        ctype = MIME.get(target.suffix.lower(), "application/octet-stream")
        self._send(target.read_bytes(), 200, ctype)

    # -- routes -------------------------------------------------------------
    def do_GET(self):
        url = urlparse(self.path)
        path, q = url.path, parse_qs(url.query)
        try:
            if path == "/":
                return self._send(PAGE.encode(), 200, "text/html; charset=utf-8")
            if path == "/api/entries":
                return self._json(self.kernel.entries())
            if path == "/api/preview":
                name = q.get("name", [""])[0]
                strength = float(q.get("strength", ["0.5"])[0])
                return self._json(self.kernel.preview(name, strength))
            if path.startswith("/cache/"):
                return self._serve_file(self.kernel.cache_dir, path[len("/cache/"):])
            if path.startswith("/pack/"):
                return self._serve_file(self.kernel.pack_dir, path[len("/pack/"):])
            if path in ("/audition/", "/audition"):
                return self._serve_file(self.kernel.pack_dir, "audition/index.html")
            # pack-internal relative links (audition page uses ../sources/...)
            top = path.lstrip("/").split("/", 1)[0]
            if top in ("sources", "mixtures", "estimates", "edits", "reference", "evidence"):
                return self._serve_file(self.kernel.pack_dir, path.lstrip("/"))
            return self._fail(404, f"unknown route {path}")
        except Exception as exc:  # kernel failure -> feedback, never hidden
            traceback.print_exc()
            return self._fail(500, f"{type(exc).__name__}: {exc}")

    def do_POST(self):
        url = urlparse(self.path)
        path, q = url.path, parse_qs(url.query)
        try:
            if path == "/api/edit":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                name = q.get("name", [""])[0]
                owners = body.get("owners")
                if not isinstance(owners, list):
                    return self._fail(400, "missing owners list")
                return self._json(self.kernel.edit(name, owners))
            return self._fail(404, f"unknown route {path}")
        except Exception as exc:
            traceback.print_exc()
            return self._fail(500, f"{type(exc).__name__}: {exc}")


def serve(pack_dir: Path, host: str, port: int) -> None:
    kernel = Kernel(pack_dir, CACHE)
    Handler.kernel = kernel
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"HPSS spectral editor on http://{host}:{port}/  (pack: {kernel.pack_dir})")
    print(f"audition project: http://{host}:{port}/audition/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", type=Path, default=PACK_DIR)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    serve(args.pack, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
