"""Measure how separable pedal hi-hats really are, using GMD ground truth.

For every annotated hi-hat hit we pull features out of the separated hi-hat stem and
group them by the articulation the drum module recorded. If pedal chicks do not form
a distinct cluster, no threshold will ever find them and the feature set must change.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
GMD = ROOT / "gmd" / "groove"
SR = 44100
sys.path.insert(0, str(ROOT))
from drum2midi import _centroid, _mono, peak_at  # noqa: E402

ARTIC = {42: "closed", 22: "closed", 46: "open", 26: "open", 44: "pedal"}


def features(sig: np.ndarray, t: float) -> dict:
    a = sig[int(t * SR):min(int((t + 0.05) * SR), len(sig))]
    b = sig[min(int((t + 0.08) * SR), len(sig)):min(int((t + 0.25) * SR), len(sig))]
    att = float(np.sqrt(np.mean(a ** 2))) if a.size else 0.0
    tail = float(np.sqrt(np.mean(b ** 2))) if b.size else 0.0
    return {"peak": peak_at(sig, t), "centroid": _centroid(sig, t),
            "attack": att, "ratio": tail / (att + 1e-9)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--separator", default="larsnet")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(GMD / "info.csv", encoding="utf-8"))
            if r.get("audio_filename") and (GMD / r["audio_filename"]).exists()
            and 8 <= float(r["duration"]) <= 45]
    rows.sort(key=lambda r: (r["split"] != "test", r["id"]))
    rows = rows[: args.limit]

    work = ROOT / "bench" / "pedal"
    work.mkdir(parents=True, exist_ok=True)
    data = {k: [] for k in ("closed", "open", "pedal")}

    for i, r in enumerate(rows, 1):
        tag = r["id"].replace("/", "_")
        stem_dir = work / tag
        if not (stem_dir / "hihat.wav").exists():
            subprocess.run(
                [sys.executable, str(ROOT / "drum2midi.py"), str(GMD / r["audio_filename"]),
                 "-o", str(work / f"{tag}.mid"), "--device", "auto",
                 "--separator", args.separator, "--keep-stems", str(stem_dir)],
                capture_output=True, text=True, encoding="utf-8", errors="replace")
        if not (stem_dir / "hihat.wav").exists():
            print(f"  [{i}] no hihat stem for {r['id']}")
            continue
        sig = _mono(np.asarray(sf.read(str(stem_dir / "hihat.wav"))[0]).T)
        pm = pretty_midi.PrettyMIDI(str(GMD / r["midi_filename"]))
        n = 0
        for inst in pm.instruments:
            for note in inst.notes:
                kind = ARTIC.get(note.pitch)
                if kind:
                    data[kind].append(features(sig, float(note.start)))
                    n += 1
        print(f"  [{i}/{len(rows)}] {r['id']}: {n} hat hits")

    print(f"\n{'articulation':<14}{'n':>6}{'peak p50':>10}{'centroid p50':>14}"
          f"{'attack p50':>12}{'ratio p50':>11}")
    print("-" * 67)
    for kind in ("closed", "open", "pedal"):
        d = data[kind]
        if not d:
            print(f"{kind:<14}{0:>6}")
            continue
        print(f"{kind:<14}{len(d):>6}"
              f"{np.median([x['peak'] for x in d]):>10.4f}"
              f"{np.median([x['centroid'] for x in d]):>14.0f}"
              f"{np.median([x['attack'] for x in d]):>12.4f}"
              f"{np.median([x['ratio'] for x in d]):>11.3f}")

    if data["pedal"] and data["closed"]:
        print("\nSeparability of pedal vs closed (how much the distributions overlap):")
        for feat in ("peak", "centroid", "attack", "ratio"):
            p = np.array([x[feat] for x in data["pedal"]])
            c = np.array([x[feat] for x in data["closed"]])
            pooled = np.sqrt((p.var() + c.var()) / 2) + 1e-12
            d_prime = abs(p.mean() - c.mean()) / pooled
            # best achievable accuracy with a single cut on this feature
            cuts = np.percentile(np.concatenate([p, c]), np.linspace(1, 99, 99))
            best = max(max((p < t).mean() * 0.5 + (c >= t).mean() * 0.5,
                           (p > t).mean() * 0.5 + (c <= t).mean() * 0.5) for t in cuts)
            print(f"  {feat:<10} d'={d_prime:5.2f}   best single-threshold balanced acc = {best:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
