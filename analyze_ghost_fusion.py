"""Do stems recover the ghost notes the mix pass cannot see.

A threshold does not pull them out (quiet-hit recall 0.333 -> 0.409 at the cost of
precision 0.970 -> 0.762). The question here is whether the ADTOF activation fires on a
quiet hit when it is fed the isolated stem instead.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
GMD = ROOT / "gmd" / "groove"
MIX_CACHE = ROOT / "bench" / "gmd_activations"
STEM_CACHE = ROOT / "bench" / "gmd_stem_activations"
LARSNET_DIR = ROOT / "larsnet"
SR = 44100
WINDOW = 0.05
BINS = [(1, 20), (20, 35), (35, 50), (50, 70), (70, 128)]
CLASSES = [(0, "kick", (35, 36)), (1, "snare", (38, 40, 37)),
           (3, "hihat", (42, 22, 46, 26, 44))]
STOCK = [0.22, 0.14, 0.32, 0.22, 0.30]


@contextlib.contextmanager
def _in_dir(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def pick(col, thr):
    from adtof_pytorch.post_processing import NotePeakPickingProcessor
    p = NotePeakPickingProcessor(threshold=float(thr), pre_avg=0.1, post_avg=0.01,
                                 pre_max=0.02, post_max=0.01, combine=0.02, fps=100)
    return np.array([t for t, _ in p.process(col)])


def match(ref, est):
    found = np.zeros(len(ref), dtype=bool)
    used = set()
    for i, t in enumerate(ref):
        best, dist = None, WINDOW
        for j, e in enumerate(est):
            if j in used:
                continue
            d = abs(e - t)
            if d <= dist:
                best, dist = j, d
        if best is not None:
            used.add(best)
            found[i] = True
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=14)
    args = ap.parse_args()

    import torch
    from adtof_pytorch import load_audio_for_model
    sys.path.insert(0, str(ROOT))
    from drum2midi import _adtof_model
    model = _adtof_model("cpu")

    rows = [r for r in csv.DictReader(open(GMD / "info.csv", encoding="utf-8"))
            if r.get("audio_filename") and (GMD / r["audio_filename"]).exists()
            and 5 <= float(r["duration"]) <= 60]
    rows.sort(key=lambda r: (r["split"] != "test", r["id"]))
    rows = rows[: args.limit]
    STEM_CACHE.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(LARSNET_DIR))
    with _in_dir(LARSNET_DIR):
        from larsnet import LarsNet
        net = None
        data = []
        for i, r in enumerate(rows, 1):
            key = r["id"].replace("/", "_")
            mix_act = np.load(MIX_CACHE / f"{key}.npy") if (MIX_CACHE / f"{key}.npy").exists() else None
            if mix_act is None:
                with torch.no_grad():
                    mix_act = model(load_audio_for_model(
                        str(GMD / r["audio_filename"]))).cpu().numpy()[0]
                np.save(MIX_CACHE / f"{key}.npy", mix_act)

            stem_file = STEM_CACHE / f"{key}.npz"
            if stem_file.exists():
                stem_act = {k: v for k, v in np.load(stem_file).items()}
            else:
                if net is None:
                    net = LarsNet(wiener_filter=True, wiener_exponent=1.0,
                                  device="cpu", config="config.yaml")
                audio, sr = sf.read(str(GMD / r["audio_filename"]), always_2d=True)
                audio = audio.T.astype(np.float32)
                if audio.shape[0] == 1:
                    audio = np.vstack([audio, audio])
                with torch.no_grad():
                    stems = {k: v.cpu().numpy()
                             for k, v in net.separate_wiener(torch.from_numpy(audio)).items()}
                tmp = Path(tempfile.mkdtemp(prefix="gst_"))
                stem_act = {}
                try:
                    for _, name, _ in CLASSES:
                        s = stems.get(name)
                        if s is None:
                            continue
                        p = tmp / f"{name}.wav"
                        sf.write(str(p), s.T, SR)
                        with torch.no_grad():
                            stem_act[name] = model(
                                load_audio_for_model(str(p))).cpu().numpy()[0]
                finally:
                    import shutil
                    shutil.rmtree(tmp, ignore_errors=True)
                np.savez_compressed(stem_file, **stem_act)

            pm = pretty_midi.PrettyMIDI(str(GMD / r["midi_filename"]))
            notes = [(float(n.start), n.pitch, int(n.velocity))
                     for inst in pm.instruments for n in inst.notes]
            data.append((mix_act, stem_act, notes))
            print(f"  [{i:2d}/{len(rows)}] {r['id']}")

    sys.path.remove(str(LARSNET_DIR))

    print(f"\n{'='*78}\nRECALL BY HIT STRENGTH: mix vs mix+stem\n{'='*78}")
    for ci, name, pitches in CLASSES:
        rows_out = {"mix": [[0, 0] for _ in BINS], "+stem": [[0, 0] for _ in BINS]}
        extra = 0
        for mix_act, stem_act, notes in data:
            ref = sorted((t, v) for t, p, v in notes if p in pitches)
            if not ref:
                continue
            times = np.array([t for t, _ in ref])
            vels = np.array([v for _, v in ref])
            o_mix = pick(mix_act[:, ci], STOCK[ci])
            sa = stem_act.get(name)
            if sa is not None:
                o_stem = pick(sa.max(axis=1), STOCK[ci])
                add = [t for t in o_stem
                       if not len(o_mix) or np.min(np.abs(o_mix - t)) > 0.05]
                extra += len(add)
                o_both = np.array(sorted(list(o_mix) + add))
            else:
                o_both = o_mix
            for label, est in (("mix", o_mix), ("+stem", o_both)):
                found = match(times, est)
                for bi, (lo, hi) in enumerate(BINS):
                    m = (vels >= lo) & (vels < hi)
                    rows_out[label][bi][0] += int(found[m].sum())
                    rows_out[label][bi][1] += int(m.sum())
        print(f"\n{name}  (onsets added from the stem: {extra})")
        print(f"{'variant':<9}" + "".join(f"{f'{lo}-{hi-1}':>11}" for lo, hi in BINS))
        for label in ("mix", "+stem"):
            cells = []
            for got, tot in rows_out[label]:
                cells.append(f"{got/tot:>10.2f} " if tot >= 20 else f"{'—':>11}")
            print(f"{label:<9}" + "".join(cells))
        d = []
        for bi in range(len(BINS)):
            a, ta = rows_out["mix"][bi]
            b, tb = rows_out["+stem"][bi]
            d.append(f"{(b/tb - a/ta):>+10.2f} " if ta >= 20 else f"{'—':>11}")
        print(f"{'delta':<9}" + "".join(d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
