"""Runs ADT_STR on our benchmark, so its claims can be checked on our numbers.

ADT_STR (arXiv 2601.09520, weights published 2026-09-17) is the first ADT model in years
that is both newer than ADTOF and licensed permissively (CC BY-SA 4.0 rather than
ADTOF's CC BY-NC-SA). Its reported tom F1 on MDB Drums is 0.77 against our 0.605, which
is our weakest class by a wide margin -- but their number is an 8-class drum-only
evaluation and ours is 5-class, so the two are not comparable as published. The only way
to know is to run it on the same files and score it with the same code.

The checkpoint on Hugging Face ships a transformers-style config.json, while their
build_model() expects an experiment YAML with an `inference.checkpoint_path` section.
This writes that YAML pointing at the downloaded weights.

    python run_adt_str.py --limit 3          # try a few tracks first
    python run_adt_str.py                    # all of MDB Drums
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
ADT = ROOT / "ADT_STR"
MDB_AUDIO = ROOT / "mdbdrums" / "MDB Drums" / "audio" / "drum_only"
OUT = ROOT / "bench" / "adtstr_midi"


def write_config(checkpoint: Path) -> Path:
    """Their YAML, with the checkpoint path pointed at our download."""
    import yaml

    src = checkpoint / "adt_config.yaml"
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    cfg.setdefault("inference", {})["checkpoint_path"] = str(checkpoint)
    dest = ROOT / "bench" / "adt_str_local.yaml"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tau", default="0.8", choices=["0.4", "0.6", "0.8"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cpu",
                    help="their code selects its own device; cpu is the safe default")
    args = ap.parse_args()

    checkpoint = ADT / "checkpoints" / f"setting-tau-{args.tau}"
    if not (checkpoint / "model.safetensors").exists():
        print(f"weights missing: run  python fetch_adt_str.py --tau {args.tau}")
        return 1
    if not MDB_AUDIO.exists():
        print(f"MDB Drums not found at {MDB_AUDIO}")
        return 1

    config = write_config(checkpoint)
    tracks = sorted(MDB_AUDIO.glob("*.wav"))[: args.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(tracks)} tracks -> {OUT}")

    t0 = time.time()
    failed = 0
    for i, wav in enumerate(tracks, 1):
        dest = OUT / wav.stem
        if list(dest.glob("*.mid")):
            continue
        dest.mkdir(parents=True, exist_ok=True)
        res = subprocess.run(
            [sys.executable, "-c",
             "import torchaudio_shim; "
             "import sys; sys.argv[0] = 'inference.py'; "
             "exec(open('inference.py', encoding='utf-8').read())",
             "--input", str(wav), "--config", str(config), "-o", str(dest)],
            cwd=str(ADT), capture_output=True, text=True,
            encoding="utf-8", errors="replace")
        if res.returncode != 0 or not list(dest.glob("*.mid")):
            failed += 1
            tail = (res.stderr or res.stdout or "").strip().splitlines()
            print(f"  ! {wav.stem}: {tail[-1][:110] if tail else 'no output'}")
        if i % 5 == 0 or i == len(tracks):
            el = time.time() - t0
            print(f"  [{i}/{len(tracks)}] {el/60:.1f} min, {failed} failed", flush=True)

    made = sum(1 for _ in OUT.rglob("*.mid"))
    print(f"\n{made} MIDI files produced in {(time.time()-t0)/60:.1f} min")
    if made:
        print("Score them with:  python score_adt_str.py")
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
