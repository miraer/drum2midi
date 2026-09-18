"""Fetches the ADT_STR checkpoint from Hugging Face.

Kept as a script rather than a shell one-liner because the repository holds three
variants (tau 0.4 / 0.6 / 0.8, the CLAP confidence threshold used when curating their
synthetic training data) and only one is wanted, so a blind snapshot download would pull
867 MB instead of 289 MB.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
REPO = "Pierfrancesco/adt-str"

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--tau", default="0.8", choices=["0.4", "0.6", "0.8"],
                help="which curation threshold; their README calls 0.8 the strongest")
ap.add_argument("--out", default=str(ROOT / "ADT_STR" / "checkpoints"))
args = ap.parse_args()

from huggingface_hub import list_repo_files, hf_hub_download

folder = f"setting-tau-{args.tau}"
try:
    files = [f for f in list_repo_files(REPO) if f.startswith(folder)]
except Exception as exc:
    print(f"could not reach {REPO}: {exc}")
    raise SystemExit(1)

if not files:
    print(f"no files under {folder} in {REPO}")
    raise SystemExit(1)

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
print(f"{len(files)} files in {folder}")
for name in files:
    path = hf_hub_download(repo_id=REPO, filename=name, local_dir=str(out))
    size = Path(path).stat().st_size
    print(f"  {name}  ({size/1024/1024:.1f} MB)")
print(f"\ndownloaded to {out / folder}")
