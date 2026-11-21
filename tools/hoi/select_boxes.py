"""Pick the ULTRA box clips for training from a QA'd library.

Drops clips the QA flagged (``qa_hoi_dataset.py``), then fills each box type up
to a frame budget: the mean frames per type of the library's HOI-Retarget
(user-curated) variants, unless ``--frames`` is given. Clips are taken in turns
over source motions (the ULTRA augmentations of one capture share a source) so
every source is used before any gets a second augmentation; within a source the
cleanest augmentation (lowest reference penetration) goes first.

  python tools/hoi/select_boxes.py <QA'd library> --out boxes.txt [--frames N]

Writes ``ultra:<clip_id>`` lines for ``build_hoi_dataset.py --clips``.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

BOXES = ("largebox", "smallbox", "plasticbox", "suitcase")
DT = 0.02


def source_of(clip: str) -> str:
  m = re.match(r"(sub\d+_[a-z]+_\d{3})_", clip)
  if not m:
    raise ValueError(f"not an ULTRA box clip: {clip}")
  return m.group(1)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("library")
  ap.add_argument("--out", required=True)
  ap.add_argument("--frames", type=int, default=0, help="frames per box type (0: curated-type mean)")
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()

  h = Path(args.library) / "hoi"
  rows = json.loads((h / "qa.json").read_text())["clips"]
  clips = json.loads((h / "clips.json").read_text())
  if len(rows) != len(clips):
    raise ValueError("qa.json and clips.json disagree on clip count; rerun qa_hoi_dataset.py")
  qa = {c["name"]: r for c, r in zip(clips, rows, strict=True)}  # both in motion order
  frames_by_variant = collections.Counter()
  for c in clips:
    frames_by_variant[qa[c["name"]]["variant"]] += c["frames"]
  curated = [n for v, n in frames_by_variant.items() if not v.endswith("_ultra")]
  budget = args.frames or round(sum(curated) / len(curated))
  print(f"budget {budget} frames ({budget * DT / 60:.0f} min) per box type")

  rng = random.Random(args.seed)
  lines = []
  for box in BOXES:
    pool = [c for c in clips if c["dataset"] == "ultra" and qa[c["name"]]["variant"] == f"{box}_ultra"]
    kept = [c for c in pool if not qa[c["name"]]["flags"]]
    by_source = collections.defaultdict(list)
    for c in kept:
      by_source[source_of(c["name"])].append(c)
    for augs in by_source.values():
      augs.sort(key=lambda c: (qa[c["name"]]["penetration_p90_m"], c["name"]))
    order = sorted(by_source)
    rng.shuffle(order)
    picked, total = [], 0
    for k in range(max(len(a) for a in by_source.values())):
      for s in order:
        if total >= budget:
          break
        if k < len(by_source[s]):
          picked.append(by_source[s][k])
          total += by_source[s][k]["frames"]
    lines += [f"ultra:{c['source_clip']}" for c in sorted(picked, key=lambda c: c["name"])]
    sources = {source_of(c["name"]) for c in picked}
    print(
      f"{box:10s} pool {len(pool):4d} flagged {len(pool) - len(kept):4d} kept {len(kept):4d} "
      f"-> picked {len(picked):4d} clips, {total} frames, {len(sources)}/{len(by_source)} sources"
    )
  Path(args.out).write_text("\n".join(lines) + "\n")
  print(f"wrote {len(lines)} clips to {args.out}")


if __name__ == "__main__":
  main()
