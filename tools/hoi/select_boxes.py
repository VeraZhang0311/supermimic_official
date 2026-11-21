"""Pick the ULTRA box clips for training from a QA report.

Drops clips the QA flagged (``qa_hoi_dataset.py``), then fills each box type up
to a frame budget: the mean frames per type of the report's HOI-Retarget
(user-curated) variants, unless ``--frames`` is given. Clips are taken in turns
over source motions (the ULTRA augmentations of one capture share a source) so
every source is used before any gets a second augmentation; within a source the
cleanest augmentation (lowest reference penetration) goes first.

``--exclude`` takes held-out clips (``select_eval.py``): every augmentation of a
held-out box source is dropped, and held-out curated clips leave the budget.

  python tools/hoi/select_boxes.py <QA'd library or qa.json> --out boxes.txt \
      [--exclude eval.txt] [--frames N]

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
  ap.add_argument("qa", help="QA'd library dir (reads hoi/qa.json) or a qa.json file")
  ap.add_argument("--out", required=True)
  ap.add_argument("--exclude", default=None, help="held-out clips, one dataset:clip_id per line")
  ap.add_argument("--frames", type=int, default=0, help="frames per box type (0: curated-type mean)")
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()

  path = Path(args.qa)
  rows = json.loads((path / "hoi" / "qa.json" if path.is_dir() else path).read_text())["clips"]
  held = set()
  if args.exclude:
    held = {line.strip() for line in Path(args.exclude).read_text().splitlines() if line.strip()}
  held_sources = {source_of(n.split(":", 1)[1]) for n in held if n.startswith("ultra:")}

  def base(name: str) -> str:
    return name.split("#", 1)[0]  # a trimmed piece belongs to its clip

  frames_by_variant = collections.Counter()
  for r in rows:
    if not r["variant"].endswith("_ultra") and base(r["name"]) not in held:
      frames_by_variant[r["variant"]] += round(r["seconds"] / DT)
  curated = list(frames_by_variant.values())
  budget = args.frames or round(sum(curated) / len(curated))
  print(f"budget {budget} frames ({budget * DT / 60:.0f} min) per box type; "
        f"{len(held_sources)} held-out box sources, {len(held) - len(held_sources)} held-out other clips")

  rng = random.Random(args.seed)
  lines = []
  for box in BOXES:
    pool = [r for r in rows if r["name"].startswith("ultra:") and r["variant"] == f"{box}_ultra"]
    kept = [r for r in pool if not r["flags"] and source_of(r["name"][6:]) not in held_sources]
    by_source = collections.defaultdict(list)
    for r in kept:
      by_source[source_of(r["name"][6:])].append(r)
    for augs in by_source.values():
      augs.sort(key=lambda r: (r["penetration_p90_m"], r["name"]))
    order = sorted(by_source)
    rng.shuffle(order)
    picked, total = [], 0
    for k in range(max(len(a) for a in by_source.values())):
      for s in order:
        if total >= budget:
          break
        if k < len(by_source[s]):
          picked.append(by_source[s][k])
          total += round(by_source[s][k]["seconds"] / DT)
    lines += sorted(r["name"] for r in picked)
    sources = {source_of(r["name"][6:]) for r in picked}
    print(
      f"{box:10s} pool {len(pool):4d} usable {len(kept):4d} "
      f"-> picked {len(picked):4d} clips, {total} frames, {len(sources)}/{len(by_source)} sources"
    )
  Path(args.out).write_text("\n".join(lines) + "\n")
  print(f"wrote {len(lines)} clips to {args.out}")


if __name__ == "__main__":
  main()
