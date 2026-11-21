"""Pick the held-out evaluation clips: per object type, hand-contact and
foot-contact clips from distinct source motions, deterministically.

A source motion is one capture: an HOI-Retarget clip, or the ULTRA augmentations
that share ``sub<N>_<object>_<seq>``. Every augmentation of a held-out source
must stay out of training (``select_boxes.py --exclude``, ``build_hoi_dataset.py
--exclude``). Candidates skip clips the QA flagged and trimmed clips.

  hand clip: >= --min-hand frames with a hand labelled in contact and
             <= --max-foot-for-hand frames with a foot in contact
  foot clip: >= --min-foot frames with a foot in contact (and, for
             HOI-Retarget, tagged "kick" in the user's table)

  python tools/hoi/select_eval.py --index included.jsonl --table chairs.xlsx \
      --table tables.xlsx --clips ultra_boxes_100.txt --qa qa.json --out eval.txt

Writes ``dataset:clip_id`` lines and ``<out>.json`` with each clip's kind,
source and contact counts.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_hoi_dataset as b  # noqa: E402


def source_of(dataset: str, clip_id: str) -> str:
  if dataset == "ultra":
    m = re.match(r"(sub\d+_[a-z]+_\d{3})_", clip_id)
    if m:
      return f"ultra:{m.group(1)}"
  return f"{dataset}:{clip_id}"


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--index", required=True)
  ap.add_argument("--table", action="append", default=[])
  ap.add_argument("--table-dataset", default="hoi_retarget")
  ap.add_argument("--clips", action="append", default=[])
  ap.add_argument("--qa", action="append", required=True,
                  help="qa.json covering the candidates (flags drop clips); repeat to merge, later files win")
  ap.add_argument("--per-type", type=int, default=12)
  ap.add_argument("--foot", type=int, default=3, help="foot-contact clips per type; the rest are hand clips")
  ap.add_argument("--min-hand", type=int, default=50)
  ap.add_argument("--max-foot-for-hand", type=int, default=5)
  ap.add_argument("--min-foot", type=int, default=25)
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  qa = {r["name"]: r for path in args.qa for r in json.loads(Path(path).read_text())["clips"]}
  index = [json.loads(line) for line in Path(args.index).read_text().splitlines() if line]
  entries = [e for t in args.table for e in b.entries_from_table(t, args.table_dataset, index)]
  entries += [e for c in args.clips for e in b.entries_from_clip_list(c, index)]

  # source -> candidate clips, per type and kind
  cands: dict[str, dict[str, dict[str, list[dict]]]] = collections.defaultdict(
    lambda: {"hand": collections.defaultdict(list), "foot": collections.defaultdict(list)}
  )
  for e in entries:
    if e.get("trim"):
      continue
    name = f"{e['dataset']}:{e['clip_id']}"
    row = qa.get(name)
    if row is None or row["flags"]:
      continue
    pieces = b.load_clip(e)
    if len(pieces) != 1:
      continue
    p = pieces[0]
    hand = int((p["hand_contact"] == 1).any(-1).sum())
    foot = int((p["foot_contact"] == 1).any(-1).sum())
    rec = {"name": name, "type": p["type"], "source": source_of(e["dataset"], e["clip_id"]),
           "frames": len(p["joint_pos"]), "hand_frames": hand, "foot_frames": foot,
           "categories": e.get("categories", [])}
    kicks = e["dataset"] != args.table_dataset or "kick" in rec["categories"]
    if foot >= args.min_foot and kicks:
      cands[p["type"]]["foot"][rec["source"]].append(rec)
    elif hand >= args.min_hand and foot <= args.max_foot_for_hand:
      cands[p["type"]]["hand"][rec["source"]].append(rec)

  rng = random.Random(args.seed)
  picked: list[dict] = []
  for t in b.OBJECT_TYPES:
    if t not in cands:
      continue
    used: set[str] = set()
    want = {"foot": args.foot, "hand": args.per_type - args.foot}
    for kind in ("foot", "hand"):
      sources = sorted(s for s in cands[t][kind] if s not in used)
      rng.shuffle(sources)
      take = sources[: want[kind]]
      if len(take) < want[kind]:
        print(f"[WARN] {t}: only {len(take)} {kind} sources for {want[kind]} slots")
      for s in take:
        clip = rng.choice(sorted(cands[t][kind][s], key=lambda r: r["name"]))
        picked.append({**clip, "kind": kind})
        used.add(s)
    n = {k: sum(r["type"] == t and r["kind"] == k for r in picked) for k in ("hand", "foot")}
    avail = {k: len(cands[t][k]) for k in ("hand", "foot")}
    print(f"{t:11s} picked hand {n['hand']} foot {n['foot']}  (candidate sources hand {avail['hand']}, foot {avail['foot']})")

  out = Path(args.out)
  out.write_text("".join(f"{r['name']}\n" for r in picked))
  out.with_suffix(".json").write_text(json.dumps(
    {"seed": args.seed, "rules": {k: getattr(args, k) for k in ("per_type", "foot", "min_hand", "max_foot_for_hand", "min_foot")},
     "clips": picked}, indent=1))
  print(f"wrote {len(picked)} clips to {out} ({len({r['source'] for r in picked})} source motions)")


if __name__ == "__main__":
  main()
