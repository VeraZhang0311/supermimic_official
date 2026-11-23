"""tools/hoi/combine_libraries.py: concatenating packed HOI libraries and dropping clips."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.hoi_fixtures import CLIP_LENGTHS, CLIP_VARIANT, make_library

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "hoi"))
import combine_libraries  # noqa: E402


def library(root: Path, prefix: str, offset: float) -> Path:
  """The fixture library plus the files a packed library also holds, with its own clip names."""
  make_library(root)
  names = [f"{prefix}{i}" for i in range(len(CLIP_LENGTHS))]
  (root / "motion_names.json").write_text(json.dumps(names))
  (root / "joint_names.json").write_text(json.dumps(["j0", "j1"]))
  (root / "hoi" / "clips.json").write_text(json.dumps([{"name": n, "frames": f} for n, f in zip(names, CLIP_LENGTHS)]))
  total = sum(CLIP_LENGTHS)
  np.save(root / "root_pos_w.npy", offset + np.arange(total * 3, dtype=np.float32).reshape(total, 3))
  return root


def run(monkeypatch, *argv: str) -> None:
  monkeypatch.setattr(sys, "argv", ["combine_libraries.py", *argv])
  assert combine_libraries.main() == 0


def test_combine_with_splits_and_exclusion(tmp_path, monkeypatch):
  a, b = library(tmp_path / "a", "a", 0.0), library(tmp_path / "b", "b", 1e4)
  drop = tmp_path / "drop.txt"
  drop.write_text("# not trained\na1  # a comment\n")
  out = tmp_path / "out"
  run(monkeypatch, "--lib", str(a), "--split", "train", "--lib", str(b), "--split", "eval", "--exclude", str(drop),
      "--out", str(out))
  names = json.loads((out / "motion_names.json").read_text())
  assert names == ["a0", "a2", "b0", "b1", "b2"]
  lengths = [CLIP_LENGTHS[0], CLIP_LENGTHS[2], *CLIP_LENGTHS]
  ends = np.load(out / "motion_ends.npy")
  assert ends.tolist() == np.cumsum(lengths).tolist()
  assert np.load(out / "hoi" / "clip_variant.npy").tolist() == [CLIP_VARIANT[0], CLIP_VARIANT[2], *CLIP_VARIANT]
  # Every clip's frames are its source's frames.
  root = np.load(out / "root_pos_w.npy")
  src_starts = np.r_[0, np.cumsum(CLIP_LENGTHS)[:-1]]
  sources = [(a, 0), (a, 2), (b, 0), (b, 1), (b, 2)]
  for k, (lib, i) in enumerate(sources):
    s = 0 if k == 0 else ends[k - 1]
    src = np.load(lib / "root_pos_w.npy")[src_starts[i] : src_starts[i] + CLIP_LENGTHS[i]]
    np.testing.assert_array_equal(root[s : ends[k]], src)
    np.testing.assert_array_equal(
      np.load(out / "hoi" / "object_pos_w.npy")[s : ends[k]],
      np.load(lib / "hoi" / "object_pos_w.npy")[src_starts[i] : src_starts[i] + CLIP_LENGTHS[i]],
    )
  clips = json.loads((out / "hoi" / "clips.json").read_text())
  assert [c["split"] for c in clips] == ["train", "train", "eval", "eval", "eval"]
  variants = json.loads((out / "hoi" / "variants.json").read_text())["variants"]
  assert [v["clips"] for v in variants] == [4, 1]
  assert sum(v["frames"] for v in variants) == ends[-1]
  assert (out / "hoi" / "objects" / "largebox" / "sdf.npz").exists()
  manifest = json.loads((out / "build_manifest.json").read_text())
  assert manifest["excluded"] == ["a1"] and manifest["clips"] == 5


def test_piece_names_are_not_comments(tmp_path):
  path = tmp_path / "names.txt"
  path.write_text("# header\nclip#0\nother  # why\n\n")
  assert combine_libraries.read_names(path) == ["clip#0", "other"]


def test_refuses_other_geometry_and_unknown_names(tmp_path, monkeypatch):
  a, b = library(tmp_path / "a", "a", 0.0), library(tmp_path / "b", "b", 1e4)
  path = b / "hoi" / "variants.json"
  variants = json.loads(path.read_text())
  variants["variants"][0]["geometry_sha256"] = "other"
  path.write_text(json.dumps(variants))
  monkeypatch.setattr(sys, "argv", ["x", "--lib", str(a), "--lib", str(b), "--out", str(tmp_path / "o1")])
  with pytest.raises(SystemExit, match="other object variants"):
    combine_libraries.main()
  drop = tmp_path / "drop.txt"
  drop.write_text("a9\n")
  monkeypatch.setattr(sys, "argv", ["x", "--lib", str(a), "--exclude", str(drop), "--out", str(tmp_path / "o2")])
  with pytest.raises(SystemExit, match="names no clip"):
    combine_libraries.main()
