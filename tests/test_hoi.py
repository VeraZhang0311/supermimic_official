"""HOI library, geometry, reward/termination math and representation."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from supermimic.hoi import terms
from supermimic.hoi.geometry import HAND_FEATURE_DIM, object_clearance, sdf_query
from supermimic.hoi.library import HoiLibrary, assign_world_variants, read_variants
from tests.hoi_fixtures import BOX_HALF, CLIP_LENGTHS, CLIP_VARIANT, UNSAFE, box_sdf, make_library


@pytest.fixture(scope="module")
def lib(tmp_path_factory) -> HoiLibrary:
  root = make_library(tmp_path_factory.mktemp("hoi_lib"))
  return HoiLibrary(root, "cpu", bin_seconds=1.0, dt=0.02)


class TestLibrary:
  def test_bins_stay_inside_their_clip(self, lib: HoiLibrary):
    ends = np.cumsum(CLIP_LENGTHS)
    starts = np.r_[0, ends[:-1]]
    for b in range(lib.bin_count):
      lo, hi = int(lib.bin_start[b]), int(lib.bin_end[b])
      clip = int(np.searchsorted(ends, lo, side="right"))
      assert starts[clip] <= lo < hi <= ends[clip] - 1 + 1
      assert hi - lo <= 50
      assert int(lib.bin_variant[b]) == CLIP_VARIANT[clip]

  def test_frame_bin_lut_matches_bins(self, lib: HoiLibrary):
    for b in range(lib.bin_count):
      lo, hi = int(lib.bin_start[b]), int(lib.bin_end[b])
      assert (lib.frame_bin[lo:hi] == b).all()
    assert (lib.frame_bin >= 0).all() and (lib.frame_bin < lib.bin_count).all()

  def test_restricted_frames_are_never_reset_starts(self, tmp_path):
    lib = HoiLibrary(make_library(tmp_path / "lib"), "cpu", bin_seconds=1.0, dt=0.02)
    ok = np.ones(sum(CLIP_LENGTHS), bool)
    ok[UNSAFE] = False
    lib.restrict_starts(ok)
    starts = torch.zeros(lib.frame_bin.numel(), dtype=torch.bool)
    for b in range(lib.bin_count):
      starts[int(lib.bin_start[b]) : int(lib.bin_end[b])] = True
    assert not starts[UNSAFE].any()
    assert starts[: UNSAFE.start].all() and starts[UNSAFE.stop : CLIP_LENGTHS[0] - 1].all()
    assert (lib.frame_bin >= 0).all() and (lib.frame_bin < lib.bin_count).all()

  def test_world_assignment(self, lib: HoiLibrary):
    variants = read_variants(lib.root)
    a = assign_world_variants(variants, 100, min_worlds=16)
    assert len(a) == 100
    counts = np.bincount(a, minlength=2)
    assert counts.min() >= 16
    frames = np.asarray([v.frames for v in variants])
    assert counts[0] > counts[1] if frames[0] > frames[1] else counts[0] <= counts[1]


class TestGeometry:
  def test_sdf_matches_analytic_box(self, lib: HoiLibrary):
    rng = np.random.default_rng(1)
    pts = rng.uniform(-0.3, 0.3, size=(1, 200, 3))
    phi, normal = sdf_query(lib, torch.tensor([0]), torch.tensor(pts, dtype=torch.float32))
    ref = box_sdf(pts[0], np.asarray(BOX_HALF["largebox"]))
    assert np.abs(phi[0].numpy() - ref).max() < 0.012
    assert torch.allclose(normal.norm(dim=-1), torch.ones(1, 200), atol=1e-3)

  def test_sdf_beyond_grid_adds_distance(self, lib: HoiLibrary):
    far = torch.tensor([[[2.0, 0.0, 0.0]]])
    phi, _ = sdf_query(lib, torch.tensor([0]), far)
    assert abs(float(phi) - (2.0 - BOX_HALF["largebox"][0])) < 0.02

  def test_sdf_is_differentiable(self, lib: HoiLibrary):
    p = torch.tensor([[[0.25, 0.01, 0.0]]], requires_grad=True)
    phi, _ = sdf_query(lib, torch.tensor([0]), p)
    phi.sum().backward()
    assert p.grad is not None and abs(float(p.grad[0, 0, 0]) - 1.0) < 0.05

  def test_clearance_of_resting_box(self, lib: HoiLibrary):
    half = BOX_HALF["smallbox"]
    pos = torch.tensor([[0.0, 0.0, half[2]]])
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert abs(float(object_clearance(lib, torch.tensor([1]), pos, quat))) < 1e-5


class TestTerms:
  cfg = terms.HoiTaskCfg()

  def _state(self, n: int = 4):
    pos = torch.randn(n, 3)
    quat = torch.nn.functional.normalize(torch.randn(n, 4), dim=-1)
    return pos, quat, torch.randn(n, 3), torch.randn(n, 3)

  def test_perfect_tracking_scores_one(self):
    pos, quat, lin, ang = self._state()
    hands = torch.randn(4, 2, 3)
    gap = torch.zeros(4, 2, 3)
    contact = torch.ones(4, 2, dtype=torch.long)
    r = terms.hoi_reward_terms(self.cfg, pos, quat, lin, ang, pos, quat, lin, ang, hands, hands, gap, contact)
    for k in ("object_pos", "object_ori", "object_lin_vel", "object_ang_vel", "hand_object", "contact"):
      assert torch.allclose(r[k], torch.ones(4), atol=1e-4), k
    assert torch.allclose(r["penetration"], torch.zeros(4))

  def test_contact_reward_ignores_unlabelled_hands(self):
    gap = torch.full((1, 2, 3), 0.5)
    assert float(terms.contact_reward(self.cfg, gap, torch.tensor([[0, -1]]))) == 0.0

  def test_penetration_sums_negative_gaps(self):
    gap = torch.tensor([[[0.01, -0.02, 0.0], [-0.01, 0.03, 0.02]]])
    assert math.isclose(float(terms.penetration(gap)), 0.03, rel_tol=1e-5)

  def test_object_termination_thresholds(self):
    anchor = torch.zeros(2, 3)
    q = torch.tensor([[1.0, 0, 0, 0]] * 2)
    obj = torch.tensor([[0.3, 0.0, 0.0], [0.3, 0.0, 0.0]])
    ref = torch.zeros(2, 3)
    contact = torch.tensor([[1, 0], [0, 0]])
    err = terms.object_termination_errors(self.cfg, anchor, q, obj, q, anchor, q, ref, q, contact)
    assert float(err[0, 0]) > 1.0  # held: 0.3 m > 0.25 m
    assert float(err[1, 0]) < 1.0  # free: 0.3 m < 0.5 m
    assert float(err[1, 1]) == 0.0


class TestRepresentation:
  @pytest.fixture
  def rep(self, lib: HoiLibrary):
    from supermimic.hoi.geometry import hand_probes
    from supermimic.hoi.representation import HoiGenCoordRepresentation

    nj = 29
    r = HoiGenCoordRepresentation(nj, torch.zeros(nj))
    r.dt, r.substep_coeff = 0.02, 0.625
    r.lib = lib
    r.probes, r.radii = hand_probes("cpu")
    r._wrist_idx = [10, 13]

    def fk(root_pos, root_quat, qj, *_):
      n = root_pos.shape[0]
      return root_pos[:, None].expand(n, 14, 3) + 0.0, root_quat[:, None].expand(n, 14, 4) + 0.0, None, None

    r.fk = MagicMock(side_effect=fk)
    return r

  def _full(self, rep, n: int = 3) -> torch.Tensor:
    nj = rep.num_joints
    root_q = torch.nn.functional.normalize(torch.randn(n, 4), dim=-1)
    obj_q = torch.nn.functional.normalize(torch.randn(n, 4), dim=-1)
    return torch.cat(
      [torch.randn(n, 3), root_q, 0.1 * torch.randn(n, nj), torch.randn(n, 6), torch.randn(n, nj),
       torch.randn(n, 3), obj_q, torch.randn(n, 6), torch.tensor([[0.0], [1.0], [0.0]])[:n]],
      dim=-1,
    )

  def test_dimensions(self, rep):
    full = self._full(rep)
    assert full.shape[-1] == rep.full_state_dim
    assert rep.represent(full).shape[-1] == rep.rep_state_dim
    assert rep.rep_state_dim == 10 + 2 * 29 + 16 + HAND_FEATURE_DIM + 1

  def test_reconstruct_round_trip(self, rep):
    full = self._full(rep)
    aux, state = rep.extract_aux_state(full), rep.represent(full)
    back = rep.reconstruct_full_state(aux, state)
    assert torch.allclose(back, full, atol=1e-4)

  def test_zero_prediction_keeps_object_velocity(self, rep):
    full = self._full(rep)
    aux, state = rep.extract_aux_state(full), rep.represent(full)
    next_aux, next_rep = rep.apply_prediction(aux, state, torch.zeros(3, rep.prediction_dim), rep.dt)
    obj_v = full[:, 71 + 7 : 71 + 10]
    assert torch.allclose(next_aux[:, 7:10], aux[:, 7:10] + obj_v * rep.dt, atol=1e-5)
    nf = rep.reconstruct_full_state(next_aux, next_rep)
    assert torch.allclose(nf[:, 71 + 7 : 71 + 10], obj_v, atol=1e-4)
    assert torch.equal(next_aux[:, 14], aux[:, 14])
    assert torch.allclose(rep.represent(nf), next_rep, atol=1e-4)


def test_object_stats_split_by_type_and_reset_after_summary():
  from supermimic.hoi.stats import ObjectStats

  stats = ObjectStats(["largebox", "whitechair"], "cpu")
  types = torch.tensor([0, 0, 1])
  stats.add_steps(types, torch.tensor([[0.1, 0.2, 0.3], [0.3, 0.4, 0.5], [1.0, 1.0, 1.0]]))
  stats.add_steps(types, torch.tensor([[0.1, 0.2, 0.3], [0.3, 0.4, 0.5], [3.0, 3.0, 3.0]]))
  outcomes = torch.tensor([[True, False, False], [False, True, False]])
  stats.add_episodes(torch.tensor([0, 0]), torch.tensor([100, 50]), outcomes)
  out = stats.summary(step_dt=0.02)
  assert out["HOI/largebox/object_pos_error"] == pytest.approx(0.2)
  assert out["HOI/whitechair/object_pos_error"] == pytest.approx(2.0)
  assert out["HOI/largebox/episodes"] == 2
  assert out["HOI/largebox/episode_length_s"] == pytest.approx(1.5)
  assert out["HOI/largebox/complete_rate"] == pytest.approx(0.5)
  assert out["HOI/largebox/bad_object_rate"] == pytest.approx(0.5)
  assert out["HOI/whitechair/episodes"] == 0
  assert "HOI/whitechair/complete_rate" not in out  # no finished episode yet
  assert stats.summary(step_dt=0.02) == {"HOI/largebox/episodes": 0.0, "HOI/whitechair/episodes": 0.0}


def test_logger_duration_shows_days():
  from supermimic.utils.logger import _duration

  assert _duration(3725) == "01:02:05"
  assert _duration(2 * 86400 + 61) == "2d 00:01:01"


def test_logger_reward_per_step_is_ratio_of_sums():
  from supermimic.utils.logger import Logger

  log = Logger(run_name=None, num_envs=2, max_episode_length=500, device="cpu")
  assert log._reward_per_step() == 0.0
  log.rewbuffer.extend([10.0, 30.0])
  log.lenbuffer.extend([100.0, 100.0])
  assert log._reward_per_step() == pytest.approx(0.2)


def _series(n: int, **overrides):
  from supermimic.hoi.evaluate import DIAGNOSTICS

  s = {k: np.zeros(n) for k in (*DIAGNOSTICS, "pelvis_drop_m")}
  s.update(overrides)
  return s


class TestEvalScoring:
  DT = 0.02

  def test_clean_clip_passes_everything(self):
    from supermimic.hoi.evaluate import score_clip

    out = score_clip(_series(100), np.arange(100) * self.DT, self.DT, diverged=False)
    assert out["phc_pass"] and out["object_pose_pass"] and out["body_and_object_pose_pass"]
    assert not out["fell"] and out["first_fail_s"] is None and out["seconds"] == pytest.approx(2.0)

  def test_one_frame_over_half_a_metre_fails_phc(self):
    from supermimic.hoi.evaluate import score_clip

    body = np.zeros(100)
    body[40] = 0.51
    out = score_clip(_series(100, body_err_m=body), np.arange(100) * self.DT, self.DT, diverged=False)
    assert not out["phc_pass"] and out["object_pose_pass"]
    assert out["first_fail_s"] == pytest.approx(0.8)

  def test_object_off_for_exactly_0p3_s_still_passes(self):
    from supermimic.hoi.evaluate import score_clip

    pos = np.zeros(100)
    pos[10:25] = 0.3  # 15 frames = 0.30 s
    assert score_clip(_series(100, object_pos_err_m=pos), np.arange(100) * self.DT, self.DT, False)["object_pose_pass"]
    pos[10:26] = 0.3  # 16 frames, longer than 0.3 s
    out = score_clip(_series(100, object_pos_err_m=pos), np.arange(100) * self.DT, self.DT, False)
    assert not out["object_pose_pass"] and out["first_fail_s"] == pytest.approx(25 * self.DT)

  def test_a_flipped_object_counts_as_off(self):
    from supermimic.hoi.evaluate import score_clip

    ori = np.full(100, 180.0)  # upside down, position perfect
    assert not score_clip(_series(100, object_ori_err_deg=ori), np.arange(100) * self.DT, self.DT, False)["object_pose_pass"]

  def test_fall_needs_0p3_s_below_the_reference(self):
    from supermimic.hoi.evaluate import score_clip

    drop = np.zeros(100)
    drop[50:64] = 0.4  # 14 frames
    assert not score_clip(_series(100, pelvis_drop_m=drop), np.arange(100) * self.DT, self.DT, False)["fell"]
    drop[50:65] = 0.4  # 15 frames = 0.3 s
    out = score_clip(_series(100, pelvis_drop_m=drop), np.arange(100) * self.DT, self.DT, False)
    assert out["fell"] and out["fall_s"] == pytest.approx(64 * self.DT)

  def test_diverged_clip_fails(self):
    from supermimic.hoi.evaluate import score_clip

    out = score_clip(_series(10), np.arange(10) * self.DT, self.DT, diverged=True)
    assert not out["phc_pass"] and not out["object_pose_pass"] and out["fell"]

  def test_summary_groups_by_object_and_kind(self):
    from supermimic.hoi.evaluate import score_clip, summarize

    good = score_clip(_series(50), np.arange(50) * self.DT, self.DT, False)
    bad = score_clip(_series(50, body_err_m=np.ones(50)), np.arange(50) * self.DT, self.DT, False)
    clips = [{"type": "largebox", "kind": "hand", **good}, {"type": "largebox", "kind": "foot", **bad},
             {"type": "whitechair", "kind": "hand", **good}]
    out = summarize(clips)
    assert out["all/phc_pass_rate"] == pytest.approx(2 / 3)
    assert out["largebox/phc_pass_rate"] == pytest.approx(0.5)
    assert out["foot/phc_pass_rate"] == 0.0 and out["hand/clips"] == 2
    assert out["all/body_err_m/mean"] == pytest.approx(1 / 3)
    assert out["foot/first_fail_s"] == pytest.approx(0.0)


def test_eval_env_cfg_is_nominal_and_unbounded(tmp_path):
  from supermimic.hoi.evaluate import KEEP_TERMINATIONS, hoi_eval_env_cfg

  cfg = hoi_eval_env_cfg(str(make_library(tmp_path)))
  assert cfg.seed == 42
  assert set(cfg.terminations) <= set(KEEP_TERMINATIONS) and "bad_object" not in cfg.terminations
  assert not {"foot_friction", "base_com", "encoder_bias"} & set(cfg.events)
  assert cfg.events["object_variant"].params["mass_scale_range"] == (1.0, 1.0)
  cmd = cfg.commands["motion"]
  assert cmd.eval_clips and cmd.sampling_mode == "uniform" and cmd.pose_range == {}
