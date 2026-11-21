"""Per-object training statistics for the HOI motion command.

The env's own ``Metrics/motion/*`` are read at the last step of each finished
episode, so episodes that fail end on their largest error. These sums cover every
world-step instead, split by object type, plus how episodes of each type end.
Everything stays on the device until ``summary()``, one host transfer per log.
"""

from __future__ import annotations

import torch

STEP_FIELDS = ("object_pos_error", "object_ori_error", "body_pos_error")
"""Per world-step errors, averaged over all steps since the last summary."""
OUTCOMES = ("complete", "bad_object", "fall")
"""How an episode ends: its clip (or time limit) ran out, it lost the object,
or the robot's anchor left its reference."""


class ObjectStats:
  """Running sums per object type since the last ``summary()``."""

  def __init__(self, type_names: list[str], device: str | torch.device):
    self.type_names = type_names
    n = len(type_names)
    self._steps = torch.zeros(n, len(STEP_FIELDS) + 1, device=device)  # sums + count
    self._episodes = torch.zeros(n, len(OUTCOMES) + 2, device=device)  # count, length, outcomes

  def add_steps(self, types: torch.Tensor, values: torch.Tensor) -> None:
    """``values``: ``(N, len(STEP_FIELDS))`` for worlds of object type ``types``."""
    ones = torch.ones_like(values[:, :1])
    self._steps.index_add_(0, types, torch.cat([values.float(), ones], dim=-1))

  def add_episodes(self, types: torch.Tensor, lengths: torch.Tensor, outcomes: torch.Tensor) -> None:
    """Finished episodes: ``lengths`` in steps, ``outcomes`` ``(N, len(OUTCOMES))`` flags."""
    ones = torch.ones_like(lengths[:, None], dtype=torch.float)
    rows = torch.cat([ones, lengths[:, None].float(), outcomes.float()], dim=-1)
    self._episodes.index_add_(0, types, rows)

  def summary(self, step_dt: float) -> dict[str, float]:
    """``HOI/<type>/<stat>`` since the last call, then start over."""
    steps, episodes = self._steps.tolist(), self._episodes.tolist()
    self._steps.zero_()
    self._episodes.zero_()
    out: dict[str, float] = {}
    for name, s, e in zip(self.type_names, steps, episodes, strict=True):
      if s[-1] > 0:
        out |= {f"HOI/{name}/{f}": v / s[-1] for f, v in zip(STEP_FIELDS, s, strict=False)}
      out[f"HOI/{name}/episodes"] = e[0]
      if e[0] > 0:
        out[f"HOI/{name}/episode_length_s"] = e[1] / e[0] * step_dt
        out |= {f"HOI/{name}/{o}_rate": v / e[0] for o, v in zip(OUTCOMES, e[2:], strict=True)}
    return out
