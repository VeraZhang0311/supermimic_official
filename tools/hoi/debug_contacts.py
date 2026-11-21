"""List the contacts of the HOI scene at its initial keyframe (CPU MuJoCo).

  uv run python tools/hoi/debug_contacts.py <packed HOI library dir>
"""

from __future__ import annotations

import collections
import sys

import mujoco

import supermimic.tasks  # noqa: F401
from mjlab.scene import Scene
from supermimic.hoi.env_cfg import configure_hoi_env, hoi_env_cfg


def main() -> None:
  cfg = hoi_env_cfg()
  configure_hoi_env(cfg, sys.argv[1])
  cfg.scene.num_envs = 1
  scene = Scene(cfg.scene, device="cpu")
  model = scene.spec.compile()
  data = mujoco.MjData(model)  # qpos0, as the simulator's first forward sees it
  if len(sys.argv) > 2 and sys.argv[2] == "--keyframe" and model.nkey:
    mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  print(f"ngeom={model.ngeom} nbody={model.nbody} ncon={data.ncon} nefc={data.nefc}")
  pairs = collections.Counter()
  for c in data.contact[: data.ncon]:
    b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom1])
    b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[c.geom2])
    g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or "?"
    g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or "?"
    pairs[(b1, b2, g1.split("/")[-1].rsplit("_", 1)[0], g2.split("/")[-1].rsplit("_", 1)[0])] += 1
  for k, v in pairs.most_common(30):
    print(v, k)


if __name__ == "__main__":
  main()
