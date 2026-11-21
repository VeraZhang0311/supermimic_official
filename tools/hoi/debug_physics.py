"""Step the HOI scene on CPU MuJoCo from a reference frame and print the object.

  uv run python tools/hoi/debug_physics.py <library dir> [frame]
"""

from __future__ import annotations

import sys

import mujoco
import numpy as np

import supermimic.tasks  # noqa: F401
from mjlab.scene import Scene
from supermimic.hoi.env_cfg import configure_hoi_env, hoi_env_cfg


def main() -> None:
  lib = sys.argv[1]
  frame = int(sys.argv[2]) if len(sys.argv) > 2 else 0
  cfg = hoi_env_cfg()
  configure_hoi_env(cfg, lib)
  cfg.scene.num_envs = 1
  model = Scene(cfg.scene, device="cpu").spec.compile()
  model.opt.timestep = 0.005
  model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  model.opt.impratio = 10
  data = mujoco.MjData(model)
  obj_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object/object")
  jnt = model.body_jntadr[obj_body]
  qa, va = model.jnt_qposadr[jnt], model.jnt_dofadr[jnt]
  print("object mass", model.body_mass[obj_body], "inertia", model.body_inertia[obj_body],
        "ipos", model.body_ipos[obj_body], "iquat", model.body_iquat[obj_body])
  pos = np.load(f"{lib}/hoi/object_pos_w.npy")[frame]
  quat = np.load(f"{lib}/hoi/object_quat_w.npy")[frame]
  data.qpos[qa : qa + 3] = pos + np.array([0.0, 3.0, 0.0])  # beside the robot
  data.qpos[qa + 3 : qa + 7] = quat
  mujoco.mj_forward(model, data)
  print(f"frame {frame} object pos {pos} quat {quat}; ncon {data.ncon}")
  for step in range(10):
    for _ in range(4):
      mujoco.mj_step(model, data)
    print(f"step {step + 1}: pos {data.qpos[qa : qa + 3].round(4)} vel {data.qvel[va : va + 3].round(3)} ncon {data.ncon}")


if __name__ == "__main__":
  main()
