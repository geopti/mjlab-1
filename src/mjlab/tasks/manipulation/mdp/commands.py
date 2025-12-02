from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg
from mjlab.utils.lab_api.math import (
  quat_from_euler_xyz,
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class PoseCommand(CommandTerm):
  cfg: PoseCommandCfg

  def __init__(self, cfg: PoseCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)

    # Set target position based on difficulty mode.
    if self.cfg.difficulty == "fixed":
      target_pos = torch.tensor(
        [0.4, 0.0, 0.3], device=self.device, dtype=torch.float32
      ).expand(n, 3)
      self.target_pos[env_ids] = target_pos + self._env.scene.env_origins[env_ids]
    else:
      assert self.cfg.difficulty == "dynamic"
      r = self.cfg.target_position_range
      lower = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
      upper = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
      target_pos = sample_uniform(lower, upper, (n, 3), device=self.device)
      self.target_pos[env_ids] = target_pos + self._env.scene.env_origins[env_ids]

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    batch = visualizer.env_idx
    if batch >= self.num_envs:
      return

    target_pos = self.target_pos[batch].cpu().numpy()
    visualizer.add_sphere(
      center=target_pos,
      radius=0.03,
      color=self.cfg.viz.target_color,
      label="target_position",
    )


@dataclass(kw_only=True)
class PoseCommandCfg(CommandTermCfg):
  class_type: type[CommandTerm] = PoseCommand
  difficulty: Literal["fixed", "dynamic"] = "fixed"

  @dataclass
  class TargetPositionRangeCfg:
    """Configuration for target position sampling in dynamic mode."""

    x: tuple[float, float] = (0.3, 0.5)
    y: tuple[float, float] = (-0.2, 0.2)
    z: tuple[float, float] = (0.2, 0.4)

  # Only used in dynamic mode.
  target_position_range: TargetPositionRangeCfg = field(
    default_factory=TargetPositionRangeCfg
  )

  @dataclass
  class VizCfg:
    target_color: tuple[float, float, float, float] = (1.0, 0.5, 0.0, 0.3)

  viz: VizCfg = field(default_factory=VizCfg)
