"""Jump command generator for height target management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class JumpCommand(CommandTerm):
  """Command term for jump height targets.

  Provides a simple command that specifies the target jump height.
  The target is set once per episode and doesn't change during the episode.
  """

  cfg: JumpCommandCfg

  def __init__(self, cfg: JumpCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    # Command is just the target height [B, 1]
    self.height_command = torch.full(
      (self.num_envs, 1),
      cfg.target_height,
      device=self.device,
    )

    # Track metrics
    self.metrics["target_height"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["peak_height"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    """Return the jump height command."""
    return self.height_command

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Update command for reset environments.

    Args:
      env_ids: Environment IDs to resample commands for.
    """
    # Update to current curriculum target
    self.height_command[env_ids] = self.cfg.target_height

  def _update_command(self) -> None:
    """Update command during episode (no-op for jump task)."""
    pass

  def _update_metrics(self) -> None:
    """Update command metrics."""
    self.metrics["target_height"][:] = self.cfg.target_height


@dataclass(kw_only=True)
class JumpCommandCfg(CommandTermCfg):
  """Configuration for jump height commands.

  Attributes:
    class_type: The command term class.
    target_height: Target jump height in meters (will be updated by curriculum).
    height_tolerance: Success window for height achievement.
  """

  class_type: type[CommandTerm] = JumpCommand
  resampling_time_range: tuple[float, float] = (1e9, 1e9)  # Never resample
  target_height: float = 0.25  # 25cm default
  height_tolerance: float = 0.05  # 5cm tolerance
