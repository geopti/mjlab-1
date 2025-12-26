"""Curriculum learning functions for progressive jump training."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch

from .commands import JumpCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class HeightStage(TypedDict):
  """Stage definition for progressive height curriculum."""

  step: int
  target_height: float
  tolerance: float


class RewardWeightStage(TypedDict):
  """Stage definition for reward weight progression."""

  step: int
  weight: float


class EpisodeLengthStage(TypedDict):
  """Stage definition for episode length progression."""

  step: int
  episode_length_s: float


def progressive_jump_height(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  height_stages: list[HeightStage],
) -> dict[str, torch.Tensor]:
  """Update jump height target based on training progress.

  Args:
    env: The environment.
    env_ids: Environment IDs being reset (unused, applies globally).
    command_name: Name of the jump command term.
    height_stages: List of height stages with step thresholds.

  Returns:
    Dictionary of current target height and tolerance for logging.
  """
  del env_ids  # Curriculum applies globally, not per-env

  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = command_term.cfg
  assert isinstance(cfg, JumpCommandCfg)

  # Update target height based on current training step
  for stage in height_stages:
    if env.common_step_counter > stage["step"]:
      cfg.target_height = stage["target_height"]
      cfg.height_tolerance = stage["tolerance"]

  return {
    "target_height": torch.tensor(cfg.target_height),
    "height_tolerance": torch.tensor(cfg.height_tolerance),
  }


def progressive_stability_requirement(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Progressively increase landing stability reward weight.

  Args:
    env: The environment.
    env_ids: Environment IDs being reset (unused).
    reward_name: Name of the reward term to adjust.
    weight_stages: List of weight stages with step thresholds.

  Returns:
    Current reward weight for logging.
  """
  del env_ids  # Applies globally

  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]

  return torch.tensor([reward_term_cfg.weight])


def progressive_episode_length(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  length_stages: list[EpisodeLengthStage],
) -> torch.Tensor:
  """Progressively increase episode length for more landing practice.

  Args:
    env: The environment.
    env_ids: Environment IDs being reset (unused).
    length_stages: List of episode length stages with step thresholds.

  Returns:
    Current episode length for logging.
  """
  del env_ids  # Applies globally

  # Update episode length based on training progress
  for stage in length_stages:
    if env.common_step_counter > stage["step"]:
      new_length_s = stage["episode_length_s"]
      # Update max episode length
      env.cfg.episode_length_s = new_length_s
      # Recompute max_episode_length (number of steps)
      env.max_episode_length = int(
        new_length_s / (env.cfg.sim.mujoco.timestep * env.cfg.decimation)
      )

  return torch.tensor([env.cfg.episode_length_s])
