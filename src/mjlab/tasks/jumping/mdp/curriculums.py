"""Curriculum functions for the jumping task."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.tasks.jumping.mdp.commands import JumpCommandCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class JumpHeightStage(TypedDict):
    """Configuration for a jump height curriculum stage."""

    step: int
    target_height: tuple[float, float]


def jump_height_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    height_stages: list[JumpHeightStage],
) -> dict[str, torch.Tensor]:
    """Update jump target height range based on training progress.

    This curriculum progressively increases the target jump height
    as training progresses, starting with small hops and working
    up to higher jumps.

    Args:
        env: The environment instance.
        env_ids: Environment indices (unused but required by interface).
        command_name: Name of the jump command term.
        height_stages: List of stages with step thresholds and height ranges.

    Returns:
        Dictionary with current height range for logging.
    """
    del env_ids  # Unused.

    command_term = env.command_manager.get_term(command_name)
    assert command_term is not None
    cfg = cast(JumpCommandCfg, command_term.cfg)

    # Apply the appropriate stage based on current step count
    for stage in height_stages:
        if env.common_step_counter > stage["step"]:
            cfg.ranges.target_height = stage["target_height"]

    return {
        "target_height_min": torch.tensor(cfg.ranges.target_height[0]),
        "target_height_max": torch.tensor(cfg.ranges.target_height[1]),
    }


class RewardWeightStage(TypedDict):
    """Configuration for a reward weight curriculum stage."""

    step: int
    weight: float


def reward_weight_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
    """Update a reward term's weight based on training progress.

    This can be used to gradually increase or decrease the importance
    of certain rewards as training progresses.

    Args:
        env: The environment instance.
        env_ids: Environment indices (unused but required by interface).
        reward_name: Name of the reward term to modify.
        weight_stages: List of stages with step thresholds and weights.

    Returns:
        Current weight value for logging.
    """
    del env_ids  # Unused.

    reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
    for stage in weight_stages:
        if env.common_step_counter > stage["step"]:
            reward_term_cfg.weight = stage["weight"]

    return torch.tensor([reward_term_cfg.weight])
