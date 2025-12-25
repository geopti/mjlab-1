"""Jump-specific reward functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class jump_height_reward:
    """Reward for achieving target jump height.

    This is a stateful reward class that tracks peak height during each jump
    and rewards the robot when it lands based on how close it got to the target.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.standing_height = cfg.params.get("standing_height", 0.76)
        self.peak_height = torch.zeros(env.num_envs, device=env.device)
        self.was_in_flight = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self.step_dt = env.step_dt

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        command_name: str,
        sensor_name: str,
        std: float = 0.1,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        standing_height: float = 0.76,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        contact_sensor: ContactSensor = env.scene[sensor_name]

        # Get current height
        current_height = asset.data.root_link_pos_w[:, 2]

        # Track peak height
        self.peak_height = torch.maximum(self.peak_height, current_height)

        # Get contact state
        found = contact_sensor.data.found
        assert found is not None
        both_feet_contact = (found > 0).all(dim=1)
        any_foot_in_air = ~both_feet_contact

        # Track if we've been airborne
        self.was_in_flight = self.was_in_flight | any_foot_in_air

        # Detect landing: was in flight AND now both feet in contact
        just_landed = self.was_in_flight & both_feet_contact

        # Get target height from command
        command = env.command_manager.get_command(command_name)
        assert command is not None
        target_height = standing_height + command[:, 1]

        # Compute reward on landing
        # Exponential reward based on how close peak was to target
        height_achieved = self.peak_height - standing_height
        error = torch.abs(self.peak_height - target_height)
        reward = torch.exp(-error / std) * just_landed.float()

        # Log metrics
        num_landings = just_landed.sum().item()
        if num_landings > 0:
            mean_height = (height_achieved * just_landed.float()).sum() / num_landings
            env.extras["log"]["Metrics/jump_height_mean"] = mean_height
            env.extras["log"]["Metrics/num_landings"] = torch.tensor(num_landings)

        # Reset state for environments that just landed
        self.peak_height = torch.where(just_landed, current_height, self.peak_height)
        self.was_in_flight = torch.where(
            just_landed,
            torch.zeros_like(self.was_in_flight),
            self.was_in_flight,
        )

        return reward


def launch_velocity_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward upward velocity when jump is commanded and feet are in contact.

    This encourages the robot to push off the ground when a jump is triggered.
    """
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]

    # Get vertical velocity
    vertical_vel = asset.data.root_link_lin_vel_w[:, 2]

    # Get contact state - reward only when feet are in contact (pushing phase)
    found = contact_sensor.data.found
    assert found is not None
    any_foot_in_contact = (found > 0).any(dim=1)

    # Get jump trigger from command
    command = env.command_manager.get_command(command_name)
    assert command is not None
    jump_triggered = command[:, 0] > 0.5

    # Reward upward velocity when jumping and in contact
    reward = torch.clamp(vertical_vel, min=0.0) * any_foot_in_contact.float() * jump_triggered.float()

    return reward


def horizontal_drift_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize horizontal velocity during jump.

    For vertical jumping, we want minimal XY movement.
    """
    asset: Entity = env.scene[asset_cfg.name]
    xy_vel = asset.data.root_link_lin_vel_w[:, :2]
    return torch.sum(torch.square(xy_vel), dim=1)


def excessive_rotation_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize excessive rotation during flight phase.

    We want the robot to stay upright during the jump.
    """
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]

    # Only penalize during flight
    found = contact_sensor.data.found
    assert found is not None
    both_feet_contact = (found > 0).all(dim=1)
    in_flight = ~both_feet_contact

    # Get angular velocity (roll and pitch components)
    ang_vel = asset.data.root_link_ang_vel_w[:, :2]  # XY angular velocity
    ang_vel_sq = torch.sum(torch.square(ang_vel), dim=1)

    return ang_vel_sq * in_flight.float()


def stable_landing_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    std: float = 0.2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward stable posture when both feet are in contact after landing.

    Combines upright orientation with both feet grounded.
    """
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]

    # Check if both feet in contact
    found = contact_sensor.data.found
    assert found is not None
    both_feet_contact = (found > 0).all(dim=1)

    # Get uprightness (projected gravity penalty)
    projected_gravity = asset.data.projected_gravity_b
    xy_squared = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
    upright_reward = torch.exp(-xy_squared / std**2)

    # Combine: reward upright posture when grounded
    return upright_reward * both_feet_contact.float()


class continuous_jump_height:
    """Continuously reward being high during flight (dense reward).

    This provides more gradient signal compared to sparse landing-only rewards.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.standing_height = cfg.params.get("standing_height", 0.76)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        standing_height: float = 0.76,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        contact_sensor: ContactSensor = env.scene[sensor_name]

        # Get current height relative to standing
        current_height = asset.data.root_link_pos_w[:, 2]
        relative_height = current_height - standing_height

        # Only reward when in flight (no contact)
        found = contact_sensor.data.found
        assert found is not None
        both_feet_contact = (found > 0).all(dim=1)
        in_flight = ~both_feet_contact

        # Reward height during flight (clipped to positive)
        return torch.clamp(relative_height, min=0.0) * in_flight.float()


def feet_slip_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize foot sliding (xy velocity while in contact).

    Simplified version for jumping task that doesn't depend on velocity commands.
    """
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]

    # Get contact state
    found = contact_sensor.data.found
    assert found is not None
    in_contact = (found > 0).float()  # [B, N]

    # Get foot velocities
    foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
    vel_xy_norm_sq = torch.sum(torch.square(foot_vel_xy), dim=-1)  # [B, N]

    # Penalize velocity while in contact
    cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1)

    return cost
