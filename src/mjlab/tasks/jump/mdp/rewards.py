"""Jump-specific reward functions for phase-aware learning."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class jump_height_reward:
  """Reward for achieving target jump height.

  Tracks the peak height achieved during the episode and rewards based on
  how close it is to the target height using an exponential reward function.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.peak_heights = torch.zeros(env.num_envs, device=env.device)
    self.initial_heights = torch.zeros(env.num_envs, device=env.device)
    self.initialized = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    current_height = asset.data.root_link_pos_w[:, 2]
    # For flat terrain, terrain height is 0
    terrain_height = 0.0
    height_above_ground = current_height - terrain_height

    # Initialize heights at episode start
    self.initial_heights = torch.where(
      ~self.initialized, height_above_ground, self.initial_heights
    )
    self.initialized = torch.ones_like(self.initialized)

    # Track peak height above initial crouch
    self.peak_heights = torch.maximum(self.peak_heights, height_above_ground)

    # Compute jump height (peak - initial)
    jump_height = self.peak_heights - self.initial_heights

    # Exponential reward for matching target
    reward = torch.exp(-((jump_height - target_height) ** 2) / (std**2))

    # Log metrics
    env.extras["log"]["Metrics/peak_jump_height"] = torch.mean(self.peak_heights)
    env.extras["log"]["Metrics/jump_height"] = torch.mean(jump_height)

    return reward

  def reset_idx(self, env_ids: torch.Tensor) -> None:
    """Reset tracked heights for specified environments."""
    self.peak_heights[env_ids] = 0.0
    self.initial_heights[env_ids] = 0.0
    self.initialized[env_ids] = False


def explosive_takeoff(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  power_threshold: float = 500.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward high joint power during takeoff phase (while in contact).

  Args:
    env: The environment.
    sensor_name: Contact sensor name for detecting ground contact.
    power_threshold: Minimum power threshold for reward.
    asset_cfg: Asset configuration.

  Returns:
    Reward for explosive power generation during contact.
  """
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]

  # Only reward during contact
  in_contact = (contact_sensor.data.found > 0).any(dim=1)  # [B]

  # Compute instantaneous power: P = τ * ω
  actuator_forces = asset.data.actuator_force  # [B, N]
  joint_vels = asset.data.joint_vel  # [B, N]
  power = torch.abs(actuator_forces * joint_vels)  # [B, N]

  # Focus on leg joints (hip, knee, ankle) for jumping
  leg_joint_indices = asset_cfg.joint_ids if asset_cfg.joint_ids else slice(None)
  leg_power = power[:, leg_joint_indices]  # [B, M]

  # Total power across legs
  total_power = torch.sum(leg_power, dim=1)  # [B]

  # Reward power above threshold during contact
  reward = torch.clamp(total_power - power_threshold, min=0.0) * in_contact.float()

  return reward / 1000.0  # Scale down to reasonable range


def synchronized_extension(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize asymmetric leg extension (left vs right).

  Encourages both legs to extend synchronously during takeoff to prevent
  rotation and ensure vertical jumping.

  Args:
    env: The environment.
    asset_cfg: Asset configuration with joint_names for legs.

  Returns:
    Penalty for leg asymmetry.
  """
  asset: Entity = env.scene[asset_cfg.name]
  joint_vel = asset.data.joint_vel  # [B, N]

  # Find left and right leg joint indices
  # Assuming joints are ordered or we can match by name patterns
  # For G1: left_hip_pitch, left_knee, left_ankle vs right_*
  # Simple approach: assume first half is left, second half is right for leg joints
  # Better: use named joint access (requires joint name matching)

  # For now, compute variance across all joints as a proxy
  # A better implementation would specifically compare left vs right leg joints
  joint_vel_mean = torch.mean(joint_vel, dim=1, keepdim=True)
  joint_vel_var = torch.mean((joint_vel - joint_vel_mean) ** 2, dim=1)

  return joint_vel_var


def vertical_impulse(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Reward vertical ground reaction forces during takeoff.

  Args:
    env: The environment.
    sensor_name: Contact sensor name.

  Returns:
    Reward for vertical force generation.
  """
  contact_sensor: ContactSensor = env.scene[sensor_name]
  forces = contact_sensor.data.force  # [B, N, 3]

  if forces is None:
    return torch.zeros(env.num_envs, device=env.device)

  # Vertical force is z-component
  vertical_forces = forces[:, :, 2]  # [B, N]

  # Only reward positive (upward) forces
  vertical_forces = torch.clamp(vertical_forces, min=0.0)

  # Sum across feet
  total_vertical_force = torch.sum(vertical_forces, dim=1)  # [B]

  # Normalize to reasonable scale
  return total_vertical_force / 500.0


def air_time_bonus(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  min_air_time: float = 0.2,
) -> torch.Tensor:
  """Reward achieving minimum air time (both feet off ground).

  Args:
    env: The environment.
    sensor_name: Contact sensor name.
    min_air_time: Minimum air time threshold for reward.

  Returns:
    Reward for achieving flight.
  """
  contact_sensor: ContactSensor = env.scene[sensor_name]
  current_air_time = contact_sensor.data.current_air_time

  if current_air_time is None:
    return torch.zeros(env.num_envs, device=env.device)

  # Both feet must be in air (min of left/right air times)
  min_foot_air_time = torch.min(current_air_time, dim=1)[0]  # [B]

  # Exponential reward for air time above threshold
  reward = torch.exp((min_foot_air_time - min_air_time) / min_air_time) - 1.0
  reward = torch.clamp(reward, min=0.0)

  # Log mean air time
  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
    num_in_air, min=1
  )
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time

  return reward


class landing_balance:
  """Reward maintaining balance after landing.

  Tracks stability duration post-landing and rewards exponentially
  increasing stability time.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.stability_timer = torch.zeros(env.num_envs, device=env.device)
    self.was_in_air = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    stability_time: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]

    # Detect contact state
    in_contact = (contact_sensor.data.found > 0).any(dim=1)  # [B]

    # Track landing events (transition from air to contact)
    in_air = ~in_contact
    just_landed = self.was_in_air & in_contact
    self.was_in_air = in_air

    # Check if stable (upright + low velocities + on ground)
    is_upright = torch.abs(asset.data.projected_gravity_b[:, 2] + 1.0) < 0.2
    lin_vel_norm = torch.norm(asset.data.root_link_lin_vel_w, dim=1)
    ang_vel_norm = torch.norm(asset.data.root_link_ang_vel_w, dim=1)
    low_vel = (lin_vel_norm < 0.5) & (ang_vel_norm < 0.5)
    is_stable = is_upright & low_vel & in_contact

    # Reset timer on landing
    self.stability_timer = torch.where(
      just_landed, torch.zeros_like(self.stability_timer), self.stability_timer
    )

    # Increment timer when stable
    self.stability_timer = torch.where(
      is_stable,
      self.stability_timer + self.step_dt,
      torch.zeros_like(self.stability_timer),
    )

    # Exponential reward based on stability duration
    reward = torch.exp(self.stability_timer / stability_time) - 1.0

    # Log landing success rate
    stable_landing = self.stability_timer > stability_time
    env.extras["log"]["Metrics/landing_success_rate"] = torch.mean(
      stable_landing.float()
    )

    return reward

  def reset_idx(self, env_ids: torch.Tensor) -> None:
    """Reset stability tracking for specified environments."""
    self.stability_timer[env_ids] = 0.0
    self.was_in_air[env_ids] = False


def symmetric_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  time_tolerance: float = 0.05,
) -> torch.Tensor:
  """Reward landing with both feet simultaneously.

  Args:
    env: The environment.
    sensor_name: Contact sensor name.
    time_tolerance: Maximum time difference for simultaneous landing (seconds).

  Returns:
    Reward for symmetric landing.
  """
  contact_sensor: ContactSensor = env.scene[sensor_name]

  # Detect first contact events
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]

  if first_contact.shape[1] < 2:
    return torch.zeros(env.num_envs, device=env.device)

  # Check if both feet land in same timestep (within tolerance)
  left_lands = first_contact[:, 0]
  right_lands = first_contact[:, 1]

  # Reward simultaneous landing
  simultaneous = left_lands & right_lands

  return simultaneous.float()
