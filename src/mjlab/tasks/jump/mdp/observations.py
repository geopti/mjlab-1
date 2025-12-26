"""Jump-specific observation functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def height_above_ground(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Return the height of the robot's root link above the terrain.

  Args:
    env: The environment.
    asset_cfg: The asset configuration.

  Returns:
    Tensor of shape (num_envs, 1) containing height above ground.
  """
  asset: Entity = env.scene[asset_cfg.name]
  root_pos_w = asset.data.root_link_pos_w  # [B, 3]
  current_height = root_pos_w[:, 2]  # [B]

  # For flat terrain (plane), terrain height is 0
  # For procedural terrain, we would need to query terrain height
  # For now, assume flat terrain at z=0
  terrain_height = 0.0

  height = current_height - terrain_height  # [B]
  return height.unsqueeze(-1)  # [B, 1]


def vertical_velocity(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Return the vertical velocity of the robot's root link.

  Args:
    env: The environment.
    asset_cfg: The asset configuration.

  Returns:
    Tensor of shape (num_envs, 1) containing vertical velocity (z-component).
  """
  asset: Entity = env.scene[asset_cfg.name]
  # Use world frame velocity and extract z-component
  root_lin_vel_w = asset.data.root_link_lin_vel_w  # [B, 3]
  return root_lin_vel_w[:, 2:3]  # [B, 1]


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Return the height of foot sites above ground.

  Args:
    env: The environment.
    asset_cfg: The asset configuration with site_names for feet.

  Returns:
    Tensor of shape (num_envs, num_sites) containing foot heights.
  """
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Return the air time for each foot from contact sensor.

  Args:
    env: The environment.
    sensor_name: Name of the contact sensor.

  Returns:
    Tensor of shape (num_envs, num_feet) containing air time.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Return binary contact state for each foot.

  Args:
    env: The environment.
    sensor_name: Name of the contact sensor.

  Returns:
    Tensor of shape (num_envs, num_feet) with 1.0 for contact, 0.0 otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Return log-scaled contact forces for each foot.

  Args:
    env: The environment.
    sensor_name: Name of the contact sensor.

  Returns:
    Tensor of shape (num_envs, num_feet*3) containing log-scaled forces.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  # Log scaling to reduce magnitude range
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))
