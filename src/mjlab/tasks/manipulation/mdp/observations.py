from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensor
from mjlab.tasks.manipulation.mdp.commands import LiftingCommand
from mjlab.utils.lab_api.math import quat_apply, quat_inv

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def ee_to_object_distance(
  env: ManagerBasedRlEnv,
  object_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Distance vector from end effector to object in robot base frame."""
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  obj_pos_w = obj.data.root_link_pos_w
  distance_vec_w = obj_pos_w - ee_pos_w
  base_quat_w = robot.data.root_link_quat_w
  distance_vec_b = quat_apply(quat_inv(base_quat_w), distance_vec_w)
  return distance_vec_b


def object_position_error(
  env: ManagerBasedRlEnv,
  object_name: str,
  command_name: str,
) -> torch.Tensor:
  """3D position error between object and target position (target - current)."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, LiftingCommand):
    raise TypeError(
      f"Command '{command_name}' must be a LiftingCommand, got {type(command)}"
    )
  obj: Entity = env.scene[object_name]
  obj_pos_w = obj.data.root_link_pos_w
  position_error = command.target_pos - obj_pos_w
  return position_error


def camera_rgb(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  normalize: bool = True,
) -> torch.Tensor:
  """Get RGB camera observation in CNN-compatible format (B, C, H, W)."""
  sensor: CameraSensor = env.scene[sensor_name]
  rgb_data = sensor.data.rgb
  assert rgb_data is not None, f"Camera '{sensor_name}' has no RGB data"
  rgb_data = rgb_data.permute(0, 3, 1, 2)
  if normalize:
    rgb_data = rgb_data.float() / 255.0
  return rgb_data
