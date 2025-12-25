"""Jump-specific observation functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def pelvis_height(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Return current pelvis/root height above ground.

    Returns:
        Tensor of shape [num_envs, 1] with pelvis z position.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2:3]


def pelvis_vertical_velocity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Return pelvis vertical velocity in world frame.

    Returns:
        Tensor of shape [num_envs, 1] with pelvis z velocity.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_w[:, 2:3]


def feet_in_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Return binary contact state for feet.

    Returns:
        Tensor of shape [num_envs, num_feet] with 1.0 if in contact, 0.0 otherwise.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    found = sensor.data.found
    assert found is not None
    return (found > 0).float()


def both_feet_in_contact(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Return whether both feet are in contact.

    Returns:
        Tensor of shape [num_envs, 1] with 1.0 if both feet in contact.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    found = sensor.data.found
    assert found is not None
    # Both feet must be in contact
    both_contact = (found > 0).all(dim=1, keepdim=True)
    return both_contact.float()


def pelvis_height_relative(
    env: ManagerBasedRlEnv,
    standing_height: float = 0.76,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Return pelvis height relative to standing height.

    This gives the policy a sense of how high above/below normal standing it is.

    Returns:
        Tensor of shape [num_envs, 1] with relative height.
    """
    asset: Entity = env.scene[asset_cfg.name]
    current_height = asset.data.root_link_pos_w[:, 2:3]
    return current_height - standing_height
