"""Jump-specific termination conditions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def excessive_landing_force(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 2500.0,
) -> torch.Tensor:
  """Terminate if landing forces exceed safety threshold.

  This prevents learning destructive landing patterns that would damage
  a real robot.

  Args:
    env: The environment.
    sensor_name: Contact sensor name for detecting ground contact forces.
    force_threshold: Maximum allowable landing force in Newtons.

  Returns:
    Boolean tensor indicating which environments should terminate.
  """
  contact_sensor: ContactSensor = env.scene[sensor_name]
  forces = contact_sensor.data.force  # [B, N, 3]

  if forces is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  # Compute force magnitude
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]

  # Check if any foot exceeds threshold
  max_force = torch.max(force_magnitude, dim=1)[0]  # [B]

  return max_force > force_threshold
