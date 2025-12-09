import mujoco
import torch


def is_position_actuator(actuator: mujoco.MjsActuator) -> bool:
  """Check if an actuator is a position actuator.

  This function works on both model.actuator and spec.actuator objects.
  """
  return (
    actuator.gaintype == mujoco.mjtGain.mjGAIN_FIXED
    and actuator.biastype == mujoco.mjtBias.mjBIAS_AFFINE
    and actuator.dyntype in (mujoco.mjtDyn.mjDYN_NONE, mujoco.mjtDyn.mjDYN_FILTEREXACT)
    and actuator.gainprm[0] == -actuator.biasprm[1]
  )


def dof_width(joint_type: int | mujoco.mjtJoint) -> int:
  """Get the dimensionality of the joint in qvel."""
  if isinstance(joint_type, mujoco.mjtJoint):
    joint_type = joint_type.value
  return {0: 6, 1: 3, 2: 1, 3: 1}[joint_type]


def qpos_width(joint_type: int | mujoco.mjtJoint) -> int:
  """Get the dimensionality of the joint in qpos."""
  if isinstance(joint_type, mujoco.mjtJoint):
    joint_type = joint_type.value
  return {0: 7, 1: 4, 2: 1, 3: 1}[joint_type]


def compute_geom_rbound(geom_type: int, size: torch.Tensor) -> torch.Tensor:
  """Compute bounding sphere radius for a geometry."""
  if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
    return size[..., 0]
  elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
    return size[..., 0] + size[..., 1]  # radius + half_length
  elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
    return torch.sqrt(size[..., 0] ** 2 + size[..., 1] ** 2)  # sqrt(r² + h²)
  elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
    return torch.sqrt(size[..., 0] ** 2 + size[..., 1] ** 2 + size[..., 2] ** 2)
  elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
    return torch.max(size, dim=-1).values
  elif geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
    return torch.zeros_like(size[..., 0])  # Plane has rbound=0
  else:
    raise ValueError(f"Unsupported geom type for rbound computation: {geom_type}")


def compute_geom_aabb(geom_type: int, size: torch.Tensor) -> torch.Tensor:
  """Compute axis-aligned bounding box size for a geometry in local frame."""
  if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
    r = size[..., 0:1]
    return r.expand(*size.shape[:-1], 3)
  elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
    # size = [radius, half_length, unused]
    # AABB = (radius, radius, radius + half_length)
    r = size[..., 0]
    h = size[..., 1]
    return torch.stack([r, r, r + h], dim=-1)
  elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
    # size = [radius, half_length, unused]
    r = size[..., 0]
    h = size[..., 1]
    return torch.stack([r, r, h], dim=-1)
  elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
    # size = [half_x, half_y, half_z]
    return size
  elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
    # size = [radius_x, radius_y, radius_z]
    return size
  elif geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
    return torch.zeros_like(size)
  else:
    raise ValueError(f"Unsupported geom type for AABB computation: {geom_type}")
