"""Protocol defining the interface mixins expect from ViserMujocoScene."""

from __future__ import annotations

from typing import Protocol

import mujoco
import viser


class ViserSceneProtocol(Protocol):
  """Protocol defining methods and attributes that mixins can call.

  This ensures type safety when mixins call methods on the parent class.
  """

  # Required attributes.
  server: viser.ViserServer
  mj_model: mujoco.MjModel
  num_envs: int
  env_idx: int
  camera_tracking_enabled: bool
  show_only_selected: bool
  geom_groups_visible: list[bool]
  site_groups_visible: list[bool]
  label_targets: set[str]
  frame_targets: set[str]
  frame_scale: float
  show_contact_points: bool
  show_contact_forces: bool
  meansize_override: float | None
  debug_visualization_enabled: bool
  contact_point_color: tuple[int, int, int]
  contact_force_color: tuple[int, int, int]
  contact_point_handle: viser.BatchedMeshHandle | None
  contact_force_shaft_handle: viser.BatchedMeshHandle | None
  contact_force_head_handle: viser.BatchedMeshHandle | None
  _label_handles: dict[str, viser.LabelHandle]
  _frame_handles: dict[str, viser.FrameHandle]
  _scene_offset: dict[str, float]
  _queued_arrows: list[tuple]
  _arrow_shaft_handle: viser.BatchedMeshHandle | None
  _arrow_head_handle: viser.BatchedMeshHandle | None
  _ghost_handles: dict[int, viser.SceneNodeHandle]
  _ghost_meshes: dict[int, dict[int, object]]
  _arrow_shaft_mesh: object | None
  _arrow_head_mesh: object | None
  _viz_data: mujoco.MjData

  # Required methods that mixins call.
  def _request_update(self) -> None: ...

  def _sync_visibilities(self) -> None: ...

  def _refresh_annotations(self) -> None: ...

  def clear_debug_all(self) -> None: ...
