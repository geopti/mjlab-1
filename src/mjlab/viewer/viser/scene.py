"""Manages all Viser visualization handles and state for MuJoCo models."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
import trimesh
import trimesh.visual
import viser
import viser.transforms as vtf
from mujoco import mj_id2name, mjtGeom, mjtObj

from mjlab.viewer.debug_visualizer import DebugVisualizer
from mjlab.viewer.viser.conversions import (
  get_body_name,
  is_fixed_body,
  merge_geoms,
  merge_sites,
)
from mjlab.viewer.viser.scene_annotations import ViserSceneAnnotationsMixin
from mjlab.viewer.viser.scene_contacts import ViserSceneContactsMixin, _Contact
from mjlab.viewer.viser.scene_debug import ViserSceneDebugMixin
from mjlab.viewer.viser.scene_gui import ViserSceneGuiMixin

# Viser visualization defaults.
_DEFAULT_ENVIRONMENT_INTENSITY = 0.8
_DEFAULT_CONTACT_POINT_COLOR = (230, 153, 51)
_DEFAULT_CONTACT_FORCE_COLOR = (255, 0, 0)
_NUM_GEOM_GROUPS = 6


@dataclass
class ViserMujocoScene(
  ViserSceneGuiMixin,
  ViserSceneAnnotationsMixin,
  ViserSceneContactsMixin,
  ViserSceneDebugMixin,
  DebugVisualizer,
):
  """Manages Viser scene handles and visualization state for MuJoCo models.

  Also implements DebugVisualizer protocol for environment-specific annotations
  like arrows, ghost meshes, and coordinate frames.
  """

  server: viser.ViserServer
  mj_model: mujoco.MjModel
  mj_data: mujoco.MjData
  num_envs: int

  fixed_bodies_frame: viser.SceneNodeHandle = field(init=False)
  mesh_handles_by_group: dict[tuple[int, int], viser.BatchedGlbHandle] = field(
    default_factory=dict
  )
  fixed_site_handles: dict[tuple[int, int], viser.GlbHandle] = field(
    default_factory=dict
  )
  site_handles_by_group: dict[tuple[int, int], viser.BatchedGlbHandle] = field(
    default_factory=dict
  )
  contact_point_handle: viser.BatchedMeshHandle | None = None
  contact_force_shaft_handle: viser.BatchedMeshHandle | None = None
  contact_force_head_handle: viser.BatchedMeshHandle | None = None

  _label_handles: dict[str, viser.LabelHandle] = field(default_factory=dict, init=False)
  _frame_handles: dict[str, viser.FrameHandle] = field(default_factory=dict, init=False)

  env_idx: int = 0
  camera_tracking_enabled: bool = False
  show_only_selected: bool = False
  geom_groups_visible: list[bool] = field(
    default_factory=lambda: [True, True, True, False, False, False]
  )
  site_groups_visible: list[bool] = field(
    default_factory=lambda: [True, True, True, False, False, False]
  )
  label_targets: set[str] = field(default_factory=set)
  frame_targets: set[str] = field(default_factory=set)
  frame_scale: float = 1.0
  show_contact_points: bool = False
  show_contact_forces: bool = False
  contact_point_color: tuple[int, int, int] = _DEFAULT_CONTACT_POINT_COLOR
  contact_force_color: tuple[int, int, int] = _DEFAULT_CONTACT_FORCE_COLOR
  meansize_override: float | None = None
  needs_update: bool = False
  _tracked_body_id: int | None = field(init=False, default=None)

  _last_body_xpos: np.ndarray | None = None
  _last_body_xmat: np.ndarray | None = None
  _last_mocap_pos: np.ndarray | None = None
  _last_mocap_quat: np.ndarray | None = None
  _last_env_idx: int = 0
  _last_contacts: list[_Contact] | None = None

  debug_visualization_enabled: bool = False
  _scene_offset: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False)
  _queued_arrows: list[
    tuple[np.ndarray, np.ndarray, tuple[float, float, float, float], float]
  ] = field(default_factory=list, init=False)
  _arrow_shaft_handle: viser.BatchedMeshHandle | None = field(default=None, init=False)
  _arrow_head_handle: viser.BatchedMeshHandle | None = field(default=None, init=False)
  _ghost_handles: dict[int, viser.SceneNodeHandle] = field(
    default_factory=dict, init=False
  )
  _ghost_meshes: dict[int, dict[int, trimesh.Trimesh]] = field(
    default_factory=dict, init=False
  )
  _arrow_shaft_mesh: trimesh.Trimesh | None = field(default=None, init=False)
  _arrow_head_mesh: trimesh.Trimesh | None = field(default=None, init=False)
  _viz_data: mujoco.MjData = field(init=False)

  @staticmethod
  def create(
    server: viser.ViserServer,
    mj_model: mujoco.MjModel,
    num_envs: int,
  ) -> ViserMujocoScene:
    """Create and populate scene with geometry.

    Visual geometry is created immediately. Collision geometry is created
    lazily when first needed.

    Args:
      server: Viser server instance.
      mj_model: MuJoCo model.
      num_envs: Number of parallel environments.

    Returns:
      ViserMujocoScene instance with scene populated.
    """
    mj_data = mujoco.MjData(mj_model)

    scene = ViserMujocoScene(
      server=server,
      mj_model=mj_model,
      mj_data=mj_data,
      num_envs=num_envs,
    )

    scene._viz_data = mujoco.MjData(mj_model)

    server.scene.configure_environment_map(
      environment_intensity=_DEFAULT_ENVIRONMENT_INTENSITY
    )

    scene.fixed_bodies_frame = server.scene.add_frame("/fixed_bodies", show_axes=False)
    scene._add_fixed_geometry()
    scene._create_mesh_handles_by_group()
    scene._create_site_handles_by_group()

    # Find first non-fixed body for camera tracking.
    for body_id in range(mj_model.nbody):
      if not is_fixed_body(mj_model, body_id):
        scene._tracked_body_id = body_id
        break

    return scene

  def _is_collision_geom(self, geom_id: int) -> bool:
    """Check if a geom is a collision geom."""
    return (
      self.mj_model.geom_contype[geom_id] != 0
      or self.mj_model.geom_conaffinity[geom_id] != 0
    )

  def _sync_visibilities(self) -> None:
    """Synchronize all handle visibilities based on current flags."""
    for (_body_id, group_id), handle in self.mesh_handles_by_group.items():
      handle.visible = (
        group_id < _NUM_GEOM_GROUPS and self.geom_groups_visible[group_id]
      )

    for (_body_id, group_id), handle in self.fixed_site_handles.items():
      handle.visible = (
        group_id < _NUM_GEOM_GROUPS and self.site_groups_visible[group_id]
      )

    for (_body_id, group_id), handle in self.site_handles_by_group.items():
      handle.visible = (
        group_id < _NUM_GEOM_GROUPS and self.site_groups_visible[group_id]
      )

    if self.contact_point_handle is not None and not self.show_contact_points:
      self.contact_point_handle.visible = False

    if not self.show_contact_forces:
      if self.contact_force_shaft_handle is not None:
        self.contact_force_shaft_handle.visible = False
      if self.contact_force_head_handle is not None:
        self.contact_force_head_handle.visible = False

  def update(self, wp_data, env_idx: int | None = None) -> None:
    """Update scene from batched simulation data.

    Args:
      wp_data: Batched Warp simulation data (mjwarp.Data).
      env_idx: Environment index to visualize. If None, uses self.env_idx.
    """
    if env_idx is None:
      env_idx = self.env_idx

    body_xpos = wp_data.xpos.numpy()
    body_xmat = wp_data.xmat.numpy()
    mocap_pos = wp_data.mocap_pos.numpy()
    mocap_quat = wp_data.mocap_quat.numpy()
    scene_offset = np.zeros(3)
    if self.camera_tracking_enabled and self._tracked_body_id is not None:
      tracked_pos = body_xpos[env_idx, self._tracked_body_id, :].copy()
      scene_offset = -tracked_pos

    contacts = None
    mj_data = None
    if (
      self.show_contact_points
      or self.show_contact_forces
      or self.label_targets
      or self.frame_targets
    ):
      self.mj_data.qpos[:] = wp_data.qpos.numpy()[env_idx]
      self.mj_data.qvel[:] = wp_data.qvel.numpy()[env_idx]
      self.mj_data.mocap_pos[:] = mocap_pos[env_idx]
      self.mj_data.mocap_quat[:] = mocap_quat[env_idx]
      mujoco.mj_forward(self.mj_model, self.mj_data)
      mj_data = self.mj_data
      if self.show_contact_points or self.show_contact_forces:
        contacts = self._extract_contacts_from_mjdata(self.mj_data)

    self._update_visualization(
      body_xpos,
      body_xmat,
      mocap_pos,
      mocap_quat,
      env_idx,
      scene_offset,
      contacts,
      mj_data,
    )

    # Update scene offset for debug visualizations and sync arrows
    if self.debug_visualization_enabled:
      self._scene_offset = scene_offset
      self._sync_arrows()

  def update_from_mjdata(self, mj_data: mujoco.MjData) -> None:
    """Update scene from single-environment MuJoCo data.

    Args:
      mj_data: Single environment MuJoCo data.
    """
    body_xpos = mj_data.xpos[None, ...]
    body_xmat = mj_data.xmat.reshape(-1, 3, 3)[None, ...]
    mocap_pos = mj_data.mocap_pos[None, ...]
    mocap_quat = mj_data.mocap_quat[None, ...]
    env_idx = 0
    scene_offset = np.zeros(3)
    if self.camera_tracking_enabled and self._tracked_body_id is not None:
      tracked_pos = mj_data.xpos[self._tracked_body_id, :].copy()
      scene_offset = -tracked_pos

    # Always extract contacts for single-environment updates (used by nan_viz).
    contacts = self._extract_contacts_from_mjdata(mj_data)

    self._update_visualization(
      body_xpos,
      body_xmat,
      mocap_pos,
      mocap_quat,
      env_idx,
      scene_offset,
      contacts,
      mj_data,
    )

    if self.debug_visualization_enabled:
      self._scene_offset = scene_offset
      self._sync_arrows()

  def _update_visualization(
    self,
    body_xpos: np.ndarray,
    body_xmat: np.ndarray,
    mocap_pos: np.ndarray,
    mocap_quat: np.ndarray,
    env_idx: int,
    scene_offset: np.ndarray,
    contacts: list[_Contact] | None,
    mj_data: mujoco.MjData | None,
  ) -> None:
    """Shared visualization update logic."""
    self._last_body_xpos = body_xpos
    self._last_body_xmat = body_xmat
    self._last_mocap_pos = mocap_pos
    self._last_mocap_quat = mocap_quat
    self._last_env_idx = env_idx
    self._scene_offset = scene_offset
    if contacts is not None:
      self._last_contacts = contacts

    self.fixed_bodies_frame.position = scene_offset
    with self.server.atomic():
      body_xquat = vtf.SO3.from_matrix(body_xmat).wxyz
      for (body_id, _group_id), handle in self.mesh_handles_by_group.items():
        if not handle.visible:
          continue
        mocap_id = self.mj_model.body_mocapid[body_id]
        if mocap_id >= 0:
          if self.show_only_selected and self.num_envs > 1:
            single_pos = mocap_pos[env_idx, mocap_id, :] + scene_offset
            single_quat = mocap_quat[env_idx, mocap_id, :]
            handle.batched_positions = np.tile(single_pos[None, :], (self.num_envs, 1))
            handle.batched_wxyzs = np.tile(single_quat[None, :], (self.num_envs, 1))
          else:
            handle.batched_positions = mocap_pos[:, mocap_id, :] + scene_offset
            handle.batched_wxyzs = mocap_quat[:, mocap_id, :]
        else:
          if self.show_only_selected and self.num_envs > 1:
            single_pos = body_xpos[env_idx, body_id, :] + scene_offset
            single_quat = body_xquat[env_idx, body_id, :]
            handle.batched_positions = np.tile(single_pos[None, :], (self.num_envs, 1))
            handle.batched_wxyzs = np.tile(single_quat[None, :], (self.num_envs, 1))
          else:
            handle.batched_positions = body_xpos[..., body_id, :] + scene_offset
            handle.batched_wxyzs = body_xquat[..., body_id, :]

      for (body_id, _group_id), handle in self.site_handles_by_group.items():
        if not handle.visible:
          continue
        mocap_id = self.mj_model.body_mocapid[body_id]
        if mocap_id >= 0:
          if self.show_only_selected and self.num_envs > 1:
            single_pos = mocap_pos[env_idx, mocap_id, :] + scene_offset
            single_quat = mocap_quat[env_idx, mocap_id, :]
            handle.batched_positions = np.tile(single_pos[None, :], (self.num_envs, 1))
            handle.batched_wxyzs = np.tile(single_quat[None, :], (self.num_envs, 1))
          else:
            handle.batched_positions = mocap_pos[:, mocap_id, :] + scene_offset
            handle.batched_wxyzs = mocap_quat[:, mocap_id, :]
        else:
          if self.show_only_selected and self.num_envs > 1:
            single_pos = body_xpos[env_idx, body_id, :] + scene_offset
            single_quat = body_xquat[env_idx, body_id, :]
            handle.batched_positions = np.tile(single_pos[None, :], (self.num_envs, 1))
            handle.batched_wxyzs = np.tile(single_quat[None, :], (self.num_envs, 1))
          else:
            handle.batched_positions = body_xpos[..., body_id, :] + scene_offset
            handle.batched_wxyzs = body_xquat[..., body_id, :]

      if contacts is not None:
        self._update_contact_visualization(contacts, scene_offset)

      self._update_annotations(body_xpos, body_xmat, env_idx, scene_offset, mj_data)

      self.server.flush()

  def _request_update(self) -> None:
    """Request a visualization update and trigger immediate re-render from cache.

    This is called when visualization settings change to provide immediate feedback.
    For viewers with continuous update loops (viser_play), the loop will refresh soon.
    For static viewers (nan_viz), this provides the only update mechanism.
    """
    self.needs_update = True
    self.refresh_visualization()

  def refresh_visualization(self) -> None:
    """Re-render the scene using cached visualization data.

    This is useful when visualization settings change (e.g., toggling contacts)
    but the underlying simulation data hasn't changed. Clears the needs_update flag.
    """
    if (
      self._last_body_xpos is None
      or self._last_body_xmat is None
      or self._last_mocap_pos is None
      or self._last_mocap_quat is None
    ):
      return

    contacts = (
      self._last_contacts
      if (self.show_contact_points or self.show_contact_forces)
      else None
    )

    scene_offset = np.zeros(3)
    if self.camera_tracking_enabled and self._tracked_body_id is not None:
      tracked_pos = self._last_body_xpos[
        self._last_env_idx, self._tracked_body_id, :
      ].copy()
      scene_offset = -tracked_pos

    self._update_visualization(
      self._last_body_xpos,
      self._last_body_xmat,
      self._last_mocap_pos,
      self._last_mocap_quat,
      self._last_env_idx,
      scene_offset,
      contacts,
      None,
    )
    self.needs_update = False

  def _add_fixed_geometry(self) -> None:
    """Add fixed world geometry to the scene."""
    body_geoms_visual: dict[int, list[int]] = {}
    body_geoms_collision: dict[int, list[int]] = {}

    for i in range(self.mj_model.ngeom):
      body_id = self.mj_model.geom_bodyid[i]
      target = body_geoms_collision if self._is_collision_geom(i) else body_geoms_visual
      target.setdefault(body_id, []).append(i)

    all_bodies = set(body_geoms_visual.keys()) | set(body_geoms_collision.keys())

    for body_id in all_bodies:
      body_name = get_body_name(self.mj_model, body_id)

      if is_fixed_body(self.mj_model, body_id):
        # Create both visual and collision geoms for fixed bodies (terrain, floor, etc.)
        # but show them all since they're static.
        all_geoms = []
        if body_id in body_geoms_visual:
          all_geoms.extend(body_geoms_visual[body_id])
        if body_id in body_geoms_collision:
          all_geoms.extend(body_geoms_collision[body_id])

        if not all_geoms:
          continue

        nonplane_geom_ids: list[int] = []
        for geom_id in all_geoms:
          geom_type = self.mj_model.geom_type[geom_id]
          if geom_type == mjtGeom.mjGEOM_PLANE:
            geom_name = mj_id2name(self.mj_model, mjtObj.mjOBJ_GEOM, geom_id)
            self.server.scene.add_grid(
              f"/fixed_bodies/{body_name}/{geom_name}",
              # For infinite grids in viser 1.0.10, the width and height
              # parameters determined the region of the grid that can
              # receive shadows. We'll just make this really big for now.
              # In a future release of Viser these two args should ideally be
              # unnecessary.
              width=2000.0,
              height=2000.0,
              infinite_grid=True,
              fade_distance=50.0,
              shadow_opacity=0.2,
              position=self.mj_model.geom_pos[geom_id],
              wxyz=self.mj_model.geom_quat[geom_id],
            )
          else:
            nonplane_geom_ids.append(geom_id)

        if len(nonplane_geom_ids) > 0:
          self.server.scene.add_mesh_trimesh(
            f"/fixed_bodies/{body_name}",
            merge_geoms(self.mj_model, nonplane_geom_ids),
            cast_shadow=False,
            receive_shadow=0.2,
            position=self.mj_model.body(body_id).pos,
            wxyz=self.mj_model.body(body_id).quat,
            visible=True,
          )

    body_group_sites: dict[tuple[int, int], list[int]] = {}
    for i in range(self.mj_model.nsite):
      body_id = self.mj_model.site_bodyid[i]
      if is_fixed_body(self.mj_model, body_id):
        site_group = self.mj_model.site_group[i]
        key = (body_id, site_group)
        body_group_sites.setdefault(key, []).append(i)

    for (body_id, group_id), site_ids in body_group_sites.items():
      body_name = get_body_name(self.mj_model, body_id)
      visible = group_id < _NUM_GEOM_GROUPS and self.site_groups_visible[group_id]
      handle = self.server.scene.add_mesh_trimesh(
        f"/fixed_bodies/{body_name}/sites_group{group_id}",
        merge_sites(self.mj_model, site_ids),
        cast_shadow=False,
        receive_shadow=0.2,
        position=self.mj_model.body(body_id).pos,
        wxyz=self.mj_model.body(body_id).quat,
        visible=visible,
      )
      self.fixed_site_handles[(body_id, group_id)] = handle

  def _create_mesh_handles_by_group(self) -> None:
    """Create mesh handles for each geom group separately to allow independent toggling."""
    body_group_geoms: dict[tuple[int, int], list[int]] = {}

    for i in range(self.mj_model.ngeom):
      body_id = self.mj_model.geom_bodyid[i]

      if is_fixed_body(self.mj_model, body_id):
        continue

      geom_group = self.mj_model.geom_group[i]
      key = (body_id, geom_group)

      if key not in body_group_geoms:
        body_group_geoms[key] = []
      body_group_geoms[key].append(i)

    with self.server.atomic():
      for (body_id, group_id), geom_indices in body_group_geoms.items():
        body_name = get_body_name(self.mj_model, body_id)

        mesh = merge_geoms(self.mj_model, geom_indices)
        lod_ratio = 1000.0 / mesh.vertices.shape[0]

        visible = group_id < _NUM_GEOM_GROUPS and self.geom_groups_visible[group_id]

        handle = self.server.scene.add_batched_meshes_trimesh(
          f"/bodies/{body_name}/group{group_id}",
          mesh,
          batched_wxyzs=np.array([1.0, 0.0, 0.0, 0.0])[None].repeat(
            self.num_envs, axis=0
          ),
          batched_positions=np.array([0.0, 0.0, 0.0])[None].repeat(
            self.num_envs, axis=0
          ),
          lod=((2.0, lod_ratio),) if lod_ratio < 0.5 else "off",
          visible=visible,
        )
        self.mesh_handles_by_group[(body_id, group_id)] = handle

  def _create_site_handles_by_group(self) -> None:
    """Create site handles for each site group on non-fixed bodies."""
    body_group_sites: dict[tuple[int, int], list[int]] = {}

    for i in range(self.mj_model.nsite):
      body_id = self.mj_model.site_bodyid[i]

      if is_fixed_body(self.mj_model, body_id):
        continue

      site_group = self.mj_model.site_group[i]
      key = (body_id, site_group)
      body_group_sites.setdefault(key, []).append(i)

    with self.server.atomic():
      for (body_id, group_id), site_indices in body_group_sites.items():
        body_name = get_body_name(self.mj_model, body_id)

        mesh = merge_sites(self.mj_model, site_indices)
        lod_ratio = 1000.0 / mesh.vertices.shape[0]

        visible = group_id < _NUM_GEOM_GROUPS and self.site_groups_visible[group_id]

        handle = self.server.scene.add_batched_meshes_trimesh(
          f"/bodies/{body_name}/sites_group{group_id}",
          mesh,
          batched_wxyzs=np.array([1.0, 0.0, 0.0, 0.0])[None].repeat(
            self.num_envs, axis=0
          ),
          batched_positions=np.array([0.0, 0.0, 0.0])[None].repeat(
            self.num_envs, axis=0
          ),
          lod=((2.0, lod_ratio),) if lod_ratio < 0.5 else "off",
          visible=visible,
        )
        self.site_handles_by_group[(body_id, group_id)] = handle
