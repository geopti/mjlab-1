"""Manages all Viser visualization handles and state for MuJoCo models."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
import torch
import trimesh
import trimesh.visual
import viser
import viser.transforms as vtf
from mujoco import mj_id2name, mjtGeom, mjtObj
from typing_extensions import override

from mjlab.viewer.debug_visualizer import DebugVisualizer
from mjlab.viewer.viser.conversions import (
  create_primitive_mesh,
  get_body_name,
  is_fixed_body,
  merge_geoms,
  merge_sites,
  mujoco_mesh_to_trimesh,
  rotation_matrix_from_vectors,
  rotation_quat_from_vectors,
)

try:
  import mujoco_warp as mjwarp
except ImportError:
  mjwarp = None  # type: ignore


# Viser visualization defaults.
_DEFAULT_FOV_DEGREES = 60
_DEFAULT_FOV_MIN = 20
_DEFAULT_FOV_MAX = 150
_DEFAULT_ENVIRONMENT_INTENSITY = 0.8
_DEFAULT_CONTACT_POINT_COLOR = (230, 153, 51)
_DEFAULT_CONTACT_FORCE_COLOR = (255, 0, 0)


@dataclass
class _Contact:
  """Contact data from MuJoCo."""

  pos: np.ndarray
  frame: np.ndarray  # 3x3 rotation matrix.
  force: np.ndarray  # Force in contact frame.
  dist: float
  included: bool


@dataclass
class _ContactPointVisual:
  """Visual representation data for a contact point."""

  position: np.ndarray
  orientation: np.ndarray  # Quaternion (wxyz).
  scale: np.ndarray  # [width, width, height].


@dataclass
class _ContactForceVisual:
  """Visual representation data for a contact force arrow."""

  shaft_position: np.ndarray
  shaft_orientation: np.ndarray  # Quaternion (wxyz).
  shaft_scale: np.ndarray  # [width, width, length].
  head_position: np.ndarray
  head_orientation: np.ndarray  # Quaternion (wxyz).
  head_scale: np.ndarray  # [width, width, width].


@dataclass
class ViserMujocoScene(DebugVisualizer):
  """Manages Viser scene handles and visualization state for MuJoCo models.

  Also implements DebugVisualizer protocol for environment-specific annotations
  like arrows, ghost meshes, and coordinate frames.
  """

  # Core.
  server: viser.ViserServer
  mj_model: mujoco.MjModel
  mj_data: mujoco.MjData
  num_envs: int

  # Handles (created once).
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

  # Annotation handles (labels and frames for sites/bodies).
  _label_handles: dict[str, viser.LabelHandle] = field(default_factory=dict, init=False)
  _frame_handles: dict[str, viser.FrameHandle] = field(default_factory=dict, init=False)

  # Visualization settings (set directly or automatically updated by create_options_gui).
  env_idx: int = 0  # Current environment index (DebugVisualizer protocol).
  camera_tracking_enabled: bool = False
  show_only_selected: bool = False
  geom_groups_visible: list[bool] = field(
    default_factory=lambda: [True, True, True, False, False, False]
  )
  site_groups_visible: list[bool] = field(
    default_factory=lambda: [True, True, True, False, False, False]
  )
  # Annotation targets: which object types to show labels/frames for.
  label_targets: set[str] = field(default_factory=set)  # "sites", "bodies"
  frame_targets: set[str] = field(default_factory=set)  # "sites", "bodies"
  frame_scale: float = 1.0  # Multiplier for coordinate frame size
  show_contact_points: bool = False
  show_contact_forces: bool = False
  contact_point_color: tuple[int, int, int] = _DEFAULT_CONTACT_POINT_COLOR
  contact_force_color: tuple[int, int, int] = _DEFAULT_CONTACT_FORCE_COLOR
  meansize_override: float | None = None
  needs_update: bool = False
  _tracked_body_id: int | None = field(init=False, default=None)

  # Cached visualization state for re-rendering when settings change.
  _last_body_xpos: np.ndarray | None = None
  _last_body_xmat: np.ndarray | None = None
  _last_mocap_pos: np.ndarray | None = None
  _last_mocap_quat: np.ndarray | None = None
  _last_env_idx: int = 0
  _last_contacts: list[_Contact] | None = None

  # Debug visualization (arrows, ghosts, frames).
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

    # Initialize debug visualization data.
    scene._viz_data = mujoco.MjData(mj_model)

    # Configure environment lighting.
    server.scene.configure_environment_map(
      environment_intensity=_DEFAULT_ENVIRONMENT_INTENSITY
    )

    # Create frame for fixed world geometry.
    scene.fixed_bodies_frame = server.scene.add_frame("/fixed_bodies", show_axes=False)

    # Add fixed geometry (planes, terrain, etc.).
    scene._add_fixed_geometry()

    # Create mesh handles per geom group.
    scene._create_mesh_handles_by_group()

    # Create site handles per site group.
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
    # Geom group meshes.
    for (_body_id, group_id), handle in self.mesh_handles_by_group.items():
      handle.visible = group_id < 6 and self.geom_groups_visible[group_id]

    # Site group meshes (fixed bodies).
    for (_body_id, group_id), handle in self.fixed_site_handles.items():
      handle.visible = group_id < 6 and self.site_groups_visible[group_id]

    # Site group meshes (non-fixed bodies).
    for (_body_id, group_id), handle in self.site_handles_by_group.items():
      handle.visible = group_id < 6 and self.site_groups_visible[group_id]

    # Contact points.
    if self.contact_point_handle is not None and not self.show_contact_points:
      self.contact_point_handle.visible = False

    # Contact forces.
    if not self.show_contact_forces:
      if self.contact_force_shaft_handle is not None:
        self.contact_force_shaft_handle.visible = False
      if self.contact_force_head_handle is not None:
        self.contact_force_head_handle.visible = False

  def _clear_annotations(self) -> None:
    """Remove all annotation handles (labels and frames)."""
    for handle in self._label_handles.values():
      handle.remove()
    self._label_handles.clear()

    for handle in self._frame_handles.values():
      handle.remove()
    self._frame_handles.clear()

  def _refresh_annotations(self) -> None:
    """Recreate all annotations based on current label_targets and frame_targets."""
    self._clear_annotations()

    # Create site labels.
    if "sites" in self.label_targets:
      for site_id in range(self.mj_model.nsite):
        site_group = self.mj_model.site_group[site_id]
        if site_group >= 6 or not self.site_groups_visible[site_group]:
          continue
        site_name = mj_id2name(self.mj_model, mjtObj.mjOBJ_SITE, site_id)
        if not site_name:
          site_name = f"site_{site_id}"
        label = self.server.scene.add_label(
          f"/annotations/labels/site_{site_id}",
          site_name,
          wxyz=(1.0, 0.0, 0.0, 0.0),
          position=(0.0, 0.0, 0.0),
        )
        self._label_handles[f"site_{site_id}"] = label

    # Create body labels.
    if "bodies" in self.label_targets:
      for body_id in range(self.mj_model.nbody):
        if is_fixed_body(self.mj_model, body_id):
          continue
        body_name = get_body_name(self.mj_model, body_id)
        label = self.server.scene.add_label(
          f"/annotations/labels/body_{body_id}",
          body_name,
          wxyz=(1.0, 0.0, 0.0, 0.0),
          position=(0.0, 0.0, 0.0),
        )
        self._label_handles[f"body_{body_id}"] = label

    # Create site frames.
    if "sites" in self.frame_targets:
      self._create_frame_handles("sites")

    # Create body frames.
    if "bodies" in self.frame_targets:
      self._create_frame_handles("bodies")

  def _create_frame_handles(self, target: str) -> None:
    """Create coordinate frame visualization handles for the given target type."""
    meansize = self.mj_model.stat.meansize
    frame_length = self.mj_model.vis.scale.framelength * meansize * self.frame_scale
    frame_width = self.mj_model.vis.scale.framewidth * meansize * self.frame_scale

    # Create frames for each site/body.
    if target == "sites":
      for site_id in range(self.mj_model.nsite):
        site_group = self.mj_model.site_group[site_id]
        if site_group >= 6 or not self.site_groups_visible[site_group]:
          continue
        key = f"site_frame_{site_id}"
        handle = self.server.scene.add_frame(
          f"/annotations/frames/{key}",
          axes_length=frame_length,
          axes_radius=frame_width,
        )
        self._frame_handles[key] = handle

    else:  # bodies
      for body_id in range(self.mj_model.nbody):
        if is_fixed_body(self.mj_model, body_id):
          continue
        key = f"body_frame_{body_id}"
        handle = self.server.scene.add_frame(
          f"/annotations/frames/{key}",
          axes_length=frame_length,
          axes_radius=frame_width,
        )
        self._frame_handles[key] = handle

  def _update_annotations(
    self,
    body_xpos: np.ndarray,
    body_xmat: np.ndarray,
    env_idx: int,
    scene_offset: np.ndarray,
  ) -> None:
    """Update positions of all annotations for the selected environment."""
    if not self.label_targets and not self.frame_targets:
      return

    body_xquat = vtf.SO3.from_matrix(body_xmat).wxyz

    # Update site labels.
    if "sites" in self.label_targets:
      for site_id in range(self.mj_model.nsite):
        key = f"site_{site_id}"
        if key not in self._label_handles:
          continue
        body_id = self.mj_model.site_bodyid[site_id]
        site_pos_local = self.mj_model.site_pos[site_id]

        if is_fixed_body(self.mj_model, body_id):
          body_pos = self.mj_model.body(body_id).pos
          body_quat = self.mj_model.body(body_id).quat
          site_world_pos = body_pos + vtf.SO3(body_quat).as_matrix() @ site_pos_local
        else:
          body_pos = body_xpos[env_idx, body_id, :]
          body_rot = vtf.SO3(body_xquat[env_idx, body_id, :]).as_matrix()
          site_world_pos = body_pos + body_rot @ site_pos_local

        self._label_handles[key].position = site_world_pos + scene_offset

    # Update body labels.
    if "bodies" in self.label_targets:
      for body_id in range(self.mj_model.nbody):
        key = f"body_{body_id}"
        if key not in self._label_handles:
          continue
        body_pos = body_xpos[env_idx, body_id, :] + scene_offset
        self._label_handles[key].position = body_pos

    # Update site frames.
    if "sites" in self.frame_targets:
      self._update_frame_positions("sites", body_xpos, body_xmat, env_idx, scene_offset)

    # Update body frames.
    if "bodies" in self.frame_targets:
      self._update_frame_positions(
        "bodies", body_xpos, body_xmat, env_idx, scene_offset
      )

  def _update_frame_positions(
    self,
    target: str,
    body_xpos: np.ndarray,
    body_xmat: np.ndarray,
    env_idx: int,
    scene_offset: np.ndarray,
  ) -> None:
    """Update frame handle positions and orientations."""
    body_xquat = vtf.SO3.from_matrix(body_xmat).wxyz

    if target == "sites":
      for site_id in range(self.mj_model.nsite):
        key = f"site_frame_{site_id}"
        if key not in self._frame_handles:
          continue

        body_id = self.mj_model.site_bodyid[site_id]
        site_pos_local = self.mj_model.site_pos[site_id]
        site_quat_local = self.mj_model.site_quat[site_id]

        if is_fixed_body(self.mj_model, body_id):
          body_pos = self.mj_model.body(body_id).pos
          body_quat = self.mj_model.body(body_id).quat
          body_rot = vtf.SO3(body_quat)
        else:
          body_pos = body_xpos[env_idx, body_id, :]
          body_rot = vtf.SO3(body_xquat[env_idx, body_id, :])

        site_world_pos = body_pos + body_rot.as_matrix() @ site_pos_local + scene_offset
        site_world_quat = (body_rot @ vtf.SO3(site_quat_local)).wxyz

        handle = self._frame_handles[key]
        handle.position = site_world_pos
        handle.wxyz = site_world_quat

    else:  # bodies
      for body_id in range(self.mj_model.nbody):
        key = f"body_frame_{body_id}"
        if key not in self._frame_handles:
          continue

        body_pos = body_xpos[env_idx, body_id, :] + scene_offset
        body_quat = body_xquat[env_idx, body_id, :]

        handle = self._frame_handles[key]
        handle.position = body_pos
        handle.wxyz = body_quat

  def create_env_selector_gui(self) -> None:
    """Add environment selector at top level (always visible across all tabs).

    Should be called before creating the tab group.
    """
    if self.num_envs > 1:
      env_slider = self.server.gui.add_slider(
        "Environment",
        min=0,
        max=self.num_envs - 1,
        step=1,
        initial_value=self.env_idx,
        hint=f"Select environment (0-{self.num_envs - 1})",
      )

      @env_slider.on_update
      def _(_) -> None:
        self.env_idx = int(env_slider.value)
        self._request_update()

  def create_visualization_gui(
    self,
    camera_distance: float = 3.0,
    camera_azimuth: float = 45.0,
    camera_elevation: float = 30.0,
    show_debug_viz_control: bool = True,
  ) -> None:
    """Add standard GUI controls that automatically update this scene's settings.

    Args:
      camera_distance: Default camera distance from tracked body when tracking is enabled.
      camera_azimuth: Default camera azimuth angle in degrees.
      camera_elevation: Default camera elevation angle in degrees.
      show_debug_viz_control: Whether to show the debug visualization checkbox.
    """
    with self.server.gui.add_folder("Camera"):
      slider_fov = self.server.gui.add_slider(
        "FOV (°)",
        min=_DEFAULT_FOV_MIN,
        max=_DEFAULT_FOV_MAX,
        step=1,
        initial_value=_DEFAULT_FOV_DEGREES,
        hint="Vertical FOV of viewer camera, in degrees.",
      )

      @slider_fov.on_update
      def _(_) -> None:
        for client in self.server.get_clients().values():
          client.camera.fov = np.radians(slider_fov.value)

      @self.server.on_client_connect
      def _(client: viser.ClientHandle) -> None:
        client.camera.fov = np.radians(slider_fov.value)

      # Camera tracking controls.
      cb_camera_tracking = self.server.gui.add_checkbox(
        "Track body",
        initial_value=self.camera_tracking_enabled,
        hint="Keep tracked body centered. Use Viser camera controls to adjust view.",
      )

      if self.num_envs > 1:
        show_only_cb = self.server.gui.add_checkbox(
          "Hide other envs",
          initial_value=self.show_only_selected,
          hint="Show only the selected environment.",
        )

        @show_only_cb.on_update
        def _(_) -> None:
          self.show_only_selected = show_only_cb.value
          self._request_update()

      @cb_camera_tracking.on_update
      def _(_) -> None:
        self.camera_tracking_enabled = cb_camera_tracking.value
        # Snap camera to default view when enabling tracking.
        if self.camera_tracking_enabled:
          # Convert to radians and calculate camera position.
          azimuth_rad = np.deg2rad(camera_azimuth)
          elevation_rad = np.deg2rad(camera_elevation)

          # Calculate forward vector from spherical coordinates.
          forward = np.array(
            [
              np.cos(elevation_rad) * np.cos(azimuth_rad),
              np.cos(elevation_rad) * np.sin(azimuth_rad),
              np.sin(elevation_rad),
            ]
          )

          # Camera position is origin - forward * distance.
          camera_pos = -forward * camera_distance

          # Snap all connected clients to this view.
          for client in self.server.get_clients().values():
            client.camera.position = camera_pos
            client.camera.look_at = np.zeros(3)

        self._request_update()

      # Debug visualization controls (only show if requested).
      if show_debug_viz_control:
        cb_debug_vis = self.server.gui.add_checkbox(
          "Debug visualization",
          initial_value=self.debug_visualization_enabled,
          hint="Show debug arrows and ghost meshes.",
        )

        @cb_debug_vis.on_update
        def _(_) -> None:
          self.debug_visualization_enabled = cb_debug_vis.value
          # Clear visualizer if hiding.
          if not self.debug_visualization_enabled:
            self.clear_debug_all()
          self._request_update()

  def create_groups_gui(self, tabs) -> None:
    """Add groups tab combining geom and site visibility controls.

    Args:
      tabs: The viser tab group to add the groups tab to.
    """
    with tabs.add_tab("Groups", icon=viser.Icon.EYE):
      self.server.gui.add_markdown("**Geoms**")
      for i in range(6):
        cb = self.server.gui.add_checkbox(
          f"G{i}",
          initial_value=self.geom_groups_visible[i],
          hint=f"Show/hide geoms in group {i}",
        )

        @cb.on_update
        def _(event, group_idx=i) -> None:
          self.geom_groups_visible[group_idx] = event.target.value
          self._sync_visibilities()
          self._request_update()

      self.server.gui.add_markdown("**Sites**")
      for i in range(6):
        cb = self.server.gui.add_checkbox(
          f"S{i}",
          initial_value=self.site_groups_visible[i],
          hint=f"Show/hide sites in group {i}",
        )

        @cb.on_update
        def _(event, group_idx=i) -> None:
          self.site_groups_visible[group_idx] = event.target.value
          self._sync_visibilities()
          # Refresh annotations if any site annotations are enabled.
          if "sites" in self.label_targets or "sites" in self.frame_targets:
            self._refresh_annotations()
            self._request_update()

  def create_overlays_gui(self, tabs) -> None:
    """Add overlays tab combining annotations and contacts.

    Args:
      tabs: The viser tab group to add the overlays tab to.
    """
    with tabs.add_tab("Overlays", icon=viser.Icon.LAYERS_LINKED):
      # Labels section.
      self.server.gui.add_markdown("**Labels**")
      cb_site_labels = self.server.gui.add_checkbox(
        "Sites",
        initial_value="sites" in self.label_targets,
        hint="Show text labels for visible sites",
      )
      cb_body_labels = self.server.gui.add_checkbox(
        "Bodies",
        initial_value="bodies" in self.label_targets,
        hint="Show text labels for bodies",
      )

      # Frames section.
      self.server.gui.add_markdown("**Frames**")
      cb_site_frames = self.server.gui.add_checkbox(
        "Sites",
        initial_value="sites" in self.frame_targets,
        hint="Show coordinate frames for visible sites",
      )
      cb_body_frames = self.server.gui.add_checkbox(
        "Bodies",
        initial_value="bodies" in self.frame_targets,
        hint="Show coordinate frames for bodies",
      )
      frame_scale_input = self.server.gui.add_slider(
        "Frame scale",
        min=0.1,
        max=5.0,
        step=0.1,
        initial_value=self.frame_scale,
        hint="Scale multiplier for coordinate frames",
      )

      # Contacts section.
      self.server.gui.add_markdown("**Contacts**")
      cb_contact_points = self.server.gui.add_checkbox(
        "Points",
        initial_value=self.show_contact_points,
        hint="Toggle contact point visualization.",
      )
      cb_contact_forces = self.server.gui.add_checkbox(
        "Forces",
        initial_value=self.show_contact_forces,
        hint="Toggle contact force visualization.",
      )
      contact_scale_input = self.server.gui.add_number(
        "Contact scale",
        step=self.mj_model.stat.meansize * 0.01,
        initial_value=self.meansize_override or self.mj_model.stat.meansize,
        hint="Scale for contact visualization",
      )

      # Labels callbacks.
      @cb_site_labels.on_update
      def _(_) -> None:
        if cb_site_labels.value:
          self.label_targets.add("sites")
        else:
          self.label_targets.discard("sites")
        self._refresh_annotations()
        self._request_update()

      @cb_body_labels.on_update
      def _(_) -> None:
        if cb_body_labels.value:
          self.label_targets.add("bodies")
        else:
          self.label_targets.discard("bodies")
        self._refresh_annotations()
        self._request_update()

      # Frames callbacks.
      @cb_site_frames.on_update
      def _(_) -> None:
        if cb_site_frames.value:
          self.frame_targets.add("sites")
        else:
          self.frame_targets.discard("sites")
        self._refresh_annotations()
        self._request_update()

      @cb_body_frames.on_update
      def _(_) -> None:
        if cb_body_frames.value:
          self.frame_targets.add("bodies")
        else:
          self.frame_targets.discard("bodies")
        self._refresh_annotations()
        self._request_update()

      @frame_scale_input.on_update
      def _(_) -> None:
        self.frame_scale = frame_scale_input.value
        if self.frame_targets:
          self._refresh_annotations()
          self._request_update()

      # Contacts callbacks.
      @cb_contact_points.on_update
      def _(_) -> None:
        self.show_contact_points = cb_contact_points.value
        self._sync_visibilities()
        self._request_update()

      @cb_contact_forces.on_update
      def _(_) -> None:
        self.show_contact_forces = cb_contact_forces.value
        self._sync_visibilities()
        self._request_update()

      @contact_scale_input.on_update
      def _(_) -> None:
        self.meansize_override = contact_scale_input.value
        self._request_update()

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
    if self.show_contact_points or self.show_contact_forces:
      self.mj_data.qpos[:] = wp_data.qpos.numpy()[env_idx]
      self.mj_data.qvel[:] = wp_data.qvel.numpy()[env_idx]
      self.mj_data.mocap_pos[:] = mocap_pos[env_idx]
      self.mj_data.mocap_quat[:] = mocap_quat[env_idx]
      mujoco.mj_forward(self.mj_model, self.mj_data)
      contacts = self._extract_contacts_from_mjdata(self.mj_data)

    self._update_visualization(
      body_xpos, body_xmat, mocap_pos, mocap_quat, env_idx, scene_offset, contacts
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
    # This allows toggling contact visualization without needing to scrub timesteps.
    # Not performance-critical since this isn't called in tight loops.
    contacts = self._extract_contacts_from_mjdata(mj_data)

    self._update_visualization(
      body_xpos, body_xmat, mocap_pos, mocap_quat, env_idx, scene_offset, contacts
    )

    # Update scene offset for debug visualizations and sync arrows
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
  ) -> None:
    """Shared visualization update logic."""
    # Cache visualization state for re-rendering when settings change.
    self._last_body_xpos = body_xpos
    self._last_body_xmat = body_xmat
    self._last_mocap_pos = mocap_pos
    self._last_mocap_quat = mocap_quat
    self._last_env_idx = env_idx
    self._scene_offset = scene_offset
    # Only update cached contacts if we have new contact data (don't overwrite with None)
    if contacts is not None:
      self._last_contacts = contacts

    self.fixed_bodies_frame.position = scene_offset
    with self.server.atomic():
      body_xquat = vtf.SO3.from_matrix(body_xmat).wxyz
      for (body_id, _group_id), handle in self.mesh_handles_by_group.items():
        if not handle.visible:
          continue
        # Check if this is a mocap body.
        mocap_id = self.mj_model.body_mocapid[body_id]
        if mocap_id >= 0:
          # Use mocap pos/quat for mocap bodies.
          # Note: mocap_quat is already in wxyz format (MuJoCo convention).
          if self.show_only_selected and self.num_envs > 1:
            single_pos = mocap_pos[env_idx, mocap_id, :] + scene_offset
            single_quat = mocap_quat[env_idx, mocap_id, :]
            handle.batched_positions = np.tile(single_pos[None, :], (self.num_envs, 1))
            handle.batched_wxyzs = np.tile(single_quat[None, :], (self.num_envs, 1))
          else:
            handle.batched_positions = mocap_pos[:, mocap_id, :] + scene_offset
            handle.batched_wxyzs = mocap_quat[:, mocap_id, :]
        else:
          # Use xpos/xmat for regular bodies.
          if self.show_only_selected and self.num_envs > 1:
            single_pos = body_xpos[env_idx, body_id, :] + scene_offset
            single_quat = body_xquat[env_idx, body_id, :]
            handle.batched_positions = np.tile(single_pos[None, :], (self.num_envs, 1))
            handle.batched_wxyzs = np.tile(single_quat[None, :], (self.num_envs, 1))
          else:
            handle.batched_positions = body_xpos[..., body_id, :] + scene_offset
            handle.batched_wxyzs = body_xquat[..., body_id, :]

      # Update site positions (non-fixed bodies only).
      for (body_id, _group_id), handle in self.site_handles_by_group.items():
        if not handle.visible:
          continue
        # Sites follow their parent body's pose.
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

      # Update annotation positions (labels and frames).
      self._update_annotations(body_xpos, body_xmat, env_idx, scene_offset)

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
      return  # No cached data yet

    # Use cached contacts (don't recompute - the data might be stale).
    # The next regular update will refresh contacts if needed.
    contacts = (
      self._last_contacts
      if (self.show_contact_points or self.show_contact_forces)
      else None
    )

    # Recalculate scene offset based on current camera tracking state.
    scene_offset = np.zeros(3)
    if self.camera_tracking_enabled and self._tracked_body_id is not None:
      tracked_pos = self._last_body_xpos[
        self._last_env_idx, self._tracked_body_id, :
      ].copy()
      scene_offset = -tracked_pos

    # Re-render with cached data (_update_visualization has its own atomic block and flush)
    self._update_visualization(
      self._last_body_xpos,
      self._last_body_xmat,
      self._last_mocap_pos,
      self._last_mocap_quat,
      self._last_env_idx,
      scene_offset,
      contacts,
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

    # Process all bodies with geoms.
    all_bodies = set(body_geoms_visual.keys()) | set(body_geoms_collision.keys())

    for body_id in all_bodies:
      # Get body name.
      body_name = get_body_name(self.mj_model, body_id)

      # Fixed world geometry. We'll assume this is shared between all environments.
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

        # Iterate over geoms.
        nonplane_geom_ids: list[int] = []
        for geom_id in all_geoms:
          geom_type = self.mj_model.geom_type[geom_id]
          # Add plane geoms as infinite grids.
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

        # Handle non-plane geoms.
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

    # Add sites attached to fixed world bodies, grouped by site_group.
    body_group_sites: dict[tuple[int, int], list[int]] = {}
    for i in range(self.mj_model.nsite):
      body_id = self.mj_model.site_bodyid[i]
      if is_fixed_body(self.mj_model, body_id):
        site_group = self.mj_model.site_group[i]
        key = (body_id, site_group)
        body_group_sites.setdefault(key, []).append(i)

    for (body_id, group_id), site_ids in body_group_sites.items():
      body_name = get_body_name(self.mj_model, body_id)
      visible = group_id < 6 and self.site_groups_visible[group_id]
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
    # Group geoms by (body_id, group_id).
    body_group_geoms: dict[tuple[int, int], list[int]] = {}

    for i in range(self.mj_model.ngeom):
      body_id = self.mj_model.geom_bodyid[i]

      # Skip fixed world geometry.
      if is_fixed_body(self.mj_model, body_id):
        continue

      geom_group = self.mj_model.geom_group[i]
      key = (body_id, geom_group)

      if key not in body_group_geoms:
        body_group_geoms[key] = []
      body_group_geoms[key].append(i)

    # Create handles for each (body, group) combination.
    with self.server.atomic():
      for (body_id, group_id), geom_indices in body_group_geoms.items():
        # Get body name.
        body_name = get_body_name(self.mj_model, body_id)

        # Merge geoms into a single mesh.
        mesh = merge_geoms(self.mj_model, geom_indices)
        lod_ratio = 1000.0 / mesh.vertices.shape[0]

        # Check if this group should be visible.
        visible = group_id < 6 and self.geom_groups_visible[group_id]

        # Create handle.
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
    # Group sites by (body_id, group_id).
    body_group_sites: dict[tuple[int, int], list[int]] = {}

    for i in range(self.mj_model.nsite):
      body_id = self.mj_model.site_bodyid[i]

      # Skip fixed world sites (handled in _add_fixed_geometry).
      if is_fixed_body(self.mj_model, body_id):
        continue

      site_group = self.mj_model.site_group[i]
      key = (body_id, site_group)
      body_group_sites.setdefault(key, []).append(i)

    # Create handles for each (body, group) combination.
    with self.server.atomic():
      for (body_id, group_id), site_indices in body_group_sites.items():
        body_name = get_body_name(self.mj_model, body_id)

        # Merge sites into a single mesh.
        mesh = merge_sites(self.mj_model, site_indices)
        lod_ratio = 1000.0 / mesh.vertices.shape[0]

        # Check if this group should be visible.
        visible = group_id < 6 and self.site_groups_visible[group_id]

        # Create batched handle.
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

  def _extract_contacts_from_mjdata(self, mj_data: mujoco.MjData) -> list[_Contact]:
    """Extract contact data from given MuJoCo data."""

    def make_contact(i: int) -> _Contact:
      con, force = mj_data.contact[i], np.zeros(6)
      mujoco.mj_contactForce(self.mj_model, mj_data, i, force)
      return _Contact(
        pos=con.pos.copy(),
        frame=con.frame.copy().reshape(3, 3),
        force=force[:3].copy(),
        dist=con.dist,
        included=con.efc_address >= 0,
      )

    return [make_contact(i) for i in range(mj_data.ncon)]

  def _update_contact_visualization(
    self, contacts: list[_Contact], scene_offset: np.ndarray
  ) -> None:
    """Update contact point and force visualization."""
    contact_points: list[_ContactPointVisual] = []
    contact_forces: list[_ContactForceVisual] = []

    meansize = self.meansize_override or self.mj_model.stat.meansize

    for contact in contacts:
      if not contact.included:
        continue

      # Transform force from contact frame to world frame.
      force_world = contact.frame.T @ contact.force
      force_mag = np.linalg.norm(force_world)

      # Contact point visualization (cylinder).
      if self.show_contact_points:
        contact_points.append(
          _ContactPointVisual(
            position=contact.pos + scene_offset,
            orientation=vtf.SO3.from_matrix(
              rotation_matrix_from_vectors(np.array([0, 0, 1]), contact.frame[0, :])
            ).wxyz,
            scale=np.array(
              [
                self.mj_model.vis.scale.contactwidth * meansize,
                self.mj_model.vis.scale.contactwidth * meansize,
                self.mj_model.vis.scale.contactheight * meansize,
              ]
            ),
          )
        )

      # Contact force visualization (arrow shaft + head).
      if self.show_contact_forces and force_mag > 1e-6:
        force_dir = force_world / force_mag
        arrow_length = (
          force_mag * (self.mj_model.vis.map.force / self.mj_model.stat.meanmass)
          if self.mj_model.stat.meanmass > 0
          else force_mag
        )
        arrow_width = self.mj_model.vis.scale.forcewidth * meansize
        force_quat = vtf.SO3.from_matrix(
          rotation_matrix_from_vectors(np.array([0, 0, 1]), force_dir)
        ).wxyz

        contact_forces.append(
          _ContactForceVisual(
            shaft_position=contact.pos + scene_offset,
            shaft_orientation=force_quat,
            shaft_scale=np.array([arrow_width, arrow_width, arrow_length]),
            head_position=contact.pos + scene_offset + force_dir * arrow_length,
            head_orientation=force_quat,
            head_scale=np.array([arrow_width, arrow_width, arrow_width]),
          )
        )

    # Update or create contact point handle.
    if contact_points:
      positions = np.array([p.position for p in contact_points], dtype=np.float32)
      orientations = np.array([p.orientation for p in contact_points], dtype=np.float32)
      scales = np.array([p.scale for p in contact_points], dtype=np.float32)
      if self.contact_point_handle is None:
        mesh = trimesh.creation.cylinder(radius=1.0, height=1.0)
        self.contact_point_handle = self.server.scene.add_batched_meshes_simple(
          "/contacts/points",
          mesh.vertices,
          mesh.faces,
          batched_wxyzs=orientations,
          batched_positions=positions,
          batched_scales=scales,
          batched_colors=np.array(self.contact_point_color, dtype=np.uint8),
          opacity=0.8,
          lod="off",
          cast_shadow=False,
          receive_shadow=False,
        )
      self.contact_point_handle.batched_positions = positions
      self.contact_point_handle.batched_wxyzs = orientations
      self.contact_point_handle.batched_scales = scales
      self.contact_point_handle.visible = True
    elif self.contact_point_handle is not None:
      self.contact_point_handle.visible = False

    # Update or create contact force handles (shaft and head separately).
    if contact_forces:
      shaft_positions = np.array(
        [f.shaft_position for f in contact_forces], dtype=np.float32
      )
      shaft_orientations = np.array(
        [f.shaft_orientation for f in contact_forces], dtype=np.float32
      )
      shaft_scales = np.array([f.shaft_scale for f in contact_forces], dtype=np.float32)
      head_positions = np.array(
        [f.head_position for f in contact_forces], dtype=np.float32
      )
      head_orientations = np.array(
        [f.head_orientation for f in contact_forces], dtype=np.float32
      )
      head_scales = np.array([f.head_scale for f in contact_forces], dtype=np.float32)
      if self.contact_force_shaft_handle is None:
        shaft_mesh = trimesh.creation.cylinder(radius=0.4, height=1.0)
        shaft_mesh.apply_translation([0, 0, 0.5])
        self.contact_force_shaft_handle = self.server.scene.add_batched_meshes_simple(
          "/contacts/forces/shaft",
          shaft_mesh.vertices,
          shaft_mesh.faces,
          batched_wxyzs=shaft_orientations,
          batched_positions=shaft_positions,
          batched_scales=shaft_scales,
          batched_colors=np.array(self.contact_force_color, dtype=np.uint8),
          opacity=0.8,
          lod="off",
          cast_shadow=False,
          receive_shadow=False,
        )
        head_mesh = trimesh.creation.cone(radius=1.0, height=1.0, sections=8)
        self.contact_force_head_handle = self.server.scene.add_batched_meshes_simple(
          "/contacts/forces/head",
          head_mesh.vertices,
          head_mesh.faces,
          batched_wxyzs=head_orientations,
          batched_positions=head_positions,
          batched_scales=head_scales,
          batched_colors=np.array(self.contact_force_color, dtype=np.uint8),
          opacity=0.8,
          lod="off",
          cast_shadow=False,
          receive_shadow=False,
        )
      assert self.contact_force_shaft_handle is not None
      assert self.contact_force_head_handle is not None
      self.contact_force_shaft_handle.batched_positions = shaft_positions
      self.contact_force_shaft_handle.batched_wxyzs = shaft_orientations
      self.contact_force_shaft_handle.batched_scales = shaft_scales
      self.contact_force_shaft_handle.visible = True
      self.contact_force_head_handle.batched_positions = head_positions
      self.contact_force_head_handle.batched_wxyzs = head_orientations
      self.contact_force_head_handle.batched_scales = head_scales
      self.contact_force_head_handle.visible = True
    elif (
      self.contact_force_shaft_handle is not None
      and self.contact_force_head_handle is not None
    ):
      self.contact_force_shaft_handle.visible = (
        self.contact_force_head_handle.visible
      ) = False

  # ============================================================================
  # DebugVisualizer Protocol Implementation
  # ============================================================================

  @override
  def add_arrow(
    self,
    start: np.ndarray | torch.Tensor,
    end: np.ndarray | torch.Tensor,
    color: tuple[float, float, float, float],
    width: float = 0.015,
    label: str | None = None,
  ) -> None:
    """Queue an arrow for batched rendering.

    Arrows are not rendered immediately but queued and rendered together
    in the next update() call for efficiency.
    """
    if not self.debug_visualization_enabled:
      return

    del label  # Unused.
    if isinstance(start, torch.Tensor):
      start = start.cpu().numpy()
    if isinstance(end, torch.Tensor):
      end = end.cpu().numpy()

    direction = end - start
    length = np.linalg.norm(direction)

    if length < 1e-6:
      return

    # Queue the arrow for batched rendering (without scene offset - applied during sync)
    self._queued_arrows.append((start, end, color, width))

  @override
  def add_ghost_mesh(
    self,
    qpos: np.ndarray | torch.Tensor,
    model: mujoco.MjModel,
    alpha: float = 0.5,
    label: str | None = None,
  ) -> None:
    """Add a ghost mesh by rendering the robot at a different pose.

    For Viser, we create meshes once and update their poses for efficiency.

    Args:
      qpos: Joint positions for the ghost pose
      model: MuJoCo model with pre-configured appearance (geom_rgba for colors)
      alpha: Transparency override
      label: Optional label for this ghost
    """
    if not self.debug_visualization_enabled:
      return

    if isinstance(qpos, torch.Tensor):
      qpos = qpos.cpu().numpy()

    # Use model hash to support models with same structure but different colors
    model_hash = hash((model.ngeom, model.nbody, model.nq))

    self._viz_data.qpos[:] = qpos
    mujoco.mj_forward(model, self._viz_data)

    # Use current scene offset
    scene_offset = self._scene_offset

    # Group geoms by body
    body_geoms: dict[int, list[int]] = {}
    for i in range(model.ngeom):
      body_id = model.geom_bodyid[i]
      is_collision = model.geom_contype[i] != 0 or model.geom_conaffinity[i] != 0
      if is_collision:
        continue

      if model.body_dofnum[body_id] == 0 and model.body_parentid[body_id] == 0:
        continue

      if body_id not in body_geoms:
        body_geoms[body_id] = []
      body_geoms[body_id].append(i)

    # Update or create mesh for each body
    for body_id, geom_indices in body_geoms.items():
      body_pos = self._viz_data.xpos[body_id] + scene_offset
      body_quat = self._mat_to_quat(self._viz_data.xmat[body_id].reshape(3, 3))

      # Check if we already have a handle for this body
      if body_id in self._ghost_handles:
        handle = self._ghost_handles[body_id]
        handle.wxyz = body_quat
        handle.position = body_pos
      else:
        # Create mesh if not cached
        if model_hash not in self._ghost_meshes:
          self._ghost_meshes[model_hash] = {}

        if body_id not in self._ghost_meshes[model_hash]:
          meshes = []
          for geom_id in geom_indices:
            mesh = self._create_geom_mesh_from_model(model, geom_id)
            if mesh is not None:
              geom_pos = model.geom_pos[geom_id]
              geom_quat = model.geom_quat[geom_id]
              transform = np.eye(4)
              transform[:3, :3] = vtf.SO3(geom_quat).as_matrix()
              transform[:3, 3] = geom_pos
              mesh.apply_transform(transform)
              meshes.append(mesh)

          if not meshes:
            continue

          combined_mesh = (
            meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
          )

          self._ghost_meshes[model_hash][body_id] = combined_mesh
        else:
          combined_mesh = self._ghost_meshes[model_hash][body_id]

        body_name = get_body_name(model, body_id)
        handle_name = f"/debug/env_{self.env_idx}/ghost/body_{body_name}"

        # Extract color from geom (convert RGBA 0-1 to RGB 0-255)
        rgba = model.geom_rgba[geom_indices[0]].copy()
        color_uint8 = (rgba[:3] * 255).astype(np.uint8)

        handle = self.server.scene.add_mesh_simple(
          handle_name,
          combined_mesh.vertices,
          combined_mesh.faces,
          color=tuple(color_uint8),
          opacity=alpha,
          wxyz=body_quat,
          position=body_pos,
          cast_shadow=False,
          receive_shadow=False,
        )
        self._ghost_handles[body_id] = handle

  @override
  def add_frame(
    self,
    position: np.ndarray | torch.Tensor,
    rotation_matrix: np.ndarray | torch.Tensor,
    scale: float = 0.3,
    label: str | None = None,
    axis_radius: float = 0.01,
    alpha: float = 1.0,
    axis_colors: tuple[tuple[float, float, float], ...] | None = None,
  ) -> None:
    """Add a coordinate frame visualization with RGB-colored axes.

    This implementation reuses add_arrow to draw the three axis arrows.

    Args:
      position: Position of the frame origin (3D vector)
      rotation_matrix: Rotation matrix (3x3)
      scale: Scale/length of the axis arrows
      label: Optional label for this frame.
      axis_radius: Radius of the axis arrows.
      alpha: Opacity for all axes (0=transparent, 1=opaque). Note: This implementation
        does not support per-arrow transparency. All arrows in the scene will share
        the same alpha value.
      axis_colors: Optional tuple of 3 RGB colors for X, Y, Z axes. If None, uses
        default RGB coloring (X=red, Y=green, Z=blue).
    """
    if not self.debug_visualization_enabled:
      return

    del label  # Unused.

    if isinstance(position, torch.Tensor):
      position = position.cpu().numpy()
    if isinstance(rotation_matrix, torch.Tensor):
      rotation_matrix = rotation_matrix.cpu().numpy()

    default_colors = [(0.9, 0, 0), (0, 0.9, 0.0), (0.0, 0.0, 0.9)]
    colors = axis_colors if axis_colors is not None else default_colors

    for axis_idx in range(3):
      axis_direction = rotation_matrix[:, axis_idx]
      end_position = position + axis_direction * scale
      rgb = colors[axis_idx]
      color_rgba = (rgb[0], rgb[1], rgb[2], alpha)
      self.add_arrow(
        start=position,
        end=end_position,
        color=color_rgba,
        width=axis_radius,
      )

  @override
  def clear(self) -> None:
    """Clear all debug visualizations.

    Clears the arrow queue. Ghost meshes are kept and pose-updated for efficiency
    within the same environment, but removed when switching environments.
    """
    self._queued_arrows.clear()

  def clear_debug_all(self) -> None:
    """Clear all debug visualizations including ghosts.

    Called when switching to a different environment or disabling debug visualization.
    """
    self.clear()

    # Remove arrow meshes
    if self._arrow_shaft_handle is not None:
      self._arrow_shaft_handle.remove()
      self._arrow_shaft_handle = None
    if self._arrow_head_handle is not None:
      self._arrow_head_handle.remove()
      self._arrow_head_handle = None

    # Remove ghost meshes
    for handle in self._ghost_handles.values():
      handle.remove()
    self._ghost_handles.clear()

  def _create_geom_mesh_from_model(
    self, mj_model: mujoco.MjModel, geom_id: int
  ) -> trimesh.Trimesh | None:
    """Create a trimesh from a MuJoCo geom using the specified model.

    Args:
      mj_model: MuJoCo model containing geom definition
      geom_id: Index of the geom to create mesh for

    Returns:
      Trimesh representation of the geom, or None if unsupported type
    """
    geom_type = mj_model.geom_type[geom_id]

    if geom_type == mjtGeom.mjGEOM_MESH:
      return mujoco_mesh_to_trimesh(mj_model, geom_id, verbose=False)
    else:
      return create_primitive_mesh(mj_model, geom_id)

  def _sync_arrows(self) -> None:
    """Render all queued arrows using batched meshes.

    This should be called after all debug visualizations have been queued
    for the current frame.
    """
    if not self.debug_visualization_enabled:
      return

    if not self._queued_arrows:
      # Remove arrow meshes if no arrows to render
      if self._arrow_shaft_handle is not None:
        self._arrow_shaft_handle.remove()
        self._arrow_shaft_handle = None
      if self._arrow_head_handle is not None:
        self._arrow_head_handle.remove()
        self._arrow_head_handle = None
      return

    # Create arrow mesh components if needed (unit-sized base meshes)
    if self._arrow_shaft_mesh is None:
      # Unit cylinder: radius=1.0, height=1.0
      self._arrow_shaft_mesh = trimesh.creation.cylinder(radius=1.0, height=1.0)
      self._arrow_shaft_mesh.apply_translation(np.array([0, 0, 0.5]))  # Center at z=0.5

    if self._arrow_head_mesh is None:
      # Unit cone: radius=2.0, height=1.0 (base at z=0, tip at z=1.0 by default)
      head_width = 2.0
      self._arrow_head_mesh = trimesh.creation.cone(radius=head_width, height=1.0)
      # No translation needed - cone already has base at z=0

    # Prepare batched data
    num_arrows = len(self._queued_arrows)
    shaft_positions = np.zeros((num_arrows, 3), dtype=np.float32)
    shaft_wxyzs = np.zeros((num_arrows, 4), dtype=np.float32)
    shaft_scales = np.zeros((num_arrows, 3), dtype=np.float32)
    shaft_colors = np.zeros((num_arrows, 3), dtype=np.uint8)

    head_positions = np.zeros((num_arrows, 3), dtype=np.float32)
    head_wxyzs = np.zeros((num_arrows, 4), dtype=np.float32)
    head_scales = np.zeros((num_arrows, 3), dtype=np.float32)
    head_colors = np.zeros((num_arrows, 3), dtype=np.uint8)

    z_axis = np.array([0, 0, 1])
    shaft_length_ratio = 0.8
    head_length_ratio = 0.2

    # Apply scene offset to all arrows
    for i, (start, end, color, width) in enumerate(self._queued_arrows):
      # Apply scene offset
      start_offset = start + self._scene_offset
      end_offset = end + self._scene_offset

      direction = end_offset - start_offset
      length = np.linalg.norm(direction)
      direction = direction / length

      rotation_quat = rotation_quat_from_vectors(z_axis, direction)

      # Shaft: scale width in XY, length in Z
      shaft_length = shaft_length_ratio * length
      shaft_positions[i] = start_offset
      shaft_wxyzs[i] = rotation_quat
      shaft_scales[i] = [width, width, shaft_length]  # Per-axis scale
      shaft_colors[i] = (np.array(color[:3]) * 255).astype(np.uint8)

      # Head: position at end of shaft
      # The cone has its base at z=0, so after scaling by head_length,
      # the base is still at z=0 in local coords
      # We want the base at the end of the shaft (at shaft_length)
      head_length = head_length_ratio * length
      head_position = start_offset + direction * shaft_length
      head_positions[i] = head_position
      head_wxyzs[i] = rotation_quat
      head_scales[i] = [width, width, head_length]  # Per-axis scale
      head_colors[i] = (np.array(color[:3]) * 255).astype(np.uint8)

    # Check if we need to recreate handles (number of arrows changed)
    needs_recreation = (
      self._arrow_shaft_handle is None
      or self._arrow_head_handle is None
      or len(shaft_positions) != len(self._arrow_shaft_handle.batched_positions)
    )

    if needs_recreation:
      # Remove old handles
      if self._arrow_shaft_handle is not None:
        self._arrow_shaft_handle.remove()
      if self._arrow_head_handle is not None:
        self._arrow_head_handle.remove()

      # Create new batched meshes
      self._arrow_shaft_handle = self.server.scene.add_batched_meshes_simple(
        f"/debug/env_{self.env_idx}/arrow_shafts",
        self._arrow_shaft_mesh.vertices,
        self._arrow_shaft_mesh.faces,
        batched_wxyzs=shaft_wxyzs,
        batched_positions=shaft_positions,
        batched_scales=shaft_scales,
        batched_colors=shaft_colors,
        cast_shadow=False,
        receive_shadow=False,
      )

      self._arrow_head_handle = self.server.scene.add_batched_meshes_simple(
        f"/debug/env_{self.env_idx}/arrow_heads",
        self._arrow_head_mesh.vertices,
        self._arrow_head_mesh.faces,
        batched_wxyzs=head_wxyzs,
        batched_positions=head_positions,
        batched_scales=head_scales,
        batched_colors=head_colors,
        cast_shadow=False,
        receive_shadow=False,
      )
    else:
      # Update existing handles (guaranteed to exist by needs_recreation check)
      assert self._arrow_shaft_handle is not None
      assert self._arrow_head_handle is not None

      self._arrow_shaft_handle.batched_positions = shaft_positions
      self._arrow_shaft_handle.batched_wxyzs = shaft_wxyzs
      self._arrow_shaft_handle.batched_scales = shaft_scales
      self._arrow_shaft_handle.batched_colors = shaft_colors

      self._arrow_head_handle.batched_positions = head_positions
      self._arrow_head_handle.batched_wxyzs = head_wxyzs
      self._arrow_head_handle.batched_scales = head_scales
      self._arrow_head_handle.batched_colors = head_colors

  @staticmethod
  def _mat_to_quat(mat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion (wxyz)."""
    return vtf.SO3.from_matrix(mat).wxyz
