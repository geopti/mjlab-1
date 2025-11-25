"""GUI creation methods for ViserMujocoScene."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import viser

if TYPE_CHECKING:
  import mujoco

  from mjlab.viewer.viser.scene_protocol import ViserSceneProtocol

# Viser visualization defaults.
_DEFAULT_FOV_DEGREES = 60
_DEFAULT_FOV_MIN = 20
_DEFAULT_FOV_MAX = 150
_NUM_GEOM_GROUPS = 6


class ViserSceneGuiMixin:
  """Mixin providing GUI creation methods for ViserMujocoScene.

  Requires parent class to have the attributes and methods declared below.
  """

  # Attributes required from parent class (for type checking).
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
        cast("ViserSceneProtocol", self)._request_update()

  def create_visualization_gui(
    self,
    camera_distance: float = 3.0,
    camera_azimuth: float = 45.0,
    camera_elevation: float = 30.0,
    show_debug_viz_control: bool = True,
  ) -> None:
    """Add standard GUI controls that automatically update this scene's settings.

    Args:
      camera_distance: Default camera distance from tracked body when tracking is
        enabled.
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
          cast("ViserSceneProtocol", self)._request_update()

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

        cast("ViserSceneProtocol", self)._request_update()

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
            cast("ViserSceneProtocol", self).clear_debug_all()
          cast("ViserSceneProtocol", self)._request_update()

  def create_groups_gui(self, tabs) -> None:
    """Add groups tab combining geom and site visibility controls.

    Args:
      tabs: The viser tab group to add the groups tab to.
    """
    with tabs.add_tab("Groups", icon=viser.Icon.EYE):
      self.server.gui.add_markdown("**Geoms**")
      for i in range(_NUM_GEOM_GROUPS):
        cb = self.server.gui.add_checkbox(
          f"G{i}",
          initial_value=self.geom_groups_visible[i],
          hint=f"Show/hide geoms in group {i}",
        )

        @cb.on_update
        def _(event, group_idx=i) -> None:
          self.geom_groups_visible[group_idx] = event.target.value
          cast("ViserSceneProtocol", self)._sync_visibilities()
          cast("ViserSceneProtocol", self)._request_update()

      self.server.gui.add_markdown("**Sites**")
      for i in range(_NUM_GEOM_GROUPS):
        cb = self.server.gui.add_checkbox(
          f"S{i}",
          initial_value=self.site_groups_visible[i],
          hint=f"Show/hide sites in group {i}",
        )

        @cb.on_update
        def _(event, group_idx=i) -> None:
          self.site_groups_visible[group_idx] = event.target.value
          cast("ViserSceneProtocol", self)._sync_visibilities()
          # Refresh annotations if any site annotations are enabled.
          if "sites" in self.label_targets or "sites" in self.frame_targets:
            cast("ViserSceneProtocol", self)._refresh_annotations()
            cast("ViserSceneProtocol", self)._request_update()

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
        cast("ViserSceneProtocol", self)._refresh_annotations()
        cast("ViserSceneProtocol", self)._request_update()

      @cb_body_labels.on_update
      def _(_) -> None:
        if cb_body_labels.value:
          self.label_targets.add("bodies")
        else:
          self.label_targets.discard("bodies")
        cast("ViserSceneProtocol", self)._refresh_annotations()
        cast("ViserSceneProtocol", self)._request_update()

      # Frames callbacks.
      @cb_site_frames.on_update
      def _(_) -> None:
        if cb_site_frames.value:
          self.frame_targets.add("sites")
        else:
          self.frame_targets.discard("sites")
        cast("ViserSceneProtocol", self)._refresh_annotations()
        cast("ViserSceneProtocol", self)._request_update()

      @cb_body_frames.on_update
      def _(_) -> None:
        if cb_body_frames.value:
          self.frame_targets.add("bodies")
        else:
          self.frame_targets.discard("bodies")
        cast("ViserSceneProtocol", self)._refresh_annotations()
        cast("ViserSceneProtocol", self)._request_update()

      @frame_scale_input.on_update
      def _(_) -> None:
        self.frame_scale = frame_scale_input.value
        if self.frame_targets:
          cast("ViserSceneProtocol", self)._refresh_annotations()
          cast("ViserSceneProtocol", self)._request_update()

      # Contacts callbacks.
      @cb_contact_points.on_update
      def _(_) -> None:
        self.show_contact_points = cb_contact_points.value
        cast("ViserSceneProtocol", self)._sync_visibilities()
        cast("ViserSceneProtocol", self)._request_update()

      @cb_contact_forces.on_update
      def _(_) -> None:
        self.show_contact_forces = cb_contact_forces.value
        cast("ViserSceneProtocol", self)._sync_visibilities()
        cast("ViserSceneProtocol", self)._request_update()

      @contact_scale_input.on_update
      def _(_) -> None:
        self.meansize_override = contact_scale_input.value
        cast("ViserSceneProtocol", self)._request_update()
