"""Annotation management for ViserMujocoScene (labels and coordinate frames)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import viser
import viser.transforms as vtf
from mujoco import mj_id2name, mjtObj

from mjlab.viewer.viser.conversions import get_body_name, is_fixed_body

if TYPE_CHECKING:
  import mujoco

_NUM_GEOM_GROUPS = 6


class ViserSceneAnnotationsMixin:
  """Mixin providing annotation management for ViserMujocoScene.

  Handles text labels and coordinate frames for sites and bodies.
  """

  # Attributes required from parent class (for type checking).
  server: viser.ViserServer
  mj_model: mujoco.MjModel
  site_groups_visible: list[bool]
  label_targets: set[str]
  frame_targets: set[str]
  frame_scale: float
  _label_handles: dict[str, viser.LabelHandle]
  _frame_handles: dict[str, viser.FrameHandle]

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

    if "sites" in self.label_targets:
      for site_id in range(self.mj_model.nsite):
        site_group = self.mj_model.site_group[site_id]
        if site_group >= _NUM_GEOM_GROUPS or not self.site_groups_visible[site_group]:
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

    if "sites" in self.frame_targets:
      self._create_frame_handles("sites")

    if "bodies" in self.frame_targets:
      self._create_frame_handles("bodies")

  def _create_frame_handles(self, target: str) -> None:
    """Create coordinate frame visualization handles for the given target type."""
    meansize = self.mj_model.stat.meansize
    frame_length = self.mj_model.vis.scale.framelength * meansize * self.frame_scale
    frame_width = self.mj_model.vis.scale.framewidth * meansize * self.frame_scale

    if target == "sites":
      for site_id in range(self.mj_model.nsite):
        site_group = self.mj_model.site_group[site_id]
        if site_group >= _NUM_GEOM_GROUPS or not self.site_groups_visible[site_group]:
          continue
        key = f"site_frame_{site_id}"
        handle = self.server.scene.add_frame(
          f"/annotations/frames/{key}",
          axes_length=frame_length,
          axes_radius=frame_width,
        )
        self._frame_handles[key] = handle

    else:
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
    mj_data: mujoco.MjData | None,
  ) -> None:
    """Update positions of all annotations for the selected environment."""
    if not self.label_targets and not self.frame_targets:
      return

    if "sites" in self.label_targets and mj_data is not None:
      for site_id in range(self.mj_model.nsite):
        key = f"site_{site_id}"
        if key not in self._label_handles:
          continue
        site_world_pos = mj_data.site(site_id).xpos
        self._label_handles[key].position = site_world_pos + scene_offset

    if "bodies" in self.label_targets:
      for body_id in range(self.mj_model.nbody):
        key = f"body_{body_id}"
        if key not in self._label_handles:
          continue
        body_pos = body_xpos[env_idx, body_id, :] + scene_offset
        self._label_handles[key].position = body_pos

    if "sites" in self.frame_targets:
      self._update_frame_positions(
        "sites", body_xpos, body_xmat, env_idx, scene_offset, mj_data
      )

    if "bodies" in self.frame_targets:
      self._update_frame_positions(
        "bodies", body_xpos, body_xmat, env_idx, scene_offset, mj_data
      )

  def _update_frame_positions(
    self,
    target: str,
    body_xpos: np.ndarray,
    body_xmat: np.ndarray,
    env_idx: int,
    scene_offset: np.ndarray,
    mj_data: mujoco.MjData | None,
  ) -> None:
    """Update frame handle positions and orientations."""
    body_xquat = vtf.SO3.from_matrix(body_xmat).wxyz

    if target == "sites" and mj_data is not None:
      for site_id in range(self.mj_model.nsite):
        key = f"site_frame_{site_id}"
        if key not in self._frame_handles:
          continue

        site_world_pos = mj_data.site(site_id).xpos + scene_offset
        site_world_mat = mj_data.site(site_id).xmat.reshape(3, 3)
        site_world_quat = vtf.SO3.from_matrix(site_world_mat).wxyz

        handle = self._frame_handles[key]
        handle.position = site_world_pos
        handle.wxyz = site_world_quat

    elif target == "bodies":
      for body_id in range(self.mj_model.nbody):
        key = f"body_frame_{body_id}"
        if key not in self._frame_handles:
          continue

        body_pos = body_xpos[env_idx, body_id, :] + scene_offset
        body_quat = body_xquat[env_idx, body_id, :]

        handle = self._frame_handles[key]
        handle.position = body_pos
        handle.wxyz = body_quat
