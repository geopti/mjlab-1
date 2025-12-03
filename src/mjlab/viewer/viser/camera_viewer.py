"""Camera image visualization for Viser viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import viser

if TYPE_CHECKING:
  from mjlab.sensor.camera_sensor import CameraSensor


class ViserCameraViewer:
  """Handles camera image visualization for the Viser viewer."""

  def __init__(
    self,
    server: viser.ViserServer,
    camera_sensor: CameraSensor,
    min_display_size: int = 256,
  ):
    """Initialize the camera viewer.

    Args:
      server: The Viser server instance
      camera_sensor: The camera sensor to visualize
      min_display_size: Minimum display size for images (will upsample if smaller)
    """
    self._server = server
    self._camera_sensor = camera_sensor

    self._rgb_handle: viser.GuiImageHandle | None = None
    self._depth_handle: viser.GuiImageHandle | None = None

    self._camera_name = camera_sensor.camera_name

    self._has_rgb = "rgb" in self._camera_sensor.cfg.type
    self._has_depth = "depth" in self._camera_sensor.cfg.type

    height = self._camera_sensor.cfg.height
    width = self._camera_sensor.cfg.width

    # Calculate display size. Upsample if image is too small
    scale = max(1, min_display_size // max(height, width))
    self._display_height = height * scale
    self._display_width = width * scale
    self._needs_upsampling = scale > 1

    if self._has_rgb:
      self._rgb_handle = self._server.gui.add_image(
        image=np.zeros((self._display_height, self._display_width, 3), dtype=np.uint8),
        label=f"{self._camera_name}_rgb",
        format="jpeg",
      )

    if self._has_depth:
      self._depth_handle = self._server.gui.add_image(
        image=np.zeros((self._display_height, self._display_width), dtype=np.uint8),
        label=f"{self._camera_name}_depth",
        format="jpeg",
      )
      self._depth_scale_slider = self._server.gui.add_slider(
        label=f"{self._camera_name}_depth_scale",
        min=0.1,
        max=10.0,
        step=0.1,
        initial_value=1.0,
      )

  def _upsample_nearest(self, image: np.ndarray, scale: int) -> np.ndarray:
    return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)

  def update(self, env_idx: int = 0) -> None:
    """Update the camera images for a single environment.

    Args:
      env_idx: Environment index to visualize
    """
    # Access sensor data - sensor's internal caching handles rate limiting
    # based on update_period, avoiding redundant GPU->CPU transfers
    data = self._camera_sensor.data

    if self._has_rgb and self._rgb_handle is not None and data.rgb is not None:
      rgb_np = data.rgb[env_idx].cpu().numpy()

      # Upsample if needed for better visibility
      if self._needs_upsampling:
        scale = self._display_height // rgb_np.shape[0]
        rgb_np = self._upsample_nearest(rgb_np, scale)

      self._rgb_handle.image = rgb_np

    if self._has_depth and self._depth_handle is not None and data.depth is not None:
      depth_np = data.depth[env_idx].squeeze().cpu().numpy()

      depth_scale = self._depth_scale_slider.value
      depth_normalized = np.clip(depth_np / depth_scale, 0.0, 1.0)
      depth_uint8 = (depth_normalized * 255).astype(np.uint8)
      if self._needs_upsampling:
        scale = self._display_height // depth_uint8.shape[0]
        depth_uint8 = self._upsample_nearest(depth_uint8, scale)

      self._depth_handle.image = depth_uint8

  def cleanup(self) -> None:
    """Clean up resources."""
    if self._rgb_handle is not None:
      self._rgb_handle.remove()
    if self._depth_handle is not None:
      self._depth_handle.remove()
      self._depth_scale_slider.remove()
