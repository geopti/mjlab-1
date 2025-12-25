"""Jump command for the jumping task."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


class JumpCommand(CommandTerm):
    """Generates jump commands with target heights.

    The command tensor has shape [num_envs, 2]:
    - [0]: jump_trigger (1.0 when jump requested, decays over time)
    - [1]: target_height (meters above standing height)
    """

    cfg: JumpCommandCfg

    def __init__(self, cfg: JumpCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self.robot: Entity = env.scene[cfg.asset_name]

        # Command buffer: [jump_trigger, target_height]
        self._command = torch.zeros(self.num_envs, 2, device=self.device)

        # Track jump state per environment
        self.jump_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.jump_completed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.was_in_flight = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # Metrics
        self.metrics["target_height"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self) -> None:
        self.metrics["target_height"] = self._command[:, 1].clone()

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Sample new jump commands for the specified environments."""
        r = torch.empty(len(env_ids), device=self.device)

        # Sample target jump height
        self._command[env_ids, 1] = r.uniform_(*self.cfg.ranges.target_height)

        # Activate jump trigger
        self._command[env_ids, 0] = 1.0

        # Reset jump state
        self.jump_active[env_ids] = True
        self.jump_completed[env_ids] = False
        self.was_in_flight[env_ids] = False

    def _update_command(self) -> None:
        """Update command state based on robot state."""
        # Get contact sensor if configured
        if self.cfg.contact_sensor_name is not None:
            contact_sensor: ContactSensor = self._env.scene[self.cfg.contact_sensor_name]
            in_contact = contact_sensor.data.found
            assert in_contact is not None
            # Both feet in contact
            both_feet_contact = (in_contact > 0).all(dim=1)
            any_foot_in_air = ~both_feet_contact

            # Track if we've been in flight (no contact)
            self.was_in_flight = self.was_in_flight | any_foot_in_air

            # Jump is completed when: was in flight AND now both feet in contact
            just_landed = self.was_in_flight & both_feet_contact
            self.jump_completed = self.jump_completed | just_landed

            # Decay trigger when jump is completed
            decay_mask = self.jump_completed
            self._command[decay_mask, 0] = (
                self._command[decay_mask, 0] * self.cfg.trigger_decay_rate
            )

            # Clear jump active flag after landing and stabilization
            # (handled by resampling timer)

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        """Visualize jump target height."""
        batch = visualizer.env_idx
        if batch >= self.num_envs:
            return

        base_pos_w = self.robot.data.root_link_pos_w[batch].cpu().numpy()
        target_height = self._command[batch, 1].item()
        standing_height = self.cfg.standing_height

        # Draw target height marker
        target_pos = base_pos_w.copy()
        target_pos[2] = standing_height + target_height

        # Vertical line from standing to target
        start_pos = base_pos_w.copy()
        start_pos[2] = standing_height

        visualizer.add_arrow(
            start_pos,
            target_pos,
            color=(0.2, 0.8, 0.2, 0.7),
            width=0.02,
        )


@dataclass(kw_only=True)
class JumpCommandCfg(CommandTermCfg):
    """Configuration for jump commands."""

    asset_name: str
    class_type: type[CommandTerm] = JumpCommand

    # Contact sensor name for detecting landing
    contact_sensor_name: str | None = None

    # Standing height of the robot (for computing target)
    standing_height: float = 0.76  # G1 knees-bent keyframe height

    # How fast the trigger decays after landing (0-1, lower = faster decay)
    trigger_decay_rate: float = 0.95

    @dataclass
    class Ranges:
        """Ranges for jump command sampling."""

        target_height: tuple[float, float] = (0.02, 0.05)

    ranges: Ranges = field(default_factory=Ranges)

    @dataclass
    class VizCfg:
        """Visualization configuration."""

        enabled: bool = True

    viz: VizCfg = field(default_factory=VizCfg)
