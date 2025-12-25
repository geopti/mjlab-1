"""MDP components for jumping task."""

# Re-export everything from envs.mdp (observations, rewards, terminations, events, actions)
from mjlab.envs.mdp import *  # noqa: F401, F403

# Re-export velocity task MDP components we need
from mjlab.tasks.velocity.mdp.observations import (
    foot_air_time,
    foot_contact,
    foot_contact_forces,
    foot_height,
)
from mjlab.tasks.velocity.mdp.rewards import (
    flat_orientation,
    soft_landing,
)

# Export jumping-specific components
from .commands import JumpCommand, JumpCommandCfg
from .curriculums import jump_height_curriculum, reward_weight_curriculum
from .observations import (
    both_feet_in_contact,
    feet_in_contact,
    pelvis_height,
    pelvis_height_relative,
    pelvis_vertical_velocity,
)
from .rewards import (
    continuous_jump_height,
    excessive_rotation_penalty,
    feet_slip_penalty,
    horizontal_drift_penalty,
    jump_height_reward,
    launch_velocity_reward,
    stable_landing_reward,
)

__all__ = [
    # Commands
    "JumpCommand",
    "JumpCommandCfg",
    # Observations (jumping-specific)
    "pelvis_height",
    "pelvis_vertical_velocity",
    "pelvis_height_relative",
    "feet_in_contact",
    "both_feet_in_contact",
    # Observations (from velocity task)
    "foot_height",
    "foot_air_time",
    "foot_contact",
    "foot_contact_forces",
    # Rewards (jumping-specific)
    "jump_height_reward",
    "launch_velocity_reward",
    "horizontal_drift_penalty",
    "excessive_rotation_penalty",
    "stable_landing_reward",
    "continuous_jump_height",
    "feet_slip_penalty",
    # Rewards (from velocity task)
    "flat_orientation",
    "soft_landing",
    # Curriculum
    "jump_height_curriculum",
    "reward_weight_curriculum",
]
