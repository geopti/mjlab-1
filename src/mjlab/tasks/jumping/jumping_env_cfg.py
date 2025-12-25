"""Jumping task configuration.

This module provides a factory function to create a base jumping task config.
Robot-specific configurations call the factory and customize as needed.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import (
    ActionTermCfg,
    CommandTermCfg,
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.jumping import mdp
from mjlab.tasks.jumping.mdp import JumpCommandCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def make_jumping_env_cfg() -> ManagerBasedRlEnvCfg:
    """Create base jumping task configuration."""

    ##
    # Observations
    ##

    policy_terms = {
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "command": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "jump"},
        ),
        "pelvis_height": ObservationTermCfg(
            func=mdp.pelvis_height,
            noise=Unoise(n_min=-0.02, n_max=0.02),
        ),
        "pelvis_vertical_velocity": ObservationTermCfg(
            func=mdp.pelvis_vertical_velocity,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        ),
    }

    critic_terms = {
        **policy_terms,
        "foot_height": ObservationTermCfg(
            func=mdp.foot_height,
            params={"asset_cfg": SceneEntityCfg("robot", site_names=())},  # Set per-robot.
        ),
        "foot_contact": ObservationTermCfg(
            func=mdp.foot_contact,
            params={"sensor_name": "feet_ground_contact"},
        ),
        "foot_contact_forces": ObservationTermCfg(
            func=mdp.foot_contact_forces,
            params={"sensor_name": "feet_ground_contact"},
        ),
    }

    observations = {
        "policy": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=True,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    ##
    # Actions
    ##

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            asset_name="robot",
            actuator_names=(".*",),
            scale=0.5,  # Override per-robot.
            use_default_offset=True,
        )
    }

    ##
    # Commands
    ##

    commands: dict[str, CommandTermCfg] = {
        "jump": JumpCommandCfg(
            asset_name="robot",
            resampling_time_range=(3.0, 6.0),
            contact_sensor_name="feet_ground_contact",
            standing_height=0.76,  # Override per-robot if needed.
            ranges=JumpCommandCfg.Ranges(
                target_height=(0.02, 0.05),  # Start small, curriculum increases this.
            ),
        )
    }

    ##
    # Events
    ##

    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
    }

    ##
    # Rewards
    ##

    rewards = {
        # Primary: Jump height reward (sparse, on landing)
        "jump_height": RewardTermCfg(
            func=mdp.jump_height_reward,
            weight=5.0,
            params={
                "command_name": "jump",
                "sensor_name": "feet_ground_contact",
                "std": 0.1,
                "standing_height": 0.76,  # Override per-robot.
            },
        ),
        # Dense: Reward height during flight
        "continuous_height": RewardTermCfg(
            func=mdp.continuous_jump_height,
            weight=2.0,
            params={
                "sensor_name": "feet_ground_contact",
                "standing_height": 0.76,  # Override per-robot.
            },
        ),
        # Launch velocity when push off
        "launch_velocity": RewardTermCfg(
            func=mdp.launch_velocity_reward,
            weight=1.0,
            params={
                "command_name": "jump",
                "sensor_name": "feet_ground_contact",
            },
        ),
        # Stay upright
        "upright": RewardTermCfg(
            func=mdp.flat_orientation,
            weight=1.0,
            params={
                "std": math.sqrt(0.2),
                "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
            },
        ),
        # Stable landing
        "stable_landing": RewardTermCfg(
            func=mdp.stable_landing_reward,
            weight=1.0,
            params={
                "sensor_name": "feet_ground_contact",
                "std": 0.2,
            },
        ),
        # Soft landing (penalize high impact)
        "soft_landing": RewardTermCfg(
            func=mdp.soft_landing,
            weight=-1e-4,
            params={
                "sensor_name": "feet_ground_contact",
            },
        ),
        # Smooth actions
        "action_rate_l2": RewardTermCfg(
            func=mdp.action_rate_l2,
            weight=-0.1,
        ),
        # Joint limits
        "dof_pos_limits": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-1.0,
        ),
        # Penalize horizontal drift
        "horizontal_drift": RewardTermCfg(
            func=mdp.horizontal_drift_penalty,
            weight=-0.3,
        ),
        # Penalize rotation during flight
        "flight_rotation": RewardTermCfg(
            func=mdp.excessive_rotation_penalty,
            weight=-0.5,
            params={"sensor_name": "feet_ground_contact"},
        ),
        # Penalize feet slip
        "foot_slip": RewardTermCfg(
            func=mdp.feet_slip_penalty,
            weight=-0.1,
            params={
                "sensor_name": "feet_ground_contact",
                "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
            },
        ),
    }

    ##
    # Terminations
    ##

    terminations = {
        "time_out": TerminationTermCfg(
            func=mdp.time_out,
            time_out=True,
        ),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(70.0)},
        ),
        "fell_down": TerminationTermCfg(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.3},
        ),
    }

    ##
    # Curriculum
    ##

    curriculum = {
        "jump_height": CurriculumTermCfg(
            func=mdp.jump_height_curriculum,
            params={
                "command_name": "jump",
                "height_stages": [
                    {"step": 0, "target_height": (0.02, 0.05)},
                    {"step": 5000 * 24, "target_height": (0.05, 0.10)},
                    {"step": 15000 * 24, "target_height": (0.08, 0.15)},
                    {"step": 30000 * 24, "target_height": (0.10, 0.25)},
                ],
            },
        ),
    }

    ##
    # Assemble and return
    ##

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainImporterCfg(
                terrain_type="plane",  # Flat terrain for jumping
                terrain_generator=None,
            ),
            num_envs=1,
            extent=2.0,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            asset_name="robot",
            body_name="",  # Set per-robot.
            distance=3.0,
            elevation=-5.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=35,
            njmax=300,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
            ),
        ),
        decimation=4,
        episode_length_s=10.0,  # Shorter episodes for jumping practice
    )
