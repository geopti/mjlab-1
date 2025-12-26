"""Jump task configuration.

This module provides a factory function to create a base jump task config.
Robot-specific configurations call the factory and customize as needed.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import events as base_events
from mjlab.envs.mdp import observations as base_obs
from mjlab.envs.mdp import rewards as base_rewards
from mjlab.envs.mdp import terminations as base_terms
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
from mjlab.tasks.jump import mdp
from mjlab.tasks.jump.mdp import JumpCommandCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def make_jump_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base jump task configuration."""

  ##
  # Observations
  ##

  policy_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=base_obs.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=base_obs.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=base_obs.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=base_obs.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=base_obs.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=base_obs.last_action),
    # Jump-specific observations
    "height_above_ground": ObservationTermCfg(func=mdp.height_above_ground),
    "vertical_velocity": ObservationTermCfg(func=mdp.vertical_velocity),
    "contact_state": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "time_in_air": ObservationTermCfg(
      func=mdp.foot_air_time,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "command": ObservationTermCfg(
      func=base_obs.generated_commands,
      params={"command_name": "jump"},
    ),
  }

  critic_terms = {
    **policy_terms,
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=())},  # Set per-robot.
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
      target_height=0.25,  # 25cm default, updated by curriculum
      height_tolerance=0.05,  # 5cm tolerance
    )
  }

  ##
  # Events
  ##

  events = {
    "reset_base": EventTermCfg(
      func=base_events.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)},
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=base_events.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.1, 0.1),  # Small variation around crouch pose
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards = {
    # ===== PRIMARY OBJECTIVE: HEIGHT =====
    "jump_height": RewardTermCfg(
      func=mdp.jump_height_reward,
      weight=10.0,
      params={
        "target_height": 0.25,
        "std": 0.15,
      },
    ),
    # ===== TAKEOFF PHASE =====
    "explosive_takeoff": RewardTermCfg(
      func=mdp.explosive_takeoff,
      weight=3.0,
      params={
        "sensor_name": "feet_ground_contact",
        "power_threshold": 500.0,
      },
    ),
    "synchronized_extension": RewardTermCfg(
      func=mdp.synchronized_extension,
      weight=-2.0,  # Penalty for asymmetry
    ),
    "vertical_impulse": RewardTermCfg(
      func=mdp.vertical_impulse,
      weight=2.0,
      params={"sensor_name": "feet_ground_contact"},
    ),
    # ===== FLIGHT PHASE =====
    "air_time_bonus": RewardTermCfg(
      func=mdp.air_time_bonus,
      weight=1.5,
      params={
        "sensor_name": "feet_ground_contact",
        "min_air_time": 0.2,
      },
    ),
    "upright_in_flight": RewardTermCfg(
      func=velocity_mdp.flat_orientation,
      weight=3.0,
      params={
        "std": math.sqrt(0.3),
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
      },
    ),
    "angular_momentum_control": RewardTermCfg(
      func=velocity_mdp.angular_momentum_penalty,
      weight=-0.5,
      params={"sensor_name": "robot/root_angmom"},
    ),
    # ===== LANDING PHASE =====
    "soft_landing": RewardTermCfg(
      func=velocity_mdp.soft_landing,
      weight=-2.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": None,  # Always active
      },
    ),
    "landing_stability": RewardTermCfg(
      func=mdp.landing_balance,
      weight=4.0,  # Will be increased by curriculum
      params={
        "sensor_name": "feet_ground_contact",
        "stability_time": 0.5,
      },
    ),
    "symmetric_landing": RewardTermCfg(
      func=mdp.symmetric_landing,
      weight=1.0,
      params={
        "sensor_name": "feet_ground_contact",
        "time_tolerance": 0.05,
      },
    ),
    # ===== REGULARIZATION =====
    "action_rate_l2": RewardTermCfg(
      func=base_rewards.action_rate_l2,
      weight=-0.05,  # Lower than walking to allow aggressive movements
    ),
    "action_smoothness": RewardTermCfg(
      func=base_rewards.action_acc_l2,
      weight=-0.01,
    ),
    "joint_torques_l2": RewardTermCfg(
      func=base_rewards.joint_torques_l2,
      weight=-1e-5,  # Very low - allow high torques
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "dof_pos_limits": RewardTermCfg(
      func=base_rewards.joint_pos_limits,
      weight=-5.0,
    ),
    # ===== EFFICIENCY =====
    "alive": RewardTermCfg(
      func=base_rewards.is_alive,
      weight=0.5,
    ),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=base_terms.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=base_terms.bad_orientation,
      params={"limit_angle": math.radians(60.0)},  # More lenient than walking
    ),
    "height_too_low": TerminationTermCfg(
      func=base_terms.root_height_below_minimum,
      params={
        "minimum_height": 0.35,  # Below crouch height
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "excessive_impact": TerminationTermCfg(
      func=mdp.excessive_landing_force,
      params={
        "sensor_name": "feet_ground_contact",
        "force_threshold": 2500.0,
      },
    ),
  }

  ##
  # Curriculum
  ##

  curriculum = {
    "jump_height_progression": CurriculumTermCfg(
      func=mdp.progressive_jump_height,
      params={
        "command_name": "jump",
        "height_stages": [
          {"step": 0, "target_height": 0.10, "tolerance": 0.05},  # 10cm
          {"step": 10000 * 24, "target_height": 0.15, "tolerance": 0.05},  # 15cm
          {"step": 20000 * 24, "target_height": 0.20, "tolerance": 0.05},  # 20cm
          {"step": 35000 * 24, "target_height": 0.25, "tolerance": 0.08},  # 25cm
        ],
      },
    ),
    "landing_stability_progression": CurriculumTermCfg(
      func=mdp.progressive_stability_requirement,
      params={
        "reward_name": "landing_stability",
        "weight_stages": [
          {"step": 0, "weight": 1.0},  # Early: low weight
          {"step": 15000 * 24, "weight": 2.5},  # Mid: increase
          {"step": 30000 * 24, "weight": 4.0},  # Late: high weight
        ],
      },
    ),
    # Note: Episode length progression removed - cannot dynamically change episode length
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainImporterCfg(terrain_type="plane"),  # Flat terrain
      num_envs=4096,
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
      distance=2.0,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=300,
      mujoco=MujocoCfg(
        timestep=0.002,  # 2ms for fine jump dynamics
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=2,  # 4ms control frequency (500Hz effective)
    episode_length_s=5.0,  # 5 seconds for jump + landing practice
  )
