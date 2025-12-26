"""Unitree G1 jump environment configuration."""

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.jump.jump_env_cfg import make_jump_env_cfg

##
# Jump-Ready Crouch Keyframe
##

JUMP_CROUCH_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.55),  # Lower CoM than HOME (0.783675)
  joint_pos={
    # Lower body - aggressive crouch for explosive power
    ".*_hip_pitch_joint": -0.6,  # Deep hip flexion (-34°)
    ".*_knee_joint": 1.2,  # Deep knee flexion (~69°)
    ".*_ankle_pitch_joint": -0.6,  # Ankle dorsiflexion for power storage
    # Hip stability
    ".*_hip_roll_joint": 0.0,
    ".*_hip_yaw_joint": 0.0,
    ".*_ankle_roll_joint": 0.0,
    # Waist - slight forward lean
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.15,  # Slight forward lean for jump prep
    # Arms - ready for counterbalance swing
    ".*_shoulder_pitch_joint": -0.5,  # Arms back for upswing
    ".*_shoulder_roll_joint": 0.0,
    "left_shoulder_roll_joint": 0.3,
    "right_shoulder_roll_joint": -0.3,
    ".*_shoulder_yaw_joint": 0.0,
    ".*_elbow_joint": 0.8,  # Elbows bent
    ".*_wrist_pitch_joint": 0.0,
    ".*_wrist_roll_joint": 0.0,
    ".*_wrist_yaw_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)


def unitree_g1_jump_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 jump environment configuration.

  Args:
    play: If True, configure for deployment/evaluation mode.

  Returns:
    Jump environment configuration for G1 robot.
  """
  cfg = make_jump_env_cfg()

  # Configure G1 robot with jump-ready crouch pose
  robot_cfg = get_g1_robot_cfg()
  robot_cfg.init_state = JUMP_CROUCH_KEYFRAME
  cfg.scene.entities = {"robot": robot_cfg}

  # Site names for feet
  site_names = ("left_foot", "right_foot")

  # Contact sensors for jump detection and landing
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,  # Critical for jump detection
  )

  cfg.scene.sensors = (feet_ground_cfg,)

  # Action scaling for G1 actuators
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  # Viewer configuration
  cfg.viewer.body_name = "torso_link"

  # Update observation configs with G1-specific site names
  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  # Update reward configs with G1-specific body names
  cfg.rewards["upright_in_flight"].params["asset_cfg"].body_names = ("torso_link",)

  # Apply play mode overrides
  if play:
    # Effectively infinite episode length for demonstration
    cfg.episode_length_s = int(1e9)

    # Disable observation corruption (noise)
    cfg.observations["policy"].enable_corruption = False

    # Remove domain randomization events
    cfg.events.clear()

  return cfg
