import mujoco

from mjlab.asset_zoo.robots import (
  YAM_ACTION_SCALE,
  get_yam_robot_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import ObservationGroupCfg, ObservationTermCfg
from mjlab.sensor import CameraSensorCfg, ContactSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg


def get_cube_spec(cube_size: float = 0.02, mass: float = 0.05) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  body.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(cube_size,) * 3,
    mass=mass,
    rgba=(0.8, 0.2, 0.2, 1.0),
  )
  return spec


def get_goal_spec(radius: float = 0.03) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="goal", mocap=True)
  body.add_geom(
    name="goal_geom",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=(radius,) * 3,
    rgba=(1.0, 0.5, 0.0, 0.3),
    contype=0,
    conaffinity=0,
  )
  return spec


def yam_lift_cube_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_lift_cube_env_cfg()

  cfg.scene.entities = {
    "robot": get_yam_robot_cfg(),
    "cube": EntityCfg(spec_fn=get_cube_spec),
    "goal": EntityCfg(spec_fn=get_goal_spec),
  }

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = YAM_ACTION_SCALE

  assert cfg.commands is not None
  lift_command = cfg.commands["lift_height"]
  assert isinstance(lift_command, LiftingCommandCfg)
  lift_command.goal_entity_name = "goal"

  cfg.observations["policy"].terms["ee_to_cube"].params["asset_cfg"].site_names = (
    "grasp_site",
  )
  cfg.rewards["lift"].params["asset_cfg"].site_names = ("grasp_site",)

  fingertip_geoms = r"[lr]f_down(6|7|8|9|10|11)_collision"
  cfg.events["fingertip_friction_slide"].params[
    "asset_cfg"
  ].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

  # Configure collision sensor pattern.
  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "link_6"

  cfg.viewer.body_name = "arm"

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False

  return cfg


def yam_lift_cube_vision_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = yam_lift_cube_env_cfg(play=play)

  # Add camera sensor.
  camera_cfg = CameraSensorCfg(
    name="wrist_camera",
    camera_name="robot/camera_d405",
    width=64,
    height=64,
    type=("rgb",),
    update_period=0.04,  # 25 Hz.
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (camera_cfg,)

  # Add camera observation group.
  camera_obs = ObservationGroupCfg(
    {
      "camera_rgb": ObservationTermCfg(
        func=manipulation_mdp.camera_rgb,
        params={"sensor_name": "wrist_camera", "normalize": True},
      ),
    },
    enable_corruption=False,
    concatenate_terms=True,
  )
  cfg.observations["camera"] = camera_obs

  # Pop privileged info from policy observations.
  policy_obs = cfg.observations["policy"]
  policy_obs.terms.pop("ee_to_cube")
  policy_obs.terms.pop("cube_to_goal")

  return cfg
