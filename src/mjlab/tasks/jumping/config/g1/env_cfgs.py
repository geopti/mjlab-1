"""Unitree G1 jumping environment configurations."""

from mjlab.asset_zoo.robots import (
    G1_ACTION_SCALE,
    get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.jumping.jumping_env_cfg import make_jumping_env_cfg


def unitree_g1_jumping_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Unitree G1 flat terrain jumping configuration."""
    cfg = make_jumping_env_cfg()

    # Set robot entity
    cfg.scene.entities = {"robot": get_g1_robot_cfg()}

    # Foot site names for G1
    site_names = ("left_foot", "right_foot")
    geom_names = tuple(
        f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
    )

    # Configure feet ground contact sensor
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
        track_air_time=True,
    )
    cfg.scene.sensors = (feet_ground_cfg,)

    # Set G1-specific action scale
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = G1_ACTION_SCALE

    # Set viewer to track torso
    cfg.viewer.body_name = "torso_link"

    # Configure G1-specific observation parameters
    cfg.observations["critic"].terms["foot_height"].params[
        "asset_cfg"
    ].site_names = site_names

    # Configure G1-specific reward parameters
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
    cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

    # G1 standing height from KNEES_BENT_KEYFRAME
    standing_height = 0.76
    cfg.rewards["jump_height"].params["standing_height"] = standing_height
    cfg.rewards["continuous_height"].params["standing_height"] = standing_height

    # Update command standing height
    assert cfg.commands is not None
    cfg.commands["jump"].standing_height = standing_height

    # Apply play mode overrides
    if play:
        # Effectively infinite episode length
        cfg.episode_length_s = int(1e9)

        # Disable observation corruption for cleaner evaluation
        cfg.observations["policy"].enable_corruption = False

    return cfg
