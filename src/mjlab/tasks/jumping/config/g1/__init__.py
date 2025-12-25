"""Unitree G1 jumping task configurations."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import unitree_g1_jumping_env_cfg
from .rl_cfg import unitree_g1_jumping_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Jumping-Flat-Unitree-G1",
    env_cfg=unitree_g1_jumping_env_cfg(),
    play_env_cfg=unitree_g1_jumping_env_cfg(play=True),
    rl_cfg=unitree_g1_jumping_ppo_runner_cfg(),
    runner_cls=None,  # Use default OnPolicyRunner
)
