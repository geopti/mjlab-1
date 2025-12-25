"""Jumping task for humanoid robots."""

from mjlab.tasks.jumping.jumping_env_cfg import make_jumping_env_cfg

# Import configs to register tasks.
from mjlab.tasks.jumping.config import g1 as g1_config

__all__ = ["make_jumping_env_cfg", "g1_config"]
