#!/usr/bin/env python
"""Play a trained G1 jump policy.

This script loads a trained jumping policy and runs it in the simulator
for visualization and evaluation.

Usage:
    uv run scripts/play_g1_jump.py --checkpoint logs/rsl_rl/g1_jump/model_XXXXX.pt

    Or use the mjlab CLI directly:
    uv run play --task Mjlab-Jump-Flat-Unitree-G1 --checkpoint logs/rsl_rl/g1_jump/model_XXXXX.pt

Arguments:
    --checkpoint: Path to the trained model checkpoint
    --num_envs: Number of parallel environments to visualize (default: 16)
"""

import subprocess
import sys

if __name__ == "__main__":
  # Run using the registered CLI command
  # Pass through any additional arguments (like --checkpoint)
  args = ["uv", "run", "play", "--task", "Mjlab-Jump-Flat-Unitree-G1"] + sys.argv[1:]
  subprocess.run(args, check=True)
