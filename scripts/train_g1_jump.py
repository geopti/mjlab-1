#!/usr/bin/env python
"""Train Unitree G1 to perform vertical jumps.

This script trains the G1 humanoid robot to jump vertically 20-30cm
and land stably within 50,000 iterations (~6-12 hours on NVIDIA T4).

Usage:
    uv run scripts/train_g1_jump.py

    Or use the mjlab CLI directly:
    uv run train --task Mjlab-Jump-Flat-Unitree-G1

The trained model will be saved in logs/rsl_rl/g1_jump/
"""

import subprocess
import sys

if __name__ == "__main__":
  # Run using the registered CLI command
  subprocess.run(
    ["uv", "run", "train", "--task", "Mjlab-Jump-Flat-Unitree-G1"],
    check=True
  )
