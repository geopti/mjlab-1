.. _faq:

FAQ & Troubleshooting
=====================

This page answers common questions about platform support, performance,
training stability, rendering, and known limitations.

.. contents::
   :local:
   :depth: 1

Platform support
----------------

Does it work on macOS?
   Yes, but only with limited performance. mjlab runs on macOS using
   CPU-only execution through MuJoCo Warp.

   - **Training is not recommended on macOS** as it lacks GPU acceleration.
   - **Evaluation works**, but is significantly slower than on Linux with CUDA.

   For serious training workloads, we strongly recommend Linux with an NVIDIA GPU.

Does it work on Windows?
   mjlab relies on MuJoCo Warp, which supports Windows. However, support
   should be considered **experimental**, and we have not validated all
   configurations internally. Community reports and contributions for Windows
   support are very welcome.

What CUDA versions are supported?
   Not all CUDA versions are supported by MuJoCo Warp.

   - See the MuJoCo Warp CUDA compatibility discussion:
     https://github.com/google-deepmind/mujoco_warp/issues/101
   - **Recommended**: CUDA 12.4+ (for conditional execution in CUDA graphs).

Performance
-----------

Is it faster than Isaac Lab?
   Based on our experience so far, mjlab is on par with or faster than Isaac
   Lab for comparable tasks, thanks to the direct MuJoCo Warp backend and the
   absence of Omniverse overhead. Exact speedups depend on the task and
   hardware configuration.

What GPU do you recommend?
   For RL workloads, we recommend:

   - RTX 40-series GPUs (or newer) for single-machine experiments,
   - L40S, H100, or similar data-center GPUs for large-scale training.

Training & debugging
--------------------

My training crashes with NaN errors. What does this mean?
   A common error when using ``rsl_rl`` is:

   .. code-block:: text

      RuntimeError: normal expects all elements of std >= 0.0

   This usually occurs when NaN/Inf values in the physics state propagate into
   the policy network, causing its output standard deviation to become
   negative or NaN.

   There are many possible causes, including potential bugs in MuJoCo Warp
   (which is still in beta). mjlab provides two complementary tools:

   **1. For training stability – add a NaN termination**

   Add a ``nan_detection`` termination term to reset environments where NaNs
   appear:

   .. code-block:: python

      from dataclasses import dataclass, field

      from mjlab.envs.mdp.terminations import nan_detection
      from mjlab.managers.manager_term_config import TerminationTermCfg

      @dataclass
      class TerminationCfg:
          # Your other termination terms...
          nan_term: TerminationTermCfg = field(
              default_factory=lambda: TerminationTermCfg(
                  func=nan_detection,
                  time_out=False,
              )
          )

   This marks NaN environments as terminated so they reset while training
   continues. Terminations are logged as
   ``Episode_Termination/nan_term`` in your metrics.

   .. attention::

      This is a **band-aid**, not a root-cause fix. If NaNs systematically
      occur in states that matter for your task (e.g., during grasping in a
      manipulation task), the policy may never learn that behavior. Always
      investigate the cause using ``nan_guard`` as well.

   **2. For debugging – enable NaN guard**

   Use the ``nan_guard`` tool to capture the simulation state when NaNs occur:

   .. code-block:: bash

      uv run train --enable-nan-guard True

   See :doc:`NaN Guard documentation <../core/nan_guard>` for detailed debugging
   instructions. ``nan_guard`` helps you record the exact states that trigger
   NaNs, making it easier to build a minimal reproducible example (MRE). If
   you suspect a framework bug, these captured states are extremely useful
   when reporting issues to the
   `MuJoCo Warp team <https://github.com/google-deepmind/mujoco_warp/issues>`_.

Rendering & visualization
-------------------------

What visualization options are available?
   We currently support two visualizers for policy evaluation and debugging:

   - **Native MuJoCo visualizer** – the built-in MuJoCo viewer,
   - **`Viser <https://github.com/nerfstudio-project/viser>`_** – a web-based 3D
     visualizer.

   We are exploring options for **training-time visualization** (e.g., live
   rollout viewers), but this is not yet available. As a current alternative,
   mjlab supports **video logging to Weights & Biases (W&B)**, so you can
   monitor rollout videos directly in your experiment dashboard.

What about camera/pixel rendering for vision-based RL?
   Camera rendering for pixel-based agents is not yet available. The MuJoCo
   Warp team is actively developing camera support; mjlab will integrate this
   functionality once it becomes available.

Assets & compatibility
----------------------

What robots are included?
   mjlab includes two reference robots:

   - **Unitree Go1** (quadruped),
   - **Unitree G1** (humanoid).

   These serve as integration examples and support our reference tasks for
   testing. We intentionally keep mjlab lean and lightweight, so we do not
   plan to grow a large built-in robot library. Additional robots may be
   provided in a separate repository.

Can I use USD or URDF models?
   No. mjlab requires MJCF (MuJoCo XML) models. You will need to convert USD
   or URDF assets to MJCF.

   `MuJoCo Menagerie <https://github.com/google-deepmind/mujoco_menagerie>`_
   provides a large collection of pre-converted robot MJCFs that you can use.

Known limitations
-----------------

What are the current limitations?
   We track missing features and blockers for the first stable release in:

   - mjlab roadmap issue: https://github.com/mujocolab/mjlab/issues/100
   - General issues: https://github.com/mujocolab/mjlab/issues

   If something is not working, or you think a feature is missing, please open
   an issue:

   - Bug reports: https://github.com/mujocolab/mjlab/issues/new

   .. attention::

      **Reminder**: mjlab is in **beta**. Breaking changes and missing features
      are expected at this stage. Feedback, bug reports, and contributions are
      very welcome.
