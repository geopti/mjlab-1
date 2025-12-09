.. _new-task:

Creating a Task
===============

In this tutorial, you will create a **CartPole reinforcement learning task**
inside your external project. You will:

- define the MDP (actions, observations, rewards, events, terminations)
- configure the simulation and viewer
- register the task with Gym
- train and evaluate a policy

Prerequisites
^^^^^^^^^^^^^

You should already have:

- created your external project (:ref:`ext-project`)
- added the CartPole robot (:ref:`new-robot`)

Step 1 - Enter your project
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd path/to/mjlab_cookbook_project

Step 2 - Create the task folder structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You will define a single environment configuration module,
``cartpole_env_cfg.py``, and an ``__init__.py`` that registers the Gym
environment.

Create the following structure under ``src/mjlab_cookbook_project/tasks``:

.. code-block:: bash

   src/mjlab_cookbook_project/tasks/
     cartpole/
       cartpole_env_cfg.py
       __init__.py

Concretely:

.. code-block:: bash

   mkdir -p src/mjlab_cookbook_project/tasks/cartpole
   touch src/mjlab_cookbook_project/tasks/cartpole/__init__.py
   touch src/mjlab_cookbook_project/tasks/cartpole/cartpole_env_cfg.py

The rest of this tutorial will build up the contents of
``cartpole_env_cfg.py`` step by step.

Step 3 - Imports, scene, and viewer setup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Start by adding the core imports, scene configuration, and viewer configuration.

Edit ``cartpole_env_cfg.py`` and add:

.. code-block:: python

   """CartPole task environment configuration."""

   import math
   from dataclasses import dataclass, field

   import torch

   from mjlab.envs import ManagerBasedRlEnvCfg
   from mjlab.envs import mdp
   from mjlab.managers.manager_term_config import (
       ObservationGroupCfg as ObsGroup,
       ObservationTermCfg as ObsTerm,
       RewardTermCfg as RewardTerm,
       TerminationTermCfg as DoneTerm,
       EventTermCfg as EventTerm,
       term,
   )
   from mjlab.managers.scene_entity_config import SceneEntityCfg
   from mjlab.rl import RslRlOnPolicyRunnerCfg
   from mjlab.scene import SceneCfg
   from mjlab.sim import MujocoCfg, SimulationCfg
   from mjlab.viewer import ViewerConfig
   from mjlab_cookbook_project.robots import CARTPOLE_ROBOT_CFG

   # Scene with a single CartPole robot replicated across multiple environments.
   SCENE_CFG = SceneCfg(
       num_envs=64,
       extent=1.0,
       entities={"robot": CARTPOLE_ROBOT_CFG},
   )

   # Camera configuration for inspecting the CartPole.
   VIEWER_CONFIG = ViewerConfig(
       origin_type=ViewerConfig.OriginType.ASSET_BODY,
       asset_name="robot",
       body_name="pole",
       distance=3.0,
       elevation=10.0,
       azimuth=90.0,
   )

Step 4 - Define the actions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The policy will output position commands for the cart’s sliding joint. These are
scaled by ``20.0`` to match the actuator’s control range.

Append to ``cartpole_env_cfg.py``:

.. code-block:: python

   @dataclass
   class ActionCfg:
       joint_pos: mdp.JointPositionActionCfg = term(
           mdp.JointPositionActionCfg,
           asset_name="robot",
           actuator_names=[".*"],
           scale=20.0,
           use_default_offset=False,
       )

Step 5 - Define the observations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The policy observes:

- pole angle (normalized by :math:`\pi`)
- pole angular velocity (normalized)
- cart position (normalized)
- cart velocity (normalized)

Append:

.. code-block:: python

   @dataclass
   class ObservationCfg:
       @dataclass
       class PolicyCfg(ObsGroup):
           angle: ObsTerm = term(
               ObsTerm,
               func=lambda env: env.sim.data.qpos[:, 1:2] / math.pi,
           )
           ang_vel: ObsTerm = term(
               ObsTerm,
               func=lambda env: env.sim.data.qvel[:, 1:2] / 5.0,
           )
           cart_pos: ObsTerm = term(
               ObsTerm,
               func=lambda env: env.sim.data.qpos[:, 0:1] / 2.0,
           )
           cart_vel: ObsTerm = term(
               ObsTerm,
               func=lambda env: env.sim.data.qvel[:, 0:1] / 20.0,
           )

       @dataclass
       class CriticCfg(PolicyCfg):
           pass

       policy: PolicyCfg = field(default_factory=PolicyCfg)
       critic: CriticCfg = field(default_factory=CriticCfg)

Step 6 - Define the rewards
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We use two reward terms:

- ``upright``: encourages the pole to stay vertical using the cosine of the
  angle
- ``effort``: penalizes large control inputs

Append:

.. code-block:: python

   def compute_upright_reward(env):
       # Cosine of the pole angle: 1.0 when upright, decreasing as it tips.
       return env.sim.data.qpos[:, 1].cos()

   def compute_effort_penalty(env):
       # Small penalty on squared control to discourage aggressive actions.
       return -0.01 * (env.sim.data.ctrl[:, 0] ** 2)

   @dataclass
   class RewardCfg:
       upright: RewardTerm = term(
           RewardTerm,
           func=compute_upright_reward,
           weight=5.0,
       )
       effort: RewardTerm = term(
           RewardTerm,
           func=compute_effort_penalty,
           weight=1.0,
       )

Step 7 - Define events (resets and perturbations)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Events allow you to reset the state or apply perturbations at specific times.

Here, we:

- reset joints at episode start
- periodically apply random pushes to the cart to improve robustness

Append:

.. code-block:: python

   def random_push_cart(env, env_ids, force_range=(-5, 5)):
       """Apply random horizontal forces to the cart."""
       n = len(env_ids)
       random_forces = (
           torch.rand(n, device=env.device)
           * (force_range[1] - force_range[0])
           + force_range[0]
       )
       env.sim.data.qfrc_applied[env_ids, 0] = random_forces

   @dataclass
   class EventCfg:
       reset_robot_joints: EventTerm = term(
           EventTerm,
           func=mdp.reset_joints_by_scale,
           mode="reset",
           params={
               "asset_cfg": SceneEntityCfg("robot"),
               "position_range": (-0.1, 0.1),
               "velocity_range": (-0.1, 0.1),
           },
       )
       random_push: EventTerm = term(
           EventTerm,
           func=random_push_cart,
           mode="interval",
           interval_range_s=(1.0, 2.0),
           params={"force_range": (-20.0, 20.0)},
       )

Step 8 - Define terminations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We terminate episodes when:

- the pole tips beyond ±30°
- the maximum episode length (10 seconds) is reached

Append:

.. code-block:: python

   def check_pole_tipped(env):
       # Terminate when the pole angle goes beyond ±30 degrees.
       return env.sim.data.qpos[:, 1].abs() > math.radians(30)

   @dataclass
   class TerminationCfg:
       timeout: DoneTerm = term(
           DoneTerm,
           func=lambda env: False,
           time_out=True,
       )
       tipped: DoneTerm = term(
           DoneTerm,
           func=check_pole_tipped,
           time_out=False,
       )

Step 9 - Combine everything into the environment configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now you tie together:

- simulation config
- scene
- MDP components (actions, observations, rewards, events, terminations)
- viewer config

Append:

.. code-block:: python

   SIM_CFG = SimulationCfg(
       mujoco=MujocoCfg(
           timestep=0.02,
           iterations=1,
       ),
   )

   @dataclass
   class CartPoleEnvCfg(ManagerBasedRlEnvCfg):
       scene: SceneCfg = field(default_factory=lambda: SCENE_CFG)
       observations: ObservationCfg = field(default_factory=ObservationCfg)
       actions: ActionCfg = field(default_factory=ActionCfg)
       rewards: RewardCfg = field(default_factory=RewardCfg)
       events: EventCfg = field(default_factory=EventCfg)
       terminations: TerminationCfg = field(default_factory=TerminationCfg)
       sim: SimulationCfg = field(default_factory=lambda: SIM_CFG)
       viewer: ViewerConfig = field(default_factory=lambda: VIEWER_CONFIG)
       decimation: int = 1
       episode_length_s: float = 10.0

Step 10 - Register the task with Gym
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, register the environment so that it can be created by Gym (and by
mjlab’s training scripts) via an ID.

Edit ``src/mjlab_cookbook_project/tasks/cartpole/__init__.py``:

.. code-block:: python

   import gymnasium as gym

   gym.register(
       id="Mjlab-Cartpole",
       entry_point="mjlab.envs:ManagerBasedRlEnv",
       disable_env_checker=True,
       kwargs={
           "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:CartPoleEnvCfg",
           "rl_cfg_entry_point": f"{__name__}.cartpole_env_cfg:RslRlOnPolicyRunnerCfg",
       },
   )

This tells Gym how to construct your environment given the ID
``"Mjlab-Cartpole"``.

Step 11 - Train the CartPole policy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can now train a policy on the CartPole task.
Here we increase the number of parallel environments to speed up learning:

.. code-block:: bash

   uv run train Mjlab-Cartpole --env.scene.num-envs 1024

Step 12 - Evaluate a trained checkpoint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once training has produced a checkpoint, you can run it in inference mode:

.. code-block:: bash

   uv run play Mjlab-Cartpole --checkpoint_file <checkpoint_path>

Replace ``<checkpoint_path>`` with the path to your saved checkpoint file.

.. image:: ../_static/content/cartpole_trained.gif
   :width: 100%
   :alt: Trained CartPole

You now have a complete CartPole task defined in your external project, ready to
be used as a template for more complex robots and environments.
