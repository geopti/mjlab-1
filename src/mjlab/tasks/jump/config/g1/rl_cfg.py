"""RL configuration for Unitree G1 jump task."""

from mjlab.rl import (
  RslRlOnPolicyRunnerCfg,
  RslRlPpoActorCriticCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_g1_jump_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 jump task.

  Hyperparameters are tuned specifically for jumping:
  - Smaller actor network (jumping is simpler than omni-directional walking)
  - Larger critic network (needs accurate landing value prediction)
  - Higher value_loss_coef (landing success is critical)
  - Lower learning rate (more conservative for stability)
  - Gamma 0.98 (short episodes don't need long horizon)

  Returns:
    RL runner configuration optimized for 50k iteration training budget.
  """
  return RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
      init_noise_std=1.0,  # Encourage exploration initially
      actor_obs_normalization=True,
      critic_obs_normalization=True,
      # Actor: smaller network for simpler jumping task
      actor_hidden_dims=(256, 128, 64),
      # Critic: larger network for complex landing prediction
      critic_hidden_dims=(512, 256, 128),
      activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=2.0,  # Higher than walking - landing value crucial
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.015,  # Slightly higher for exploration
      num_learning_epochs=6,  # More epochs per iteration
      num_mini_batches=4,
      learning_rate=3e-4,  # Conservative for stable learning
      schedule="adaptive",
      gamma=0.98,  # Short episodes (3-5s)
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_jump",
    save_interval=100,  # Save every 100 iterations
    num_steps_per_env=24,  # With 4096 envs = 98,304 samples/iter
    max_iterations=1000,  # TESTING: 1k iterations (change back to 50_000 for full training)
  )
