# G1 Humanoid Vertical Jump Task

Train the Unitree G1 humanoid robot to perform vertical jumps of 20-30cm with stable landing in **50,000 iterations** (~6-12 hours on NVIDIA T4).

## 🎯 Task Overview

This task trains the G1 humanoid to:
- Jump vertically from a crouch position
- Achieve 20-30cm peak height
- Maintain stable orientation during flight
- Land softly with both feet
- Balance for 0.5s after landing

**Why this is novel**: Most humanoid RL research focuses on walking or motion tracking. Vertical jumping with landing stability is rarely demonstrated and requires complex phase-aware reward shaping.

## 🚀 Quick Start

### Train from Scratch

**Option 1: Using the training script**
```bash
uv run scripts/train_g1_jump.py
```

**Option 2: Using mjlab CLI directly**
```bash
uv run train --task Mjlab-Jump-Flat-Unitree-G1
```

This will:
- Train for 50,000 iterations
- Save checkpoints every 100 iterations to `logs/rsl_rl/g1_jump/`
- Log metrics to Weights & Biases (if configured)

### Visualize Trained Policy

**Option 1: Using the play script**
```bash
uv run scripts/play_g1_jump.py --checkpoint logs/rsl_rl/g1_jump/model_50000.pt
```

**Option 2: Using mjlab CLI directly**
```bash
uv run play --task Mjlab-Jump-Flat-Unitree-G1 --checkpoint logs/rsl_rl/g1_jump/model_50000.pt
```

## 📊 Expected Learning Curve

| Iterations | Expected Behavior |
|-----------|-------------------|
| 0-5k | Discover crouch extension, 5-10cm hops |
| 5-15k | Synchronized leg extension, 10-15cm jumps |
| 15-30k | Explosive takeoff mastered, 20cm consistently |
| 30-45k | Landing stability optimization, approach 25cm |
| 45-50k | Polish: 25-30cm jumps, 90% stable landings |

## 🎓 Key Design Decisions

### 1. **Jump-Ready Crouch Initial Pose**
Starting from a deep crouch (`src/mjlab/tasks/jump/config/g1/env_cfgs.py:16`):
- **Hip pitch**: -0.6 rad (-34°) - deep hip flexion
- **Knee**: 1.2 rad (69°) - deep knee flexion
- **Ankle pitch**: -0.6 rad - dorsiflexion for power storage
- **Arms**: Back position (-0.5 rad) for upswing momentum

**Rationale**: Eliminates 5-10k iterations of learning to crouch first.

### 2. **Phase-Aware Reward Structure**
Rewards are designed for three phases (`src/mjlab/tasks/jump/mdp/rewards.py`):

**Takeoff Phase** (on ground):
- `jump_height` (weight 10.0): Primary objective - exponential reward for peak height
- `explosive_takeoff` (3.0): High joint power during contact
- `synchronized_extension` (-2.0): Penalize left/right asymmetry
- `vertical_impulse` (2.0): Reward vertical ground reaction forces

**Flight Phase** (in air):
- `air_time_bonus` (1.5): Require actual flight (both feet off ground)
- `upright_in_flight` (3.0): Maintain orientation
- `angular_momentum_control` (-0.5): Prevent rotation

**Landing Phase** (recontact):
- `soft_landing` (-2.0): Minimize impact forces
- `landing_stability` (4.0): Balance for 0.5s post-landing
- `symmetric_landing` (1.0): Feet land simultaneously

### 3. **Progressive Height Curriculum**
(`src/mjlab/tasks/jump/mdp/curriculums.py:24`)

```
Iterations 0-10k:      10cm target (easily achievable)
Iterations 10-20k:     15cm target
Iterations 20-35k:     20cm target (minimum goal)
Iterations 35-50k:     25cm target (stretch goal)
```

**Rationale**: Avoids reward sparsity - agent discovers jumping behavior immediately at 10cm.

### 4. **Finer Simulation Timestep**
- **Timestep**: 2ms (vs 5ms for walking)
- **Decimation**: 2 (4ms control, 500Hz effective)

**Rationale**: Jumping involves rapid ballistic motion (takeoff ~0.2-0.3s) requiring fine temporal resolution.

### 5. **Optimized Hyperparameters**
(`src/mjlab/tasks/jump/config/g1/rl_cfg.py`)

- **Actor network**: (256, 128, 64) - smaller than walking (jumping is simpler)
- **Critic network**: (512, 256, 128) - larger (needs landing prediction)
- **Learning rate**: 3e-4 (conservative for stability)
- **value_loss_coef**: 2.0 (higher - landing value crucial)
- **Gamma**: 0.98 (short 3-5s episodes)
- **Entropy**: 0.015 (encourage exploration)

## 📁 File Structure

```
src/mjlab/tasks/jump/
├── __init__.py                    # Task package
├── jump_env_cfg.py                # Base environment configuration
├── mdp/
│   ├── commands.py                # JumpCommand (height targets)
│   ├── observations.py            # Jump-specific observations
│   ├── rewards.py                 # Phase-aware rewards ⭐
│   ├── curriculums.py             # Progressive height curriculum ⭐
│   └── terminations.py            # Excessive impact termination
└── config/g1/
    ├── env_cfgs.py                # G1 robot + JUMP_CROUCH_KEYFRAME ⭐
    ├── rl_cfg.py                  # PPO hyperparameters
    └── __init__.py                # Task registration
```

## 🔍 Monitoring Training

Key metrics logged to Weights & Biases:

```python
Metrics/peak_jump_height       # Max height achieved
Metrics/air_time_mean          # Average flight time
Metrics/landing_force_mean     # Impact forces
Metrics/landing_success_rate   # % stable landings
Curriculum/target_height       # Current curriculum target
```

## 🐛 Troubleshooting

### Not converging by 25k iterations?
1. Increase `jump_height` reward weight: 10.0 → 15.0
2. Decrease action penalties: -0.05 → -0.02
3. Add intermediate curriculum stages (12cm, 17cm, 22cm)
4. Increase exploration: `entropy_coef` 0.015 → 0.02

### Landing unstable?
1. Increase `landing_stability` weight: 4.0 → 6.0
2. Increase `upright_in_flight` weight: 3.0 → 5.0
3. Extend episode length for more landing practice

### Agent "cheats" by standing on tiptoes?
- `air_time_bonus` reward requires both feet off ground (detected via contact sensor)
- `jump_height` measures peak above initial crouch height

## 🎯 Success Criteria

### Minimum Viable Product (20k iterations):
- ✅ Consistent 15-20cm jumps
- ✅ Stable landing 70% of the time

### Target Goal (50k iterations):
- ✅ Consistent 20-30cm jumps
- ✅ Stable landing 90% of the time
- ✅ Symmetric takeoff and landing
- ✅ Flight time >0.2s

### Stretch Goal:
- ⭐ 30-35cm jumps
- ⭐ Running start into jump
- ⭐ Forward/backward jumping variants

## 🔬 Technical Details

### Observation Space (~103 dims)
**Policy** (with noise for sim-to-real):
- IMU data (lin_vel, ang_vel, projected_gravity): 9 dims
- Joint positions/velocities: 58 dims (29 + 29)
- Previous actions: 29 dims
- Height above ground: 1 dim
- Vertical velocity: 1 dim
- Contact state (left/right): 2 dims
- Air time per foot: 2 dims
- Target height command: 1 dim

**Critic** (privileged, no noise):
- All policy observations + foot heights + contact forces

### Termination Conditions
- `time_out`: Episode exceeds 3-5s (progressive)
- `fell_over`: Orientation > 60° from upright
- `height_too_low`: CoM below 0.35m (crouch height)
- `excessive_impact`: Landing force > 2500N (safety)

## 📚 Citation

If you use this jumping task in your research, please cite:

```bibtex
@software{g1_jump_task_2025,
  title={Vertical Jump Task for Humanoid Reinforcement Learning},
  author={Your Name},
  year={2025},
  url={https://github.com/mujocolab/mjlab}
}
```

## 🤝 Contributing

Improvements welcome! Areas for contribution:
- [ ] Add horizontal jump variants (forward/backward)
- [ ] Multi-jump sequences (consecutive jumping)
- [ ] Obstacle jumping (hurdles)
- [ ] ONNX policy export for deployment
- [ ] Sim-to-real transfer validation

## 📝 License

This task follows the mjlab project license.

---

**Built with ❤️ using mjlab** - Combining Isaac Lab's API with MuJoCo physics
