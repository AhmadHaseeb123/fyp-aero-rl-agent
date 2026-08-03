import os
from stable_baselines3 import PPO
from envs.toy_grid_env import ToyGridEnv

print("=" * 70)
print("🚀 TASK 4: TRAINING PROXIMAL POLICY OPTIMIZATION (PPO) AGENT 🚀")
print("=" * 70)

# 1. Custom Gymnasium Environment Instantiate Karein
env = ToyGridEnv(grid_size=5)

# 2. PPO Model Setup
# MlpPolicy: Multi-Layer Perceptron (Neural Network for continuous/discrete states)
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=0.001,
    n_steps=128,
    batch_size=64
)

# 3. Model Training (20,000 timesteps)
print("\n[INFO] Agent training started for 20,000 timesteps. Please wait...\n")
model.learn(total_timesteps=50000)

# 4. Save the Trained Model
# Model ka naam bhi script ke naam se milta julta rakh diya hai
model_path = "trained_ppo_agent_model"
model.save(model_path)

print("\n" + "=" * 70)
print(f"✅ TRAINING COMPLETE! Model saved successfully as '{model_path}.zip'")
print("=" * 70)