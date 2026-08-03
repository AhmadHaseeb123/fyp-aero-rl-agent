from stable_baselines3 import PPO
from envs.toy_grid_env import ToyGridEnv

env = ToyGridEnv(grid_size=5)
model = PPO.load("trained_ppo_agent_model")

print("=" * 60)
print("🤖 TESTING IMPROVED PPO AGENT IN GRID-WORLD 🤖")
print("=" * 60)

obs, _ = env.reset()
total_reward = 0
done = False
step_count = 0

while not done and step_count < 30:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Denormalize position for clean print
    pos_x = int(round(obs[0] * 4))
    pos_y = int(round(obs[1] * 4))
    
    total_reward += reward
    step_count += 1
    done = terminated or truncated
    
    print(f"Step {step_count:02d} | Action: {action} | New Pos: [{pos_x}, {pos_y}] | Step Reward: {reward:.2f}")

print("=" * 60)
print(f"🎯 Total Cumulative Reward Achieved: {total_reward:.2f}")
print("=" * 60)