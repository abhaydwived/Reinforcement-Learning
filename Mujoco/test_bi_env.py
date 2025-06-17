import gymnasium as gym
from biped_env import BipedMujocoEnv
import numpy as np
import time

# Create environment with human rendering
env = BipedMujocoEnv(render_mode="human")

# Reset the environment
obs, info = env.reset()
done = False
total_reward = 0
step_count = 0

while not done:
    action = env.action_space.sample()  # Take random action
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    total_reward += reward
    step_count += 1

    print(f"Step: {step_count} | Reward: {reward:.2f} | Total: {total_reward:.2f}")
    time.sleep(1 / 60)  # Optional slow-down for visualization

# Close environment
env.close()
print(f"Episode finished in {step_count} steps with total reward {total_reward:.2f}")
