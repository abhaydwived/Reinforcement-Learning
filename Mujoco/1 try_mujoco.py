import gymnasium as gym

env = gym.make("Humanoid-v5", render_mode="human")
obs, _ = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
    