# import dependencies
import os
import gym
import shimmy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Manual Interaction with the Environment
manual_env = gym.make('CartPole-v1', render_mode="human")

episodes = 5
for episode in range(1, episodes + 1):
    state, _ = manual_env.reset()
    terminated = False
    truncated = False
    score = 0

    while not (terminated or truncated):
        manual_env.render()
        action = manual_env.action_space.sample()
        n_state, reward, terminated, truncated, info = manual_env.step(action)
        score += reward

    print(f"Episode: {episode} Score: {score}")
manual_env.close()

# Training Setup
log_path = os.path.join("Training", "logs")
print("Log path:", log_path)

# NOTE: Use a different instance of env for training (non-rendered)
env = DummyVecEnv([lambda: gym.make('CartPole-v1')])

# Train the model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=10000)
