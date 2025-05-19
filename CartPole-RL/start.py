# import dependencies
import os
import gymnasium as gym 
# import shimmy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


# Setting up Environment
env = gym.make('CartPole-v1', render_mode="human")

episodes = 5
for episode in range(1,episodes+1):
    state,_ = env.reset() # initial obervations for env to take action
    terminated = False
    truncated = False
    score = 0
   
    while not (terminated or truncated):
        env.render()
        action = env.action_space.sample() # list of actions and it will return randomly one of them
        n_state, reward, terminated, truncated, info = env.step(action)
        score += reward
    print(f"Episode: {episode} Score: {score}")
# print(env.step(1))
env.close() 

# env.action_space                    list of actions
# env.action_space.sample()           it will return randomly one of actions
# env.observation_space               environment actions return in box 
# env.observation_space.sample()      return one of them randomly

# Training 
log_path = os.path.join("Training","logs")
print(log_path)
env = DummyVecEnv([lambda: gym.make('CartPole-v1', render_mode='human')])
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
# Using cpu device
model.learn(total_timesteps=20000)

# Save and reload model
PPO_path = os.path.join("Training","Saved Models", "PPO_Model_Cartpole")
model.save(PPO_path)

# del model
# print(PPO_path)
# model = PPO.load(PPO_path, env=env)
# model.learn(total_timesteps=10000)

# Evaluation
evaluate_policy(model, env, n_eval_episodes=10, render=True)
env.close()

# Testing Model
for episode in range(1,episodes+1):
    obs = env.reset() # initial obervations for env to take action
    terminated = False
    truncated = False
    score = 0
   
    while not (terminated or truncated):
        env.render()
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
    print(f"Episode: {episode} Score: {score}")
# print(env.step(1))
env.close() 