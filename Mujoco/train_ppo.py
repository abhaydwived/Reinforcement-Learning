import gymnasium as gym
from stable_baselines3 import PPO
from biped_env import BipedMuJoCoEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import os

# Create environment
env = BipedMuJoCoEnv()

# Define save directories
model_dir = "models/PPO"
log_dir = "logs/PPO"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Define model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

# Optional: Save a checkpoint every 50k steps
checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=model_dir, name_prefix="ppo_biped")

# Train
model.learn(total_timesteps=100000, callback=checkpoint_callback)

# Save final model
model.save(f"{model_dir}/ppo_biped_final")
