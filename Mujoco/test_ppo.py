from stable_baselines3 import PPO
from biped_env import BipedMuJoCoEnv

env = BipedMuJoCoEnv(render_mode="human")
model = PPO.load("models/PPO/ppo_biped_final", env=env)

obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    env.render()
env.close()
 