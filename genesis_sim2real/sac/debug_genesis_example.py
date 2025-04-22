import gymnasium as gym
import genesis_sim2real

from stable_baselines3 import SAC

env = gym.make("genesis_lift-v0", render_mode="human")

model = SAC("MultiInputPolicy" , env, verbose=1)




model.learn(total_timesteps=10000, log_interval=4)
model.save("genesis_lift")

del model # remove to demonstrate saving and loading

model = SAC.load("genesis_lift")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()