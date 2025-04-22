import gymnasium as gym
import genesis_sim2real
import cv2
from stable_baselines3 import SAC
# Read the demos from the demo path
import numpy as np
import pathlib as pl

env = gym.make("genesis_lift_SB-v0", render_mode="human")
model = SAC("MultiInputPolicy" , env, verbose=1, buffer_size=350000)

model = SAC.load("genesis_lift")

obs, info = env.reset()
total_reward = 0
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    # Render the environment
    # if env.render_mode == "human":
    img = obs['image']
    cv2.imshow("genesis", img)
    cv2.waitKey(1)


    if terminated or truncated:
        print(f"Total reward: {total_reward}")
        obs, info = env.reset()
        total_reward = 0

    