import gymnasium as gym
import genesis_sim2real
import cv2
from stable_baselines3 import SAC
import datetime
# Read the demos from the demo path
import numpy as np
import pathlib as pl

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--env-name', type=str, default='genesis_lift_SB-v0', help='Environment name')
parser.add_argument('-v', '--vis', action='store_true', help='Visualize')
args = parser.parse_args()

config = {
    "policy_type": "MultiInputPolicy",
    "env_name": args.env_name,
}

task_name = config['env_name'].split('_')[1]

print(f"Config: {config}")

env = gym.make(config['env_name'], render_mode="human")
model = SAC.load(f"genesis_{task_name}")

obs, info = env.reset()
total_reward = 0
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if args.vis:
        print(f"Reward: {reward:.2f}, Total reward: {total_reward:.2f}")
        try:
            img = obs['image']
            cv2.imshow("img", img)
        except:
            import IPython
            IPython.embed(); exit()
        cv2.waitKey(1)

    if terminated or truncated:
        print(f"Total reward: {total_reward}")
        obs, info = env.reset()
        total_reward = 0

    