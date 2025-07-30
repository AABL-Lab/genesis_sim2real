import numpy as np
import matplotlib.pyplot as plt
import pathlib as pl
import gymnasium as gym
from gymnasium import spaces
import genesis as gs
import torch
from genesis_sim2real.envs.kinova import JOINT_NAMES as kinova_joint_names, EEF_NAME as kinova_eef_name, TRIALS_POSITION_0, TRIALS_POSITION_1, TRIALS_POSITION_2
from genesis_sim2real.envs.actions import KinovaActions
from scipy.spatial.transform import Rotation as R
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
print("Importing genesis gym")
from minimal_example import GenesisGym

env = GenesisGym()
env.init_env()
print("loading model")
model = PPO.load("PPO_model", env=env)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
model.save("Trained model after eval")
# Visualize the trained agent
vec_env = model.get_env()
obs = vec_env.reset()
print("Visualizing the agent")
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")