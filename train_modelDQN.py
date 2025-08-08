import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
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
from discrete_gym import GenesisGym

gripper_open_signal = 100
gripper_close_signal = -100

# # Set up genesis environment
# simulation = GenesisGym()
# simulation.init_env()
# #simulation.reset()
# simulation.get_obs()

# default_pose = [0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0]

# print("Goint to pose:", default_pose)
# simulation.apply_target_action(default_pose)
# # simulation.kinova.control_dofs_position(arm_pos, dofs_idx_local=simulation.kdofs_idx[:len(arm_pos)])
# action = []
# # Step the scene
# for _ in range(100):
#     simulation.scene.step()

# # reset scene first
# print("Poke the clay")
# simulation.apply_action([0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0])
# for _ in range(100):
#     simulation.scene.step()

# simulation.apply_action([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0])
# for _ in range(100):
#     simulation.scene.step()

# print("getting observation")
# simulation.get_obs()

# print("Resetting the simulation")
# simulation.reset()
# #simulation.get_obs()


############ Create a RL model ###############

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

# create an instance of the genesis environment
env = GenesisGym()
env.init_env()
check_env(env)
#env.reset()
#env.set_viewer(True)

model = DQN("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=1000, progress_bar=True)
model.save("DQN_model")
print("done learning")
model = DQN.load("DQN_model", env=env)

print("evaluating policy")
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=2)

# Visualize the trained agent
vec_env = model.get_env()
# vec_env.set_viewer(True)
obs = vec_env.reset()

for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")