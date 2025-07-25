import gymnasium
from gymnasium import spaces
from genesis.utils.geom import quat_to_xyz
import numpy as np
import random
import genesis as gs
import pathlib as pl
import cv2
import torch
from genesis_sim2real.envs.kinova import JOINT_NAMES as kinova_joint_names, EEF_NAME as kinova_eef_name, TRIALS_POSITION_0, TRIALS_POSITION_1, TRIALS_POSITION_2
from matplotlib import pyplot as plt

from genesis_gym import GenesisGym

class ParallelGenesisGym(GenesisGym):
    """
    Custom Gymnasium environment for the Genesis game.
    """
    
    # make a class wide action space
    # Actions are 7 continuous actions. 6 dof joint angles, 1 gripper position
    action_space = spaces.Box(low=np.array([-3.14, -3.14, -3.14, -3.14, -3.14, -3.14, 0]), high=np.array([3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 100.]), shape=(7,), dtype=np.float32)
    
    # actions are eef position, orientation, and gripper position
    # action_space = spaces.Box(low=np.array([-1, -1, -1, -3.14, -3.14, -3.14, 0]), high=np.array([1, 1, 1, 3.14, 3.14, 3.14, 100.]), shape=(7,), dtype=np.float32) 
    # action_space = spaces.Box(low=np.array([-1, -1, -1, -3.14, 0]), high=np.array([1, 1, 1, 3.14, 100.]), shape=(5,), dtype=np.float32) 


    def __init__(self, size=(96, 96), use_truncated_in_return=False, debug=False, stable_baselines=False, check_saved_positions=False, **kwargs):
       super().__init__(**kwargs)