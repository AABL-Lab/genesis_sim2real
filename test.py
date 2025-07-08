import os
import numpy as np
import torch
import pathlib as pl
import cv2
import argparse
import matplotlib.pyplot as plt

from genesis_sim2real.envs.genesis_gym import GenesisGym
from genesis_sim2real.envs.genesis_gym import (
    DEFAULT_FRICTION, DEFAULT_HEIGHT, DEFAULT_RADIUS, DEFAULT_RHO,
    DEFAULT_STARTING_X
)

os.environ['PYOPENGL_PLATFORM'] = 'osmesa'  # For headless rendering

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Minimal Genesis Discrete Control Script')
    parser.add_argument('--vis', action='store_true', help='Enable visualization')
    args = parser.parse_args()

    # Create and initialize the environment
    env = GenesisGym(
        friction=DEFAULT_FRICTION,
        height=DEFAULT_HEIGHT,
        radius=DEFAULT_RADIUS,
        rho=DEFAULT_RHO,
        starting_x=DEFAULT_STARTING_X,
        vis=args.vis
    )

    obs = env.reset()

    # Define discrete actions (joint + gripper states)
    # These are full joint arrays: 10 DOFs for Kinova Lite + Gripper
    # Adjust these values as needed for your use case
    discrete_actions = [
        np.array([0.3, -1.4, 2.3, -1.3, 2.2, -1.5, 0.0, 0.0, 1.0, 1.0]),  # Open gripper
        np.array([0.3, -1.4, 2.3, -1.3, 2.2, -1.5, 0.0, 0.0, 0.5, 0.5]),  # Half-close
        np.array([0.3, -1.4, 2.3, -1.3, 2.2, -1.5, 0.0, 0.0, 0.0, 0.0]),  # Fully closed
    ]

    video_frames = []

    # Run through the discrete actions
    for idx, action in enumerate(discrete_actions):
        print(f"\n>>> Step {idx+1}: Applying action: {action}")
        obs, reward, done, *_ = env.step(action)

        # Render image if requested
        if args.vis:
            env.render(use_imshow=True)
            video_frames.append(obs['image'])

    # Save video if rendering was enabled
    if args.vis and len(video_frames) > 0:
        video_frames = np.array(video_frames)
        video_path = './discrete_demo_video.mp4'
        pl.Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (video_frames.shape[2], video_frames.shape[1]))

        for frame in video_frames:
            out.write(frame)
        out.release()
        print(f"\n🎥 Saved demo video to: {video_path}")
