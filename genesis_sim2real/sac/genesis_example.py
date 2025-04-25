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
parser.add_argument('-ts', '--total-timesteps', type=int, default=200000, help='Total timesteps')
parser.add_argument('-d', '--use-demos', action='store_true', help='Use demos')
parser.add_argument('--use-wandb', action='store_true', help='Use wandb')
parser.add_argument('-v', '--vis', action='store_true', help='Visualize')
args = parser.parse_args()

config = {
    "policy_type": "MultiInputPolicy",
    "total_timesteps": args.total_timesteps,
    "env_name": args.env_name,
}

task_name = config['env_name'].split('_')[1]

print(f"Config: {config}")

if args.use_wandb:
    import wandb
    run = wandb.init(
        project="sb3",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
    )

timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")

env = gym.make(config['env_name'], render_mode="human")
model = SAC(config['policy_type'] , env, verbose=1, buffer_size=500000, tensorboard_log=f"runs/{timestamp}")

if args.use_demos:
    demo_path = pl.Path("~/workspace/genesis_sim2real/inthewild_trials_eef_SB3/").expanduser()
    print(f"Demo path: {demo_path}, exists: {demo_path.exists()}, is_dir: {demo_path.is_dir()}")
    # print the files in the demo path
    for file in demo_path.iterdir():
        print(f"Loading {file}")
        data = np.load(file, allow_pickle=True).flatten()[0]

        states = data['state']
        actions = data['action']
        rewards = data['reward']
        next_states = data['next_state']
        dones = data['done']
        images = data['image']
        next_images = data['next_image']

        for state, action, reward, next_state, done, image, next_image in zip(states, actions, rewards, next_states, dones, images, next_images):
            state_dict = {'image': image.reshape(3, 96, 96), 'state': state}
            next_state_dict = {'image': next_image.reshape(3, 96, 96), 'state': next_state}

            # a = [actions[0], actions[1], actions[2], actions[4], actions[6]]
            a = action

            try:
                model.replay_buffer.add(state_dict, next_state_dict, np.array(a), np.array([reward], dtype=np.float32), np.array([done]), infos=[{}])
            except Exception as e:
                # import the stack trace
                import traceback
                traceback.print_exc()
                print(f"Error adding to replay buffer: {e}")
                print(f"State: {state_dict}")
                print(f"Action: {action}")
                print(f"Reward: {reward}")
                print(f"Next state: {next_state_dict}")
                print(f"Done: {done}")
                print(f"Image shape: {image.shape}")
                print(f"Next image shape: {next_image.shape}")

                import IPython
                IPython.embed()

if args.use_wandb:
    from wandb.integration.sb3 import WandbCallback
    model.learn(total_timesteps=config['total_timesteps'], log_interval=4, callback=WandbCallback())
else:
    model.learn(total_timesteps=config['total_timesteps'], log_interval=4)

model.save(f"genesis_{args.env_name}")

del model # remove to demonstrate saving and loading


model = SAC.load(f"genesis_{args.env_name}")   

obs, info = env.reset()
total_reward = 0
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if args.vis:
        img = obs['image']
        cv2.imshow("img", img)
        cv2.waitKey(1)

    if terminated or truncated:
        print(f"Total reward: {total_reward}")
        obs, info = env.reset()
        total_reward = 0

    