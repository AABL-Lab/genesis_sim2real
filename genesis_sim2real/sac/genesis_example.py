import gymnasium as gym
import genesis_sim2real
import cv2
from stable_baselines3 import SAC
# Read the demos from the demo path
import numpy as np
import pathlib as pl

env = gym.make("genesis_lift_SB-v0", render_mode="human")
model = SAC("MultiInputPolicy" , env, verbose=1, buffer_size=350000)

demo_path = pl.Path("./inthewild_trials_eef_SB3/")
print(f"Demo path: {demo_path}, exists: {demo_path.exists()}, is_dir: {demo_path.is_dir()}")


# print the files in the demo path
for file in demo_path.iterdir():
    print(f"Loading {file}")
    data = np.load(file, allow_pickle=True).flatten()[0]
    # print(data.files)
    # print(data['image'].shape)
    # print(data['state'].shape)
    # print(data['action'].shape)
    # print(data['reward'].shape)
    # print(data['next_state'].shape)
    # print(data['next_image'].shape)
    # print(data['done'].shape)
    

    # for img0, img1 in zip(data['image'], data['next_image']):
    #     bigimg = np.concatenate((img0, img1), axis=1)
    #     # resize
    #     bigimg = cv2.resize(bigimg, (800, 400))
    #     cv2.imshow("bigimg", bigimg)
    #     cv2.waitKey(1)

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

        try:
            model.replay_buffer.add(state_dict, next_state_dict, action, np.array([reward], dtype=np.float32), np.array([done]), infos=[{}])
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

            from IPython import embed as ipshell; ipshell()


model.learn(total_timesteps=100000, log_interval=4)
model.save("genesis_lift")

del model # remove to demonstrate saving and loading

model = SAC.load("genesis_lift")

obs, info = env.reset()
total_reward = 0
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        print(f"Total reward: {total_reward}")
        obs, info = env.reset()
        total_reward = 0

    