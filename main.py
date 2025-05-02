from genesis_sim2real.envs.genesis_gym import GenesisGym
from genesis_sim2real.envs.demo_holder import GenesisDemoHolder
from genesis_sim2real.envs.genesis_gym import DEFAULT_FRICTION, DEFAULT_HEIGHT, DEFAULT_RADIUS, DEFAULT_RHO, DEFAULT_STARTING_X
import numpy as np
import cv2
import os
import pathlib as pl
import matplotlib.pyplot as plt
import torch
import gymnasium

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Genesis Gym Environment')
    parser.add_argument('--vis', action='store_true', help='Enable visualization')
    parser.add_argument('--radius', type=float, default=DEFAULT_RADIUS, help='Bottle radius')
    parser.add_argument('-e', '--height', type=float, default=DEFAULT_HEIGHT, help='Bottle height')
    parser.add_argument('-o', '--rho', type=float, default=DEFAULT_RHO, help='Density of the bottle')
    parser.add_argument('--friction', type=float, default=DEFAULT_FRICTION, help='Friction of the bottle')
    parser.add_argument('--starting_x', type=float, default=DEFAULT_STARTING_X, help='Starting x position of the bottle')
    parser.add_argument('--max-demos', type=int, default=1e7, help='Max number of demos to load')
    parser.add_argument('--random-agent', action='store_true', help='Use a random agent')
    parser.add_argument('--subsample', type=int, default=2, help='Subsample ratio for the demos')
    parser.add_argument('--env-name', type=str, default='lift', help='Environment name')
    args = parser.parse_args()

    use_eef = False



    env = GenesisGym(**args.__dict__)
    obs = env.reset()

    done = False
    max_reward = float('-inf'); reward = 0
    trials = 1; successful_trials = 0; steps = 0; pickups = 0


    from collections import defaultdict
    demonstrations = defaultdict(lambda: {'image': [], 'state': [], 'action': [], 'reward': [], 'next_state': [], 'next_image': [], 'done': []})

    demo_player = GenesisDemoHolder(max_demos=args.max_demos, use_eef=use_eef, subsample_ratio=args.subsample)
    def get_action():
        if args.random_agent:
            return GenesisGym.action_space.sample()
        else:
            action = demo_player.next_action(normalize=False)
            ret = action['action'] if action is not None else None

            if ret is not None and np.isnan(ret).any():
                print(f"!!NaN action!! {ret=} at index {demo_player.action_idx-1}")

            return ret
    trial_id = demo_player.get_trial_id()

    TRIAL_CAN_ADJUSTED = defaultdict(lambda: False)

    # diff_eef_demo = demo_player.convert_eef_to_diff_eef(); action_idx = 0
    video_frames = []
    while True:
        # action = env.action_space.sample()  # Sample random action
        action = get_action()
        # action = diff_eef_demo[action_idx]
        # action_idx += 1

        if action is None or steps > env._max_episode_steps() or done:
        # if action_idx >= len(diff_eef_demo) or done or steps > env._max_episode_steps():
            bottleZ = env.bottle.get_pos().cpu().numpy()[2]
            print(f"\t Max Reward {max_reward:+1.2f}. {bottleZ=}")
            max_reward = float('-inf')

            # close off the last demo
            demonstrations[trial_id]['done'][-1] = True

            trial_id = demo_player.next_demo()

            # reset the env
            if reward > 0: successful_trials += 1
            if bottleZ > 0.15: pickups += 1
            if trial_id == -1:
                print("No more demos")
                break
            # diff_eef_demo = demo_player.convert_eef_to_diff_eef(); action_idx = 0
            trials += 1; steps = 0; done = False

            # write out the video if it was successful:
            if reward > 0:
                # make the new directory if it doesn't exist
                pl.Path(f'./videos').mkdir(parents=True, exist_ok=True)
                video_frames = np.array(video_frames)
                video_path = f'./videos/{trial_id}_video.mp4'
                out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (video_frames.shape[2], video_frames.shape[1]))
                for frame in video_frames:
                    out.write(frame)
                out.release()
                print(f"Video saved to {video_path}")

            video_frames= []

            # write out a histogram of the dp
            # plt.hist(env.dp, bins=50, range=(0, 0.2), alpha=0.5)
            # plt.title(f"Demo {trial_id} DP Histogram")
            # plt.xlabel('DP')
            # plt.ylabel('Frequency')
            # plt.savefig(f'results/{trial_id}_ss{SUBSAMPLE_RATIO}_dp_histogram.png')

            env.reset(trial_id=trial_id)
        else:
            steps += 1
            # print(action)
            next_obs, reward, done, *_ = env.step(action)
            if args.vis: env.render(use_imshow=True)
            if reward > max_reward:
                max_reward = reward

            video_frames.append(obs['image'])

            # if the gripper action is closing and the can is nearby, move the can and restart the demo
            gripper_pos = env.kinova.get_link('end_effector_link').get_pos().cpu().numpy()
            can_pose = env.bottle.get_pos().cpu().numpy()
            dp = np.linalg.norm(gripper_pos - can_pose)
            if action[-1] > 0.5 and dp < 0.2 and not TRIAL_CAN_ADJUSTED[trial_id]:
                # get the average pos of the last 4 links 
                grip_pos = env.get_grip_pose()
                env.reset(trial_id=trial_id)
                demo_player.reset_current_demo()
                env.set_can_to_pose(torch.Tensor(grip_pos))
                print("Gripper closing and can is nearby, restarting demo and setting can to gripper pose")
                TRIAL_CAN_ADJUSTED[trial_id] = True
                

            demonstrations[trial_id]['image'].append(obs['image'])
            demonstrations[trial_id]['state'].append(obs['state'])
            demonstrations[trial_id]['action'].append(action)
            demonstrations[trial_id]['reward'].append(reward)
            demonstrations[trial_id]['next_state'].append(next_obs['state'])
            demonstrations[trial_id]['next_image'].append(next_obs['image'])
            demonstrations[trial_id]['done'].append(done)
            obs = next_obs
            
            # if reward > -0.10:
            #     print(f"Reward: {reward}")

    print(f"Trials: {trials} Successful Trials: {successful_trials} Success Rate: {successful_trials/trials:.2%}")
    print(f"Pickups: {pickups} Pickup Rate: {pickups/trials:.2%}")

    # Save the demonstrations to a file
    for trial_id, demo in demonstrations.items():
        # save the demo out
        output_path = pl.Path(f'./inthewild_trials_{"eef_" if use_eef else ""}SB3/{trial_id}_episodes.npy')
        # make the new directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, demo)

    # Append the results to a file. Create it if it doesn't exist.
    pl.Path('results').mkdir(parents=True, exist_ok=True)
    with open('results/results.txt', 'a') as f:
        f.write(f"subsample ratio {args.subsample} -- {'EEF' if use_eef else ''} {successful_trials/trials:.2%}\n")
        f.write(f"Trials: {trials} Successful Trials: {successful_trials} Success Rate: {successful_trials/trials:.2%}\n")
        f.write(f"Pickups: {pickups} Pickup Rate: {pickups/trials:.2%}\n")
        f.write(f"Max Reward: {max_reward}\n")
        f.write("================================================\n")