from genesis_sim2real.envs.genesis_gym import GenesisGym
from genesis_sim2real.envs.demo_holder import GenesisDemoHolder
from genesis_sim2real.envs.genesis_gym import DEFAULT_FRICTION, DEFAULT_HEIGHT, DEFAULT_RADIUS, DEFAULT_RHO, DEFAULT_STARTING_X, STATIC_BOTTLE_POSITION, PZ
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
    parser.add_argument('--noise-scale', type=float, default=0.02, help='Noise scale for setting cans at closed gripper position')
    args = parser.parse_args()

    use_eef = False



    env = GenesisGym(**args.__dict__)
    obs = env.reset()

    done = np.array([False] * env.B, dtype=bool)  # Initialize done for all environments
    max_reward = float('-inf'); reward = 0
    trials = 1; successful_trials = 0; steps = 0; pickups = 0


    from collections import defaultdict
    demonstrations = defaultdict(lambda: {'image': [], 'state': [], 'action': [], 'reward': [], 'next_state': [], 'next_image': [], 'done': []})

    demo_player = GenesisDemoHolder(max_demos=args.max_demos, use_eef=False, subsample_ratio=args.subsample)
    def get_action():
        if args.random_agent:
            return GenesisGym.action_space.sample()
        else:
            action = demo_player.next_action(normalize=False)
            ret = action['action'] if action is not None else None

            if ret is not None and np.isnan(ret).any():
                print(f"!!NaN action!! {ret=} at index {demo_player.action_idx-1}")

            return ret
    trial_id = demo_player.get_trial_id(); demo_resets = 0
    TRIAL_CAN_ADJUSTED = defaultdict(lambda: False)
    ADJUSTED_CAN_POS = {}
    TRIAL_SUCCESS_RATES = defaultdict(lambda: 0.0)

    # diff_eef_demo = demo_player.convert_eef_to_diff_eef(); action_idx = 0
    video_frames = []
    while trials < len(demo_player.demos):
        # action = env.action_space.sample()  # Sample random action
        action = get_action()
        # action = diff_eef_demo[action_idx]
        # action_idx += 1

        # NOTE: Stopping under these conditions will let episodes go from done to not done, but it's still good to get the success rate at the end of the episode
        if action is None or steps > env._max_episode_steps() or done.all():
        # if action_idx >= len(diff_eef_demo) or done or steps > env._max_episode_steps():
            bottleZ = env.bottle.get_pos().cpu().numpy()[2]
            print(f"\t Max Reward {max_reward:+1.2f}. {bottleZ=}")
            max_reward = float('-inf')

            # close off the last demo
            # demonstrations[trial_id]['done'][-1] = True

            if False and reward < 9.99 and demo_resets < 5:
                print(f"Reset demo {trial_id} due to low reward {reward}")
                demo_player.reset_current_demo()
                demo_resets += 1
            else:
                demo_resets = 0
                trial_id = demo_player.next_demo()

                # reset the env
                # if reward > 0: successful_trials += 1
                # if bottleZ > 0.15: pickups += 1
                # if trial_id == -1:
                #     print("No more demos")
                #     break

                successful_trials += sum(done)
                
                print(f"Trial {trial_id} done. Successful trials: {sum(done).item()} of {len(done)}. {sum(done).item()/len(done):.2%} success rate")
                TRIAL_SUCCESS_RATES[trial_id] += sum(done).item() / len(done)
                # diff_eef_demo = demo_player.convert_eef_to_diff_eef(); action_idx = 0
                trials += 1

                # write out the video if it was successful:
            #     if reward > 0:
            #         # make the new directory if it doesn't exist
            #         vid_dir = f'./videos_ss{args.subsample}'
            #         pl.Path(vid_dir).mkdir(parents=True, exist_ok=True)
            #         video_frames = np.array(video_frames)
            #         video_path = f'{vid_dir}/{trial_id}_video.mp4'
            #         out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (video_frames.shape[2], video_frames.shape[1]))
            #         for frame in video_frames:
            #             out.write(frame)
            #         out.release()
            #         print(f"Video saved to {video_path}")

            video_frames= []; steps = 0;     
            done = np.array([False] * env.B, dtype=bool)  # Initialize done for all environments


            env.reset(trial_id=trial_id)
        else:
            steps += 1
            # print(action)
            next_obs, reward, done, *_ = env.step(action)

            # if args.vis: env.render(use_imshow=True)
            if True:
### FOR NOW JUST PRINT OUT SUCCESS STATS ###
                # if reward > max_reward:
                #     max_reward = reward

                # video_frames.append(obs['image'])

                try:
                    # if the gripper action is closing and the can is nearby, move the can and restart the demo
                    gripper_pos = env.kinova.get_link('end_effector_link').get_pos().cpu().numpy()
                    left_fingertip = env.kinova.get_link('left_finger_dist_link')
                    right_fingertip = env.kinova.get_link('right_finger_dist_link')
                    can_pose = env.bottle.get_pos().cpu().numpy()
                    dp_left = np.linalg.norm(gripper_pos - left_fingertip.get_pos().cpu().numpy(), axis=1)
                    dp_right = np.linalg.norm(gripper_pos - right_fingertip.get_pos().cpu().numpy(), axis=1)
                    if action[-1] > 50 and np.mean(dp_left) < 0.9 and np.mean(dp_right) < 0.9 and not TRIAL_CAN_ADJUSTED[trial_id] and np.mean(gripper_pos[...,2]) < 0.1:
                        # get the average pos of the last 4 links 
                        grip_pos = env.get_grip_pose()
                        grip_pos[..., -1] = PZ
                        # make a debug sphere
                        # debug_arrow = env.scene.draw_debug_arrow(pos=gripper_pos, vec=grip_pos - gripper_pos, radius=0.01, color=(1, 0, 0, 0.5))  # Green
                        # env.scene.draw_debug_sphere(gripper_pos, 0.01, color=(0, 1, 1))
                        # env.scene.draw_debug_sphere(grip_pos, 0.01, color=(0, 0, 1))
                        env.reset(trial_id=trial_id)
                        demo_player.reset_current_demo()

                        env.step(get_action())

                        for _ in range(30):
                            env.scene.step() # let the arm get back before we reset the can

                        # Add random noise to the grip position
                        noise = np.random.random(grip_pos.shape) * args.noise_scale
                        grip_pos += noise

                        env.set_can_to_pose(torch.Tensor(grip_pos))
                        # print("Gripper closing and can is nearby, restarting demo and setting can to gripper pose")
                        ADJUSTED_CAN_POS[trial_id] = grip_pos
                        TRIAL_CAN_ADJUSTED[trial_id] = True
                except Exception as e:
                    print(f"Error adjusting can position: {e}")

                # demonstrations[trial_id]['image'].append(obs['image'])
                # demonstrations[trial_id]['state'].append(obs['state'])
                # demonstrations[trial_id]['action'].append(action)
                # demonstrations[trial_id]['reward'].append(reward)
                # demonstrations[trial_id]['next_state'].append(next_obs['state'])
                # demonstrations[trial_id]['next_image'].append(next_obs['image'])
                # demonstrations[trial_id]['done'].append(done)
                obs = next_obs
            
    avg_success_rates = sum(TRIAL_SUCCESS_RATES.values()) / len(TRIAL_SUCCESS_RATES) if TRIAL_SUCCESS_RATES else 0.0
    print(f"Average success rate across trials: {avg_success_rates:.2%}")
    for k,v in TRIAL_SUCCESS_RATES.items():
        print(f"Trial {k} success rate: {v:.2%}")

    # make a results directory if it doesn't exist
    pl.Path('./results').mkdir(parents=True, exist_ok=True)
    # write out the arguments and results to a file
    import datetime
    fn = f'./results/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{avg_success_rates:.2f}_results.txt'
    with open(fn, 'w') as f:
        f.write(f"Arguments: {args}\n")
        for k,v in TRIAL_SUCCESS_RATES.items():
            f.write(f"Trial {k} success rate: {v:.2%}\n")
        f.write("================================================\n")
if False:
### FOR NOW JUST PRINT OUT SUCCESS STATS ###
    # save out the ADJUSTED_CAN_POS dictionary to a file
    adjusted_can_pos_path = f'./trial_can_adjusted.npy'
    np.save(adjusted_can_pos_path, ADJUSTED_CAN_POS)

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
    output_dir = f'results'
    pl.Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f'{output_dir}/results.txt', 'a') as f:
        f.write(f"subsample ratio {args.subsample} -- {'EEF' if use_eef else ''} {successful_trials/trials:.2%}\n")
        f.write(f"Trials: {trials} Successful Trials: {successful_trials} Success Rate: {successful_trials/trials:.2%}\n")
        f.write(f"Pickups: {pickups} Pickup Rate: {pickups/trials:.2%}\n")
        f.write(f"Max Reward: {max_reward}\n")
        f.write("================================================\n")
### END FOR NOW JUST PRINT OUT SUCCESS STATS ###

