import gymnasium
from gymnasium import spaces
import numpy as np
import random
import genesis as gs
import pathlib as pl
import cv2
import torch
from kinova import JOINT_NAMES as kinova_joint_names, EEF_NAME as kinova_eef_name, TRIALS_POSITION_0, TRIALS_POSITION_1, TRIALS_POSITION_2

KINOVA_START_DOFS_POS = [0.3268500269015339, -1.4471734542578538, 2.3453266624159497, -1.3502152158191212, 2.209384006676201, -1.5125125137062945, -1, 1, 0.5 , 0.5]
STATIC_BOTTLE_POSITION = torch.tensor((0.65, -0.225, 0.19))
PX, PZ = 0.465, 0.05
POSITION_0 = torch.tensor((PX, 0.1, PZ))
POSITION_1 = torch.tensor((PX, -0.05, PZ))
POSITION_2 = torch.tensor((PX, -0.2, PZ))


def _normalize_action(action):
    """
    Normalize the action from the action space to the range [-1, 1].
    """
    action_space = GenesisGym.action_space
    action = (action - action_space.low) / (action_space.high - action_space.low)
    return 2 * action - 1

def _unnormalize_action(action, action_space):
    """
    Unnormalize the action from the range [-1, 1] to the action space.
    """
    action = (action + 1) / 2 * (action_space.high - action_space.low) + action_space.low
    return action

class GenesisDemoHolder:
    """
    Class to hold the demo data for the Genesis environment.
    """
    def __init__(self, max_demos=float('inf')):
        self.dir = pl.Path('/home/j/workspace/genesis_sim2real/inthewild_trials/')
        self.paths = self.dir.glob('*episodes.npy')

        self.demos = []
        for idx, path in enumerate(self.paths):
            if idx >= max_demos:
                break
            demo = np.load(path, allow_pickle=True).item()
            arm_pos = np.array(demo['vel_cmd'])
            gripper_pos = np.array([entry[0] for entry in demo['gripper_pos']])

            assert len(arm_pos) == len(gripper_pos), f"Arm pos and gripper pos lengths do not match: {len(arm_pos)} vs {len(gripper_pos)}"

            # add a dimension to the gripper_pos
            gripper_pos = np.expand_dims(gripper_pos, axis=1)

            action = np.concatenate((arm_pos, gripper_pos), axis=1)
            trial_id = str(path).split('_episodes')[0].split('/')[-1]
            self.demos.append((int(trial_id), action))

        self.idx = 0
        self.action_idx = 0
        print(f"Loaded {len(self.demos)} demos from {self.dir}")

        for trial_id, d in self.demos:
            print(trial_id, d.shape)

        self.COMPLETED = False

    def next_demo(self):
        self.idx += 1
        self.action_idx = 0
        if self.idx >= len(self.demos):
            print(f"!!No more demos!!")
            self.COMPLETED = True
            return -1
        trial_id = self.demos[self.idx][0]
        print(f"Demo {trial_id} loaded")
        return trial_id

    def next_action(self, normalize=False):
        if self.action_idx >= len(self.demos[self.idx][1]):
            return None
        
        action = self.demos[self.idx][1][self.action_idx]

        if normalize: # map from action space to [-1, 1]
            # print(f'original action: {" ".join([f"{x:+.2f}" for x in action])}')
            action = _normalize_action(action)
            # print(f"\tnorm action: {' '.join([f'{x:+.2f}' for x in action])}")

        self.action_idx += 1

        return {'action': action}


class GenesisGym(gymnasium.Env):
    """
    Custom Gymnasium environment for the Genesis game.
    """
    
    # make a class wide action space
    # Actions are 7 continuous actions. 6 dof joint angles, 1 gripper position
    action_space = spaces.Box(low=np.array([-3.14, -3.14, -3.14, -3.14, -3.14, -3.14, 0]), high=np.array([3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 100.]), shape=(7,), dtype=np.float32)
    
    def __init__(self, args={}, size=(96, 96), use_truncated_in_return=False):
        super().__init__()
        self.args = {
            'rho': args.rho if hasattr(args, 'rho') else 1000,
            'radius': args.radius if hasattr(args, 'radius') else 0.0325,
            'height': args.height if hasattr(args, 'height') else 0.085,
            'friction': args.friction if hasattr(args, 'friction') else 0.2,
            'vis': args.vis if hasattr(args, 'vis') else False,
            # 'starting_x': args.starting_x if hasattr(args, 'starting_x') else 0.65
            }

        self.size = size
        # Define action and observation space
        # Observations are either an image, a state, or a combination
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(*size, 3), dtype=np.uint8),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(10 + 3,), dtype=np.float32), # joint angles and gripper state as well as can location and differential to goal
            'reward': spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
            'is_first': spaces.Box(low=0, high=1, shape=(), dtype=bool),
            'is_last': spaces.Box(low=0, high=1, shape=(), dtype=bool),
            'is_terminal': spaces.Box(low=0, high=1, shape=(), dtype=bool),
        })

        self.last_arm_dofs = None

        gs.init(backend=gs.gpu, seed=random.randint(0, 2**30), precision="32", logging_level="warning")
        self.metadata = {
            "render_fps": 30
        }

        self.use_truncated_in_return = use_truncated_in_return
        self.force_sparse = True

        self.init_env()


    def _max_episode_steps(self):
        return 2000
        # return 100

    def init_env(self):
        self.kp = kp = 5

        self.scene = scene = gs.Scene(
            show_viewer=self.args['vis'],
        )

        plane = scene.add_entity(
            gs.morphs.Plane(),
        )

        BOTTLE_RADIUS = self.args['radius']
        BOTTLE_HEIGHT = self.args['height']
        BOX_WIDTH, BOX_HEIGHT = 0.75, 0.14

        box_pos = (0.78, -BOX_WIDTH / 4, 0.02)
        box = scene.add_entity(
            material=gs.materials.Rigid(rho=1000,
                                        friction=0.05,),
                                        # coup_friction=0.05,),
            morph=gs.morphs.Box(
                size=(0.43, BOX_WIDTH, BOX_HEIGHT),
                pos=box_pos,
            )
        )

        self.cam_0 = scene.add_camera(
            pos=(1, 0, 1),
            lookat=(0.6, 0, 0.25),
            fov=30,
            GUI=True,
        )

        import pathlib as pl
        # TODO: see if you can prevent the gripper from being convexified
        self.kinova = kinova = scene.add_entity(
            gs.morphs.URDF(
                file=str(pl.Path(__file__).parent / 'gen3_lite_2f_robotiq_85.urdf'),
                fixed=True,
                convexify=True,
                pos=(0.0, 0.0, 0.055), # raise to account for table mount
            ),
            material=gs.materials.Rigid(friction=0.5),
            vis_mode="collision"

            # gs.morphs.MJCF(file="/home/j/workspace/genesis_pickaplace/005_tomato_soup_can/google_512k/kinbody.xml"),
        )


        # TODO: make the bottle slightly deformable
        self.bottle = bottle = scene.add_entity(
            material=gs.materials.Rigid(rho=self.args['rho'],
                                        friction=self.args['friction']),
            # material=gs.materials.Rigid(rho=self.args.rho,
            #                             friction=self.args.friction),
            morph=gs.morphs.Cylinder(
                pos=POSITION_0,
                radius=BOTTLE_RADIUS,
                height=BOTTLE_HEIGHT,
            ),
            # visualize_contact=True,
        )

        self.goal_bottle = goal_bottle = scene.add_entity(
            material=gs.materials.Rigid(rho=self.args['rho'],
                                    friction=2.0),
            morph=gs.morphs.Cylinder(
                pos=STATIC_BOTTLE_POSITION,
                radius=BOTTLE_RADIUS,
                height=BOTTLE_HEIGHT,
            ),
            # visualize_contact=True,
        )


        from kinova import JOINT_NAMES as kinova_joint_names, EEF_NAME as kinova_eef_name, TRIALS_POSITION_0, TRIALS_POSITION_1, TRIALS_POSITION_2
        self.kdofs_idx = kdofs_idx = [kinova.get_joint(name).dof_idx_local for name in kinova_joint_names]
        eef = kinova.get_link(kinova_eef_name)
        print(f"Kinova end effector: {eef}")
        scene.build()

        ############ Optional: set control gains ############
        # set positional gains
        kinova.set_dofs_kp(
            kp             = 3*np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100]),
            dofs_idx_local = kdofs_idx,
        )
        kinova.set_dofs_position(np.array(KINOVA_START_DOFS_POS), kdofs_idx)

    def step(self, action):
        # Apply the action to the scene
        self.apply_action(action)
        self.scene.step()
        obs = self.get_obs()
        reward = obs['reward']
        done = obs['is_last']

        self.n_steps += 1

        if self.use_truncated_in_return:
            return obs, reward, done, self.n_steps >= self._max_episode_steps(), {'is_success': done}
        return obs, reward, done, {}
    
    def reset(self, trial_id=0, **kwargs):
        # Reset the scene and get the initial observation
        self.n_steps = 0

        if trial_id in TRIALS_POSITION_0:
            bottle_pos = POSITION_0
        elif trial_id in TRIALS_POSITION_1:
            bottle_pos = POSITION_1
        elif trial_id in TRIALS_POSITION_2:
            bottle_pos = POSITION_2
        else:
            rand_idx = random.randint(0,2)
            bottle_pos = [POSITION_0, POSITION_1, POSITION_2][rand_idx]
            # random_y = random.uniform(POSITION_2[1], POSITION_0[1])
            # bottle_pos = [PX, random_y, PZ]

        self.bottle.set_pos(bottle_pos); self.bottle.set_quat(torch.Tensor([1, 0, 0, 0]))
        self.goal_bottle.set_pos(STATIC_BOTTLE_POSITION); self.goal_bottle.set_quat(torch.Tensor([1, 0, 0, 0]))
        self.kinova.set_dofs_position(np.array(KINOVA_START_DOFS_POS), self.kdofs_idx)
        self.scene.step()
        obs = self.get_obs()
        if self.use_truncated_in_return:
            ret = obs, {}
        else:
            ret = obs
        return ret
    
    def get_obs(self, is_first=False):
        # Get the current observation from the scene
        image = self.cam_0.render(rgb=True, depth=False, segmentation=False, normal=False, use_imshow=False)
        # from IPython import embed; embed(); exit(0)
        image = image[0] # grab the rgb
        # resize the image to the desired size
        image = cv2.resize(image, self.size)
        # image = None
        # if not image:
        #     image = np.zeros((*self.size, 3), dtype=np.uint8)

        arm_pos = self.kinova.get_dofs_position(dofs_idx_local=self.kdofs_idx).cpu().numpy()
        bottle_pos = self.bottle.get_pos().cpu().numpy()
        state = np.concatenate((arm_pos, bottle_pos))

        self.last_arm_dofs = arm_pos

        reward, done = self.compute_reward(state)
        return {"image": image, "state": state, "reward": reward, "is_first": is_first, "is_last": done, "is_terminal": False}
    
    def calc_gripper_force(self, cmd_gripper_pos):
        # Calculate the gripper force based on the gripper position
        pos = self.last_arm_dofs
        output_force = [0., 0.] #, 0., 0.]
        motor_cmd = (100 - cmd_gripper_pos) / 100
        right_error = pos[-4] + motor_cmd; right_error = right_error if abs(right_error) > 0.05 else [0.0]
        left_error = pos[-3] - motor_cmd; left_error = left_error if abs(left_error) > 0.05 else [0.0]
        #right_fingertip_error = pos[-2] - KINOVA_START_DOFS_POS[-2]; right_fingertip_error = right_fingertip_error if abs(right_fingertip_error) > 0.05 else 0.0
        #left_fingertip_error = pos[-1] - KINOVA_START_DOFS_POS[-1]; left_fingertip_error = left_fingertip_error if abs(left_fingertip_error) > 0.05 else 0.0

        output_force[0] = -self.kp*right_error[0]; # output_force[2] = self.kp*right_fingertip_error
        output_force[1] = -self.kp*left_error[0]; # output_force[3] = self.kp*left_fingertip_error
        # print(output_force)
        return np.array(output_force)

    def apply_action(self, action):

        # map from -1, 1 to the action space
        # print(f'\t gym action: {" ".join([f"{x:+.2f}" for x in action])}')
        # if not self.use_truncated_in_return: # TODO: this should be handled in fastrl
        #     unnorm_action = _unnormalize_action(action, self.action_space)
        #     print(f"\t unnorm action: {' '.join([f'{x:+.2f}' for x in unnorm_action])}")
        #     action = unnorm_action

        arm_pos, gripper_pos = action[:6], action[6:]

        gripper_force = self.calc_gripper_force(gripper_pos)

        # gripper_force[2] = 2.0
        # gripper_force[3] = 2.0
        # print(f"Gripper force: {' '.join([f'{x:.2f}' for x in gripper_force])}")

        self.kinova.control_dofs_force(gripper_force, dofs_idx_local=np.array(self.kdofs_idx[-4:-2]))
        self.kinova.control_dofs_position(arm_pos, dofs_idx_local=self.kdofs_idx[:len(arm_pos)])

    def compute_reward(self, obs):
        bottle_pos = self.bottle.get_pos()
        goal_pos = self.goal_bottle.get_pos()
        distance = torch.linalg.norm(bottle_pos - goal_pos, ord=2, dim=-1, keepdim=True)

        reward = -distance.item() # TODO: implement reward function
        done = reward > -0.1 and (bottle_pos[2].cpu().numpy().item() >= (STATIC_BOTTLE_POSITION[2] - 0.09)) and (goal_pos[2].cpu().numpy().item() >= (STATIC_BOTTLE_POSITION[2] - 0.09))
        if done: 
            print(f"SUCCESS!")
            reward = 1.0
        elif self.force_sparse:
            reward = 0.0
        # else:
        #     print(f"\treward: {reward:.2f} distance: {distance.item():.2f}")

        return reward, done # TODO: implement reward function

    def render(self, mode='human', use_imshow=False):
        # Render the scene
        # if self.args.vis:
        img = None
        if mode == 'human':
            img = self.cam_0.render(rgb=True, depth=False, segmentation=False, normal=False, use_imshow=False)[0]
            img = cv2.resize(img, self.size)
            if use_imshow:
                cv2.imshow('Genesis Gym', img)
                cv2.waitKey(1)
        return img
    
    def playback_demo(self, demo, num_steps=10):
        # Play back a demo
        for i in range(num_steps):
            action = demo['actions'][i]
            self.apply_action(action)
            self.scene.step()
            obs = self.get_obs(is_first=(i == 0))
            self.render()
        return obs

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Genesis Gym Environment')
    parser.add_argument('--vis', action='store_true', help='Enable visualization')
    parser.add_argument('--radius', type=float, default=0.035, help='Bottle radius')
    parser.add_argument('-e', '--height', type=float, default=0.085, help='Bottle height')
    parser.add_argument('-o', '--rho', type=float, default=1000, help='Density of the bottle')
    parser.add_argument('--friction', type=float, default=0.5, help='Friction of the bottle')
    parser.add_argument('--starting_x', type=float, default=0.65, help='Starting x position of the bottle')
    parser.add_argument('--max-demos', type=int, default=1e7, help='Max number of demos to load')
    args = parser.parse_args()
    

    normalize = True
    demo_player = GenesisDemoHolder(max_demos=args.max_demos)

    # print(GenesisGym.action_space, GenesisGym.action_space.low, GenesisGym.action_space.high)
    # ### Action normalization / unnormalization
    # while action := demo_player.next_action(normalize=False):
    #     original_action = action['action']
    #     normalized_action = _normalize_action(original_action, GenesisGym.action_space)
    #     unnormalized_action = _unnormalize_action(normalized_action, GenesisGym.action_space)
    #     print(f"orig, norm, unnorm: {' || '.join([f'{a:+.2f} {x:+.2f} {y:+.2f}' for a,x,y in zip(original_action, normalized_action, unnormalized_action)])}")

    # exit()


    env = GenesisGym(args)
    obs = env.reset()
    done = False
    max_reward = float('-inf')
    trials = 1; successful_trials = 0
    while True:
        # action = env.action_space.sample()  # Sample random action
        action = demo_player.next_action(normalize=False)
        if action is None:
            print(f"\t Max Reward {max_reward:+1.2f}")
            max_reward = float('-inf')
            trial_id = demo_player.next_demo()
            if done: successful_trials += 1
            if trial_id == -1:
                print("No more demos")
                break
            trials += 1
            env.reset(trial_id=trial_id)
        else:
            # print(action)
            obs, reward, done, *_ = env.step(action['action'])
            # env.render()
            if reward > max_reward:
                max_reward = reward
            # if reward > -0.10:
            #     print(f"Reward: {reward}")

    print(f"Trials: {trials} Successful Trials: {successful_trials} Success Rate: {successful_trials/trials:.2%}")