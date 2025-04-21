import gymnasium
from gymnasium import spaces
from genesis.utils.geom import quat_to_xyz
import numpy as np
import random
import genesis as gs
import pathlib as pl
import cv2
import torch
from .kinova import JOINT_NAMES as kinova_joint_names, EEF_NAME as kinova_eef_name, TRIALS_POSITION_0, TRIALS_POSITION_1, TRIALS_POSITION_2
from matplotlib import pyplot as plt

FINGERTIP_POS = -0.9
KINOVA_START_DOFS_POS = [0.3268500269015339, -1.4471734542578538, 2.3453266624159497, -1.3502152158191212, 2.209384006676201, -1.5125125137062945, -1, 1, FINGERTIP_POS, FINGERTIP_POS]
STATIC_BOTTLE_POSITION = torch.tensor((0.65, -0.225, 0.17))
PX, PZ = 0.465, 0.05
POSITION_0 = torch.tensor((PX, 0.1, PZ))
POSITION_1 = torch.tensor((PX, -0.05, PZ))
POSITION_2 = torch.tensor((PX, -0.2, PZ))

## Default Args
DEFAULT_RADIUS = 0.034
DEFAULT_HEIGHT = 0.09
DEFAULT_RHO = 2000
DEFAULT_FRICTION = 0.5
DEFAULT_STARTING_X = 0.65

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
    def __init__(self, max_demos=float('inf'), use_eef=False, subsample_ratio=1):

        self.dir = pl.Path('/home/j/workspace/genesis_sim2real/inthewild_trials_eef/') if use_eef else pl.Path('/home/j/workspace/genesis_sim2real/inthewild_trials/')
        self.paths = self.dir.glob('*episodes.npy')
        self.subsample_ratio = subsample_ratio

        self.demos = []
        for idx, path in enumerate(self.paths):
            if idx >= max_demos:
                break

            # old 
            if not use_eef:
                demo = np.load(path, allow_pickle=True).item()
                arm_pos = np.array(demo['vel_cmd'])
                gripper_pos = np.array([entry[0] for entry in demo['gripper_pos']])
                assert len(arm_pos) == len(gripper_pos), f"Arm pos and gripper pos lengths do not match: {len(arm_pos)} vs {len(gripper_pos)}"

                # add a dimension to the gripper_pos
                gripper_pos = np.expand_dims(gripper_pos, axis=1)

                action = np.concatenate((arm_pos, gripper_pos), axis=1)
            else: # use eef
                # new
                demo = np.load(path, allow_pickle=True)
                # arm_pos = demo[:, :6]
                # gripper_pos = demo[:, 6]
                action = demo

            trial_id = str(path).split('_episodes')[0].split('/')[-1]
            self.demos.append((int(trial_id), action))

        self.NEXT_DEMO_CALLED = False # We've already loaded the first demo, so do nothing the first time

        ## postprocess
        for idx, (trial_id, d) in enumerate(self.demos):
            if self.subsample_ratio > 1:
                # subsamples the n, d sequence to 1/10th of the original. Averages the action over the 10 samples.
                subsampled_d = []
                for i in range(1, len(d), self.subsample_ratio):
                    subsampled_d.append(np.mean(d[i-self.subsample_ratio:i], axis=0))
                # fill in the first action
                subsampled_d[0] = d[0]

                # check for nans
                if np.isnan(subsampled_d).any():
                    print(f"!!NaN in demo {trial_id}!!")
                    continue

                self.demos[idx] = (trial_id, np.array(subsampled_d))

        self.idx = 0
        self.action_idx = 0
        print(f"Loaded {len(self.demos)} demos from {self.dir}")

        total_samples = 0
        for trial_id, d in self.demos:
            print(trial_id, d.shape, end=' -- ')
            total_samples += d.shape[0]
        print()
        print(f"Total samples: {total_samples}")

        self.COMPLETED = False

    def get_trial_id(self):
        """
        Get the trial ID of the current demo.
        """
        return self.demos[self.idx][0]

    def next_demo(self):
        if not self.NEXT_DEMO_CALLED: 
            self.NEXT_DEMO_CALLED = True
            return self.demos[self.idx][0]
        self.idx += 1
        self.action_idx = 0
        if self.idx >= len(self.demos):
            print(f"!!No more demos!!")
            self.COMPLETED = True
            return -1
        trial_id = self.demos[self.idx][0]
        print(f"Demo {trial_id} loaded")
        return trial_id
    
    def reset_current_demo(self):
        self.action_idx = 0
        print(f"Reset current demo {self.demos[self.idx][0]}")

    def next_action(self, normalize=False):
        if not self.NEXT_DEMO_CALLED: self.NEXT_DEMO_CALLED = True # If we take an action, next demo should move us forward
        if self.action_idx >= len(self.demos[self.idx][1]):
            return None
        
        action = self.demos[self.idx][1][self.action_idx]

        if normalize: # map from action space to [-1, 1]
            # print(f'original action: {" ".join([f"{x:+.2f}" for x in action])}')
            action = _normalize_action(action)
            # print(f"\tnorm action: {' '.join([f'{x:+.2f}' for x in action])}")

        self.action_idx += 1

        return {'action': action}
    
    def convert_joint_to_eef(self, genesis_arm, output_dir='./inthewild_trials_eef'):
        assert '_eef' not in output_dir, f"Output directory {output_dir} already contains _eef"
        output_dir = pl.Path(output_dir)
        for idx, (trial_id, d) in enumerate(self.demos):
            # if trial_id != 235: continue
            new_d = []
            for i in range(d.shape[0]): # for each joint position action [j0, j1, j2, j3, j4, j5, gripper]
                joint_pos_theta = d[i, :6]
                joint_pos, joint_quat = genesis_arm.forward_kinematics(torch.tensor(joint_pos_theta))
                eef_pos = joint_pos[6].cpu().numpy(); eef_quat = joint_quat[6].cpu().numpy()
                eef_euler = gs.utils.geom.quat_to_xyz(eef_quat)
                
                action = np.concatenate((eef_pos, eef_euler, d[i, 6:]), axis=-1)
                # print(', '.join([f'{x:+.1f}' for x in action]), '||', ', '.join([f'{x:+.1f}' for x in d[i, :]]))
                new_d.append(action)

            # save the demo out
            output_path = output_dir / f'{trial_id}_episodes.npy'
            # make the new directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, new_d)
            

class GenesisGym(gymnasium.Env):
    """
    Custom Gymnasium environment for the Genesis game.
    """
    
    # make a class wide action space
    # Actions are 7 continuous actions. 6 dof joint angles, 1 gripper position
    # action_space = spaces.Box(low=np.array([-3.14, -3.14, -3.14, -3.14, -3.14, -3.14, 0]), high=np.array([3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 100.]), shape=(7,), dtype=np.float32)
    
    # actions are eef position, orientation, and gripper position
    action_space = spaces.Box(low=np.array([-1, -1, -1, -3.14, -3.14, -3.14, 0]), high=np.array([1, 1, 1, 3.14, 3.14, 3.14, 100.]), shape=(7,), dtype=np.float32) 

    def __init__(self, size=(96, 96), use_truncated_in_return=False, debug=False, stable_baselines=False, **kwargs):
        super().__init__()
        self.args = {
            'rho': kwargs.rho if hasattr(kwargs, 'rho') else DEFAULT_RHO,
            'radius': kwargs.radius if hasattr(kwargs, 'radius') else DEFAULT_RADIUS,
            'height': kwargs.height if hasattr(kwargs, 'height') else DEFAULT_HEIGHT,
            'friction': kwargs.friction if hasattr(kwargs, 'friction') else DEFAULT_FRICTION,
            'vis': kwargs.vis if hasattr(kwargs, 'vis') else False,
            'grayscale': kwargs.grayscale if hasattr(kwargs, 'grayscale') else False,
            'time_limit': kwargs.time_limit if hasattr(kwargs, 'time_limit') else 800,
            # 'starting_x': args.starting_x if hasattr(args, 'starting_x') else 0.65
            }

        print(f"GenesisGym args: {self.args}")

        self.size = size
        # Define action and observation space
        # Observations are either an image, a state, or a combination
        if not stable_baselines:
            self.observation_space = spaces.Dict({
                "image": spaces.Box(low=0, high=255, shape=(*size, 3 if not self.args['grayscale'] else 1), dtype=np.uint8),
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(3 + 3 + 3,), dtype=np.float32), # joint angles and gripper state as well as can location and differential to goal
                'reward': spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
                'is_first': spaces.Box(low=0, high=1, shape=(), dtype=bool),
                'is_last': spaces.Box(low=0, high=1, shape=(), dtype=bool),
                'is_terminal': spaces.Box(low=0, high=1, shape=(), dtype=bool),
            })
        else:
            self.observation_space = spaces.Dict({
                "image": spaces.Box(low=0, high=255, shape=(*size, 3 if not self.args['grayscale'] else 1), dtype=np.uint8),
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(3 + 3 + 3,), dtype=np.float32), # joint angles and gripper state as well as can location and differential to goal
            })

        self.last_arm_dofs = None

        # initialize if we haven't already
        gs.init(backend=gs.gpu, seed=random.randint(0, 2**30), precision="32", logging_level="warning")
        self.metadata = {
            "render_fps": 30
        }

        self.use_truncated_in_return = use_truncated_in_return

        self.stable_baselines = stable_baselines
        self.debug = debug
        if not self.debug:
            self.init_env()

        self.dp = []


    def _max_episode_steps(self):
        return self.args['time_limit']
        # return 100

    def init_env(self):
        self.kp = kp = 5

        self.scene = scene = gs.Scene(
            show_viewer=self.args['vis'],
        )

        self.plane = scene.add_entity(
            gs.morphs.Plane(),
        )

        BOTTLE_RADIUS = self.args['radius']
        BOTTLE_HEIGHT = self.args['height']
        BOX_WIDTH, BOX_HEIGHT = 0.75, 0.14

        self.box_pos = torch.Tensor((0.78, -BOX_WIDTH / 4, 0.02))
        self.box = scene.add_entity(
            material=gs.materials.Rigid(rho=5000),
                                        # friction=0.05),
                                        # coup_friction=0.05,),
            morph=gs.morphs.Box(
                size=(0.43, BOX_WIDTH, BOX_HEIGHT),
                pos=self.box_pos,
            )
        )

        self.cam_0 = scene.add_camera(
            fov=45,
            GUI=True,
        )

        self.cam_1 = scene.add_camera(
            # pos=(0.04, 0.0, 0.75),
            # lookat=(0.58, -BOX_WIDTH / 4, 0.02),
            pos=(0.3, 0.6, 0.5), 
            lookat=(0.5 ,0.0, 0.1), 
            up=(0, 0, 1),
            # up=(0, 1, 0),
            fov=45,
            GUI=False,
        )
        

        # TODO: see if you can prevent the gripper from being convexified
        self.kinova = kinova = scene.add_entity(
            gs.morphs.URDF(
                file=str(pl.Path(__file__).parent / 'gen3_lite_2f_robotiq_85.urdf'),
                fixed=True,
                convexify=True,
                pos=(0.0, 0.0, 0.055), # raise to account for table mount
            ),
            material=gs.materials.Rigid(friction=1.0),
            vis_mode="visual"

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
            surface=gs.surfaces.Default(
                color=(0.9, 0.3, 0.3, 1.0),
            ),
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

        # Wrist position
        wrist = kinova.get_link('end_effector_link')
        self.wrist_pos_offset = torch.Tensor([0.0, 0.0, 0.07]).to(device=wrist.get_pos().device)
        self.update_camera_position()

    def step(self, action):
        if self.debug:
            return self.observation_space.sample(), 0, False, False, {}
        # self.scene.clear_debug_objects()

        # Apply the action to the scene
        self.apply_action(action)
        self.update_camera_position()
        
        self.scene.step()
        obs = self.get_obs()
        reward = obs['reward']
        done = obs['is_last']
        if self.stable_baselines: 
            obs.pop('reward'); obs.pop('is_last')

        self.n_steps += 1

        if self.use_truncated_in_return:
            return obs, reward, done, self.n_steps >= self._max_episode_steps(), {'is_success': done}
        return obs, reward, done, {}
    
    def reset(self, trial_id=0, **kwargs):
        if self.debug: 
            return self.observation_space.sample(), {}
        # Reset the scene and get the initial observation
        self.n_steps = 0

        if trial_id in TRIALS_POSITION_0: bottle_pos = POSITION_0
        elif trial_id in TRIALS_POSITION_1: bottle_pos = POSITION_1
        elif trial_id in TRIALS_POSITION_2: bottle_pos = POSITION_2
        else: rand_idx = random.randint(0,2); bottle_pos = [POSITION_0, POSITION_1, POSITION_2][rand_idx]

        self.bottle.set_pos(bottle_pos); self.bottle.set_quat(torch.Tensor([1, 0, 0, 0]))
        self.goal_bottle.set_pos(STATIC_BOTTLE_POSITION); self.goal_bottle.set_quat(torch.Tensor([1, 0, 0, 0]))
        self.box.set_pos(self.box_pos); self.box.set_quat(torch.Tensor([1, 0, 0, 0]))
        self.kinova.set_dofs_position(np.array(KINOVA_START_DOFS_POS), self.kdofs_idx)

        # run a few steps to stabilize the scene
        for _ in range(10):
            self.scene.step()

        obs = self.get_obs()
        if self.stable_baselines: 
            obs.pop('reward'); obs.pop('is_last')

        if self.use_truncated_in_return:
            ret = obs, {}
        else:
            ret = obs
        return ret
    
    def get_obs(self, is_first=False):
        # Get the current observation from the scene
        # image = self.cam_0.render(rgb=True, depth=False, segmentation=False, normal=False, use_imshow=False)
        image = self.cam_1.render(rgb=True, depth=False, segmentation=False, normal=False, use_imshow=False)
        # from IPython import embed; embed(); exit(0)
        image = image[0] # grab the rgb
        # resize the image to the desired size
        image = cv2.resize(image, self.size)

        if self.args['grayscale']:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.expand_dims(image, axis=-1)

        # image = None
        # if not image:
        #     image = np.zeros((*self.size, 3), dtype=np.uint8)

        arm_pos = self.kinova.get_dofs_position(dofs_idx_local=self.kdofs_idx).cpu().numpy()

        eef_pos = self.kinova.get_link('end_effector_link').get_pos().cpu().numpy()
        eef_ang = self.kinova.get_link('end_effector_link').get_ang().cpu().numpy()
        bottle_pos = self.bottle.get_pos().cpu().numpy()
        # state = np.concatenate((arm_pos, bottle_pos))

        state = np.concatenate((eef_pos, eef_ang, bottle_pos))

        self.last_arm_dofs = arm_pos

        reward, done = self.compute_reward(state)

        if self.stable_baselines:
            # return image, state, reward, done
            return {"image": image, "state": state, "reward": reward, "is_last": done}
        else:
            return {"image": image, "state": state, "reward": reward, "is_first": is_first, "is_last": done, "is_terminal": False}
    
    def calc_gripper_force(self, cmd_gripper_pos, threshold=0.03):
        # Calculate the gripper force based on the gripper position
        pos = self.last_arm_dofs
        output_force = [0., 0.] #, 0., 0.]
        motor_cmd = (100 - cmd_gripper_pos) / 100
        right_error = pos[-4] + motor_cmd; right_error = right_error if abs(right_error) > threshold else [0.0]
        left_error = pos[-3] - motor_cmd; left_error = left_error if abs(left_error) > threshold else [0.0]
        right_fingertip_error = pos[-2] - KINOVA_START_DOFS_POS[-2]; right_fingertip_error = right_fingertip_error if abs(right_fingertip_error) > threshold else 0.0
        left_fingertip_error = pos[-1] - KINOVA_START_DOFS_POS[-1]; left_fingertip_error = left_fingertip_error if abs(left_fingertip_error) > threshold else 0.0

        output_force[0] = -self.kp*right_error[0];# output_force[2] = self.kp*right_fingertip_error
        output_force[1] = -self.kp*left_error[0]; #output_force[3] = self.kp*left_fingertip_error
        # print(output_force)
        return np.array(output_force)

    def apply_action(self, action, use_eef=True):
        if use_eef:
            eef_pos, eef_euler, gripper_pos = action[:3], action[3:6], action[6:]
            eef_quat = gs.utils.geom.xyz_to_quat(eef_euler)
            ik_joints = self.kinova.inverse_kinematics(self.kinova.get_link('end_effector_link'), pos=eef_pos, quat=eef_quat, rot_mask=[True, True, True])
            arm_pos = ik_joints[:-4]
        else: # use joint angles
            arm_pos, gripper_pos = action[:6], action[6:]

        gripper_force = self.calc_gripper_force(gripper_pos)


        # calculate the current positions distance from the desired positions
        current_position = self.kinova.get_dofs_position(dofs_idx_local=self.kdofs_idx).cpu().numpy()


        dp = 0 #np.linalg.norm(current_position[:6] - arm_pos, ord=2, axis=-1)
        self.dp.append(dp)

        # self.kinova.control_dofs_force(gripper_force, dofs_idx_local=np.array(self.kdofs_idx[-4:]))
        self.kinova.control_dofs_force(gripper_force, dofs_idx_local=np.array(self.kdofs_idx[-4:-2]))
        self.kinova.control_dofs_position(arm_pos, dofs_idx_local=self.kdofs_idx[:len(arm_pos)])

    def compute_reward(self, obs):
        reward = 0.
        done = False
        bottle_pos = self.bottle.get_pos()

        # Cup to goal distance
        # goal_pos = self.goal_bottle.get_pos()
        # distance = torch.linalg.norm(bottle_pos - goal_pos, ord=2, dim=-1, keepdim=True)
        # reward = -distance.item() # TODO: implement reward function
        # done = reward > -0.1 and (bottle_pos[2].cpu().numpy().item() >= (STATIC_BOTTLE_POSITION[2] - 0.07)) and (goal_pos[2].cpu().numpy().item() >= (STATIC_BOTTLE_POSITION[2] - 0.07))
        # cup slide contact

        # Pick up the cup, and penalize for contact with the ground plane.
        plane_contacts = self.kinova.get_contacts(self.plane)
        if plane_contacts['position'].shape[0] > 0:
            # print(f"CONTACT with plane.")
            reward -= 0.01
        elif bottle_pos[2].cpu().numpy().item() >= 0.14:
            print(f"SUCCESS!")
            reward = 1.
            done = True
        ## pick up cup

        return reward, done

    def update_camera_position(self):
        # Update the camera position based on the end effector position
        wrist = self.kinova.get_link('end_effector_link')
        position = wrist.get_pos() + self.wrist_pos_offset
        rotation = wrist.get_ang()

        # get the mean position of the two fingertips
        left_fingertip = self.kinova.get_link('left_finger_dist_link')
        right_fingertip = self.kinova.get_link('right_finger_dist_link')
        middle = (left_fingertip.get_pos() + right_fingertip.get_pos()) / 2


        # self.scene.draw_debug_sphere(middle, 0.01, color=(1, 0, 0))
        # self.scene.draw_debug_sphere(position, 0.01, color=(0, 1, 0))

        self.cam_0.set_pose(pos=position.cpu().numpy(), lookat=middle.cpu().numpy(), up=(0, 0, 1))

    def render(self, mode='human', use_imshow=False):
        # Render the scene
        img = None
        if mode == 'human':
            img = self.cam_0.render(rgb=True, depth=False, segmentation=False, normal=False, use_imshow=False)[0]
            img = cv2.resize(img, self.size)

            img2 = self.cam_1.render(rgb=True, depth=False, segmentation=False, normal=False, use_imshow=False)[0]
            if use_imshow:
                cv2.imshow('Genesis Gym', img)
                cv2.imshow('Genesis Gym 1', img2)
                cv2.waitKey(1)
        return img2
    
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
    parser.add_argument('--subsample', type=int, default=1, help='Subsample ratio for the demos')
    args = parser.parse_args()

    use_eef = True
    demo_player = GenesisDemoHolder(max_demos=args.max_demos, use_eef=use_eef, subsample_ratio=args.subsample)

    # print(GenesisGym.action_space, GenesisGym.action_space.low, GenesisGym.action_space.high)
    # ### Action normalization / unnormalization
    # while action := demo_player.next_action(normalize=False):
    #     original_action = action['action']
    #     normalized_action = _normalize_action(original_action, GenesisGym.action_space)
    #     unnormalized_action = _unnormalize_action(normalized_action, GenesisGym.action_space)
    #     print(f"orig, norm, unnorm: {' || '.join([f'{a:+.2f} {x:+.2f} {y:+.2f}' for a,x,y in zip(original_action, normalized_action, unnormalized_action)])}")

    # exit()

    def get_action():
        if args.random_agent:
            return GenesisGym.action_space.sample()
        else:
            action = demo_player.next_action(normalize=False)
            ret = action['action'] if action is not None else None

            if ret is not None and np.isnan(ret).any():
                print(f"!!NaN action!! {ret=} at index {demo_player.action_idx-1}")

            return ret


    env = GenesisGym(**args.__dict__)
    obs = env.reset()

    done = False
    max_reward = float('-inf'); reward = 0
    trials = 1; successful_trials = 0; steps = 0; pickups = 0

    from collections import defaultdict
    demonstrations = defaultdict({'image': [], 'state': [], 'action': [], 'reward': [], 'next_state': [], 'next_image': []})

    trial_id = demo_player.get_trial_id()
    while True:
        # action = env.action_space.sample()  # Sample random action
        action = get_action()

        if action is None or steps > env._max_episode_steps() or done:
            bottleZ = env.bottle.get_pos().cpu().numpy()[2]
            print(f"\t Max Reward {max_reward:+1.2f}. {bottleZ=}")
            max_reward = float('-inf')

            # close off the last demo
            demonstrations[trial_id]['done'][-1] = True

            trial_id = demo_player.next_demo()
            if reward > 0: successful_trials += 1
            if bottleZ > 0.15: pickups += 1
            if trial_id == -1:
                print("No more demos")
                break
            trials += 1; steps = 0; done = False

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
        output_path = pl.Path(f'./inthewild_trials_eef/{trial_id}_episodes.npy')
        # make the new directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, demo)

    # Append the results to a file. Create it if it doesn't exist.
    with open('results/results.txt', 'a') as f:
        f.write(f"subsample ratio {args.subsample} -- {'EEF' if use_eef else ''} {successful_trials/trials:.2%}\n")
        f.write(f"Trials: {trials} Successful Trials: {successful_trials} Success Rate: {successful_trials/trials:.2%}\n")
        f.write(f"Pickups: {pickups} Pickup Rate: {pickups/trials:.2%}\n")
        f.write(f"Max Reward: {max_reward}\n")
        f.write("================================================\n")