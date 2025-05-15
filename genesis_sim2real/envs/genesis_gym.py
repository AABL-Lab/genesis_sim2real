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
from geometry_msgs.msg import PoseStamped

if DIGITAL_TWIN := False:
    import rospy
    import armpy
    from sensor_msgs.msg import JointState


FINGERTIP_POS = -0.9
KINOVA_START_DOFS_POS = [0.3268500269015339, -1.4471734542578538, 2.3453266624159497, -1.3502152158191212, 2.209384006676201, -1.5125125137062945, -1, 1, FINGERTIP_POS, FINGERTIP_POS]
STATIC_BOTTLE_POSITION = torch.tensor((0.65, -0.225, 0.17))
PX, PZ = 0.465, 0.05
POSITION_0 = torch.tensor((PX, 0.1, PZ))
POSITION_1 = torch.tensor((PX, -0.05, PZ))
POSITION_2 = torch.tensor((PX, -0.2, PZ))

## Default Args
DEFAULT_RADIUS = 0.03
DEFAULT_HEIGHT = 0.085
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

class GenesisGym(gymnasium.Env):
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
        super().__init__()
        self.args = {
            'rho': kwargs['rho'] if 'rho' in kwargs else DEFAULT_RHO,
            'radius': kwargs['radius'] if 'radius' in kwargs else DEFAULT_RADIUS,
            'height': kwargs['height'] if 'height' in kwargs else DEFAULT_HEIGHT,
            'friction': kwargs['friction'] if 'friction' in kwargs else DEFAULT_FRICTION,
            'vis': kwargs['vis'] if 'vis' in kwargs else False,
            'grayscale': kwargs['grayscale'] if 'grayscale' in kwargs else False,
            'time_limit': kwargs['time_limit'] if 'time_limit' in kwargs else 2000, #4000,
            'env_name': kwargs['env_name'] if 'env_name' in kwargs else 'lift',
            # 'starting_x': args.starting_x if 'starting_x' in args else 0.65
            }
        
        # print(f"**kwargs: {kwargs}")
        print(f"GenesisGym Task={self.args['env_name']} args: {self.args}")

        self.size = size
        # Define action and observation space
        # Observations are either an image, a state, or a combination
        if not stable_baselines:
            self.observation_space = spaces.Dict({
                "image": spaces.Box(low=0, high=255, shape=(*size, 3 if not self.args['grayscale'] else 1), dtype=np.uint8),
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(3 + 3 + 3 + 3 + 3 + 1,), dtype=np.float32), # eef pos, angles, vel, and angvel and gripper state as well as can location
                # "state": spaces.Box(low=-np.inf, high=np.inf, shape=(6 + 3 + 1,), dtype=np.float32), # joint angles and gripper state as well as can location
                'reward': spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
                'is_first': spaces.Box(low=0, high=1, shape=(), dtype=bool),
                'is_last': spaces.Box(low=0, high=1, shape=(), dtype=bool),
                'is_terminal': spaces.Box(low=0, high=1, shape=(), dtype=bool),
            })
        else:
            self.observation_space = spaces.Dict({
                "image": spaces.Box(low=0, high=255, shape=(*size, 3 if not self.args['grayscale'] else 1), dtype=np.uint8),
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(3 + 3 + 3 + 3 + 3 + 1,), dtype=np.float32), # eef pos, angles, vel, and angvel and gripper state as well as can location
                # "state": spaces.Box(low=-np.inf, high=np.inf, shape=(6 + 3 + 1,), dtype=np.float32), # joint angles and gripper state as well as can location
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
        self.total_steps = 0; self.n_steps = 0

        self.check_saved_positions = check_saved_positions
        if self.check_saved_positions:
            adjusted_can_pos_path = pl.Path(f'~/workspace/genesis_sim2real/trial_can_adjusted.npy').expanduser()
            self.adjusted_can_pos = None
            if adjusted_can_pos_path.exists():
                print(f"Loading adjusted can positions from {adjusted_can_pos_path}")
                adjusted_can_pos = np.load(adjusted_can_pos_path, allow_pickle=True).item()
                for trial_id, pos in adjusted_can_pos.items():
                    print(f"Trial {trial_id} adjusted can position: trial_id > 0 and {pos}")
                self.adjusted_can_pos = adjusted_can_pos
            else: print(f"WARNING: No adjusted can positions found at {adjusted_can_pos_path}. Will not check for adjusted can positions.")


        self.DIGITAL_TWIN = DIGITAL_TWIN
        self.angular_waypoints = [] 
        self.can_position = None
        if self.DIGITAL_TWIN:
            self.arm = armpy.initialize('gen3_lite')
            self.joint_state = None; self.gripper_closed = False;
            rospy.init_node("arm_reacher")
            rospy.sleep(1.0)
            rospy.Subscriber('/my_gen3_lite/base_feedback/joint_state', JointState, self.joint_state_callback)
            rospy.Subscriber('/can_detector/pose', PoseStamped, self.can_pose_callback)

    def joint_state_callback(self, msg):
        self.joint_state = msg

    def can_pose_callback(self, msg):
        self.can_position = msg.pose.position

    def _max_episode_steps(self):
        return self.args['time_limit']
        # return 100

    def init_env(self):
        self.kp = kp = 2
        self.gripper_kp = 10

        self.scene = scene = gs.Scene(
            show_viewer=self.args['vis'],
        )

        self.plane = scene.add_entity(
            gs.morphs.Plane(),
        )

        self.BOTTLE_RADIUS = BOTTLE_RADIUS = self.args['radius']
        self.BOTTLE_HEIGHT = BOTTLE_HEIGHT = self.args['height']
        BOX_WIDTH, BOX_HEIGHT = 0.75, 0.12

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

        self.rec_cam = scene.add_camera(
            res    = (1280, 960),
            pos    = (-0.5, -0.4, 0.2),
            lookat = (1.0, 0, 0.0),
            fov    = 30,
            GUI    = False
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
        scene.build()

        ############ Optional: set control gains ############
        # set positional gains
        kinova.set_dofs_kp(
            kp             = self.kp*np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100]),
            dofs_idx_local = kdofs_idx,
        )
        kinova.set_dofs_position(np.array(KINOVA_START_DOFS_POS), kdofs_idx)

        # Wrist position
        self.eef_link = self.kinova.get_link(kinova_eef_name)
        self.wrist_pos_offset = torch.Tensor([0.0, 0.0, 0.07]).to(device=self.eef_link.get_pos().device)
        self.update_camera_position()

    def step(self, action):
        if self.debug:
            return self.observation_space.sample(), 0, False, False, {}
        # self.scene.clear_debug_objects()

        if self.use_truncated_in_return:
            action = _unnormalize_action(action, self.action_space)
        if self.total_steps % 1000 == 0:
            print(f'gym sees action {action}', ', '.join(f"{a:+0.1f}" for a in action))


        # Apply the action to the scene
        self.apply_action(action)
        self.update_camera_position()
        
        self.scene.step()
        obs = self.get_obs()
        reward = obs['reward']
        done = obs['is_last']
        if self.stable_baselines: 
            obs.pop('reward'); obs.pop('is_last')

        self.n_steps += 1; self.total_steps += 1

        # self.angular_waypoints.append(action[:6])
        gripper_toggle = (action[-1] > 50 and not self.gripper_closed) or (action[-1] < 50 and self.gripper_closed)
        if self.DIGITAL_TWIN and self.n_steps % 5 == 0 or gripper_toggle:
        # if self.DIGITAL_TWIN and len(self.angular_waypoints) > 200 or gripper_toggle:
        # if False and self.DIGITAL_TWIN and self.n_steps % 3 == 0:
            # self.arm.goto_joint_waypoints(self.angular_waypoints, radians=True, block=True); self.angular_waypoints = []

            self.arm.goto_joint_pose(action[:6], radians=True, block=False)
            if gripper_toggle:
                if self.gripper_closed:  self.arm.open_gripper(); self.gripper_closed = False
                else: self.arm.close_gripper(); self.gripper_closed = True

            # if not self.gripper_closed and action[-1] > 50:
            #     self.arm.close_gripper(); self.gripper_closed = True
            # elif self.gripper_closed and action[-1] < 50:
            #     self.arm.open_gripper(); self.gripper_closed = False

        if self.use_truncated_in_return:
            return obs, reward, done, self.n_steps >= self._max_episode_steps(), {'is_success': done}


        return obs, reward, done, {}
    
    def reset(self, trial_id=0, **kwargs):
        if self.debug: 
            return self.observation_space.sample(), {}
        # Reset the scene and get the initial observation
        self.n_steps = 0

        # add some gaussian noise in the x and y direction
        # random_offset = 0.005 * torch.Tensor([torch.randn(1), torch.randn(1), 0.0])
        if self.can_position is not None:
            sensed_bottle_pos = torch.Tensor([self.can_position.x, self.can_position.y, self.can_position.z])
            # apply a static offset that moves the position from the middle of the bottom of the can to the front face halfway up
            static_offset = torch.Tensor([-self.BOTTLE_RADIUS * 0.8, self.BOTTLE_RADIUS * 0.2, self.BOTTLE_HEIGHT * 0.4])
            print(f"Sensed can position: {sensed_bottle_pos}. Static offset: {static_offset}")

            bottle_pos = sensed_bottle_pos + static_offset
            # draw a debug sphere centered
            self.scene.draw_debug_arrow(pos=sensed_bottle_pos, vec=static_offset, radius=0.01, color=(1, 0, 0, 0.5))  # Green

        elif trial_id > 0 and self.check_saved_positions and self.adjusted_can_pos is not None:
            # check if the trial_id is in the adjusted_can_pos
            if trial_id in self.adjusted_can_pos:
                print(f"Trial {trial_id} adjusted can position: {self.adjusted_can_pos[trial_id]}")
                bottle_pos = torch.Tensor(self.adjusted_can_pos[trial_id])
            else:
                print(f"Trial {trial_id} not in adjusted can positions. Using default position.")
                bottle_pos = POSITION_0
        elif trial_id in TRIALS_POSITION_0: bottle_pos = POSITION_0
        elif trial_id in TRIALS_POSITION_1: bottle_pos = POSITION_1
        elif trial_id in TRIALS_POSITION_2: bottle_pos = POSITION_2
        else: rand_idx = random.randint(0,2); bottle_pos = [POSITION_0, POSITION_1, POSITION_2][rand_idx]

        # bottle_pos += random_offset

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

        if self.DIGITAL_TWIN:
            backup_position = [0.34551798719466237, -0.8454950565561763, 2.169129261535217, -1.232747441193471, 1.4586096006108726, -1.686383909690952] #, 0.5953540153613426]
            target_joint_positions = [0.3268500269015339, -1.4471734542578538, 2.3453266624159497, -1.3502152158191212, 2.209384006676201, -1.5125125137062945] #, -0.0877648122691288]
            print(f"Opening gripper ", end='')
            self.arm.open_gripper(); rospy.sleep(1.0) # deal with problems from switching between vel mode and pos mode.
            print(f"Done.")
            while self.joint_state is None:
                print(f"Waiting for joint state...")
                rospy.sleep(1.0)
            for tjp in [backup_position, target_joint_positions]:
                while not np.allclose(self.joint_state.position[:6], tjp, atol=0.1):
                    print(f"\tMoving to {tjp}. Distance from target {np.linalg.norm(np.array(self.joint_state.position[:6]) - np.array(tjp))}")
                    self.arm.goto_joint_pose(tjp, radians=True, block=False)
                    rospy.sleep(5.0)


        return ret
    
    def picture_in_picture(self, image0, image1):
        # Resize the image to the desired size
        image0 = cv2.resize(image0, (int(self.size[0] / 4), int(self.size[1] / 4)))
        image1[:int(self.size[0] / 4), :int(self.size[1] / 4)] = image0
        return image1

    def get_obs(self, is_first=False, picture_in_picture=True):
        # Get the current observation from the scene
        # image = self.cam_0.render(rgb=True, depth=False, segmentation=False, normal=False, use_imshow=False)
        image = self.cam_1.render(rgb=True, depth=False, segmentation=False, normal=False, use_imshow=False)
        # from IPython import embed; embed(); exit(0)
        image = image[0] # grab the rgb
        # resize the image to the desired size
        image = cv2.resize(image, self.size)

        if picture_in_picture:
            # grab the image from cam_0, shrink it, and put it in the corner of the main image
            image2 = self.cam_0.render(rgb=True, depth=False, segmentation=False, normal=False, use_imshow=False)[0]
            # image2 = cv2.resize(image2, (int(self.size[0] / 4), int(self.size[1] / 4)))
            # image[:int(self.size[0] / 4), :int(self.size[1] / 4)] = image2
            image = self.picture_in_picture(image2, image)

        if self.args['grayscale']:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.expand_dims(image, axis=-1)

        # image = None
        # if not image:
        #     image = np.zeros((*self.size, 3), dtype=np.uint8)

        arm_pos = self.kinova.get_dofs_position(dofs_idx_local=self.kdofs_idx).cpu().numpy()

        eef_pos = self.eef_link.get_pos().cpu().numpy()
        eef_euler = gs.utils.geom.quat_to_xyz(self.eef_link.get_quat()).cpu().numpy()
        eef_lin_vel = self.eef_link.get_vel().cpu().numpy() # linear velocity
        eef_ang_vel = self.eef_link.get_ang().cpu().numpy() # angular velocity
        bottle_pos = self.bottle.get_pos().cpu().numpy()
        finger_joint_pos = [arm_pos[-4]]
        state = np.concatenate((eef_pos, eef_euler, eef_lin_vel, eef_ang_vel, bottle_pos, finger_joint_pos))
        
        # arm_no_gripper_pos = arm_pos[:-4]
        # state = np.concatenate((arm_no_gripper_pos, bottle_pos, finger_joint_pos))

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
        
        if motor_cmd > 0.5: motor_cmd = max([1.0], motor_cmd) # make the gripper close more

        right_error = pos[-4] + motor_cmd; right_error = right_error if abs(right_error) > threshold else [0.0]
        left_error = pos[-3] - motor_cmd; left_error = left_error if abs(left_error) > threshold else [0.0]
        right_fingertip_error = pos[-2] - KINOVA_START_DOFS_POS[-2]; right_fingertip_error = right_fingertip_error if abs(right_fingertip_error) > threshold else 0.0
        left_fingertip_error = pos[-1] - KINOVA_START_DOFS_POS[-1]; left_fingertip_error = left_fingertip_error if abs(left_fingertip_error) > threshold else 0.0

        output_force[0] = -self.gripper_kp*right_error[0];# output_force[2] = self.kp*right_fingertip_error
        output_force[1] = -self.gripper_kp*left_error[0]; #output_force[3] = self.kp*left_fingertip_error
        # print(output_force)
        return np.array(output_force)

    def apply_action(self, action, use_eef=False):
        if use_eef:
            # eef_pos, eef_euler, gripper_pos = action[:3], action[3:6], action[6:]
            # eef_quat = gs.utils.geom.xyz_to_quat(eef_euler)
            eef_link = self.eef_link
            # curr_eef_pos = eef_link.get_pos().cpu().numpy()
            curr_eef_quat = eef_link.get_quat().cpu().numpy()

            eef_pos, eef_yaw, gripper_pos = action[:3], action[3], action[-1:]
            # eef_quat = gs.utils.geom.xyz_to_quat(np.array([curr_eef_ang[2], curr_eef_ang[1], curr_eef_ang[0]]))

            curr_eef_euler = gs.utils.geom.quat_to_xyz(curr_eef_quat)
            # curr_eef_quat = gs.utils.geom.xyz_to_quat(curr_eef_euler)
            eef_quat = gs.utils.geom.xyz_to_quat(np.array([curr_eef_euler[0], eef_yaw, curr_eef_euler[2]]))


            # print(f'eef_ang {curr_eef_ang}, decoded eef_ang {gs.utils.geom.quat_to_xyz(curr_eef_quat)}')
            # print(f'eef_quat {curr_eef_quat}, decoded eef_quat {gs.utils.geom.xyz_to_quat(curr_eef_ang)}')
            # eef_quat = gs.utils.geom.xyz_to_quat(curr_eef_ang)

            ik_joints = self.kinova.inverse_kinematics(eef_link, pos=eef_pos, quat=eef_quat, rot_mask=[False, False, True])
            # ik_joints = self.kinova.inverse_kinematics(eef_link, pos=curr_eef_pos, quat=curr_eef_quat, rot_mask=[False, False, True])

            arm_pos = ik_joints[:-4]
        else: # use joint angles
            arm_pos, gripper_pos = action[:6], action[6:]

        gripper_force = self.calc_gripper_force(gripper_pos)


        # calculate the current positions distance from the desired positions
        # current_position = self.kinova.get_dofs_position(dofs_idx_local=self.kdofs_idx).cpu().numpy()


        dp = 0 #np.linalg.norm(current_position[:6] - arm_pos, ord=2, axis=-1)
        self.dp.append(dp)

        # self.kinova.control_dofs_force(gripper_force, dofs_idx_local=np.array(self.kdofs_idx[-4:]))
        self.kinova.control_dofs_force(gripper_force, dofs_idx_local=np.array(self.kdofs_idx[-4:-2]))
        self.kinova.control_dofs_position(arm_pos, dofs_idx_local=self.kdofs_idx[:len(arm_pos)])

    def compute_reward(self, vstate):
        reward = 0.
        done = False
        bottle_pos = self.bottle.get_pos()


        plane_contacts = self.kinova.get_contacts(self.plane)
        if DO_REAL_TASK := True:
            # Cup to goal distance
            goal_pos = self.goal_bottle.get_pos()
            distance = torch.linalg.norm(bottle_pos - goal_pos, ord=2, dim=-1, keepdim=True)

            if bottle_pos[2].cpu().numpy().item() >= 0.14: # Give some shaped reward, if you lift the bottle you get a reward
                reward += 0.01

            if plane_contacts['position'].shape[0] > 0:
                # print(f"CONTACT with plane.")
                reward -= 0.001
            
            if distance < 0.09: # and vstate[-1] < -0.5: # open gripper and close to goal
                # make sure the gripper and bottle are not in collision
                bottle_contacts = self.kinova.get_contacts(self.bottle)
                if bottle_contacts['position'].shape[0] > 0:
                    pass # Don't reward until the bottle is released
                    # print(f"CONTACT with bottle at distance {distance.item():.2f} {vstate[-1]:.2f}")
                    # reward -= 0.001
                else:
                    print(f"SUCCESS! {distance.item():.2f} {vstate[-1]:.2f}")
                    reward = 10.
                    done = True

            # elif distance < 0.15:
            #     print(f"{distance.item():.2f} {vstate[-1]:.2f} ")

            # print(f'{vstate[-1]:+1.2f}')
            # cup slide contact
        else:
            # Pick up the cup, and penalize for contact with the ground plane.
            if self.args['env_name'] == 'point':
                eef_joint_pos = self.eef_link.get_pos().cpu().numpy()
                goal_pos = self.goal_bottle.get_pos().cpu().numpy()
                distance = np.linalg.norm(eef_joint_pos - goal_pos, ord=2)
                reward -= distance
                if distance < 0.1:
                    print(f"SUCCESS!")
                    reward = 1.
                    done = True
                # else:
                    # print the points and distance
                    # print(f"Distance: {distance:.2f}, EEF pos: {eef_joint_pos}, Goal pos: {goal_pos}")
            else:
                if plane_contacts['position'].shape[0] > 0:
                    # print(f"CONTACT with plane.")
                    reward -= 0.001
                elif bottle_pos[2].cpu().numpy().item() >= 0.14: # lift task
                    print(f"SUCCESS!")
                    reward = 1.
                    done = True
            ## pick up cup

        return reward, done

    def update_camera_position(self):
        # Update the camera position based on the end effector position
        wrist = self.eef_link
        position = wrist.get_pos() + self.wrist_pos_offset
        rotation = wrist.get_ang()

        # get the mean position of the two fingertips
        left_fingertip = self.kinova.get_link('left_finger_dist_link')
        right_fingertip = self.kinova.get_link('right_finger_dist_link')
        middle = (left_fingertip.get_pos() + right_fingertip.get_pos()) / 2


        # self.scene.draw_debug_sphere(middle, 0.01, color=(1, 0, 0))
        # self.scene.draw_debug_sphere(position, 0.01, color=(0, 1, 0))

        self.cam_0.set_pose(pos=position.cpu().numpy(), lookat=middle.cpu().numpy(), up=(0, 0, 1))

    def get_grip_pose(self):
        # get the average position of the fingertips
        left_fingertip = self.kinova.get_link('left_finger_dist_link').get_pos().cpu().numpy()
        right_fingertip = self.kinova.get_link('right_finger_dist_link').get_pos().cpu().numpy()
        return np.mean([left_fingertip, right_fingertip], axis=0)
    
    def set_can_to_pose(self, pos):
        self.bottle.set_pos(pos)
        self.bottle.set_quat(torch.Tensor([1, 0, 0, 0]))

    def render(self, mode='human', use_imshow=False):
        # Render the scene
        img = None
        if mode == 'human':
            img = self.cam_0.render(rgb=True, depth=False, segmentation=False, normal=False, use_imshow=False)[0]
            img = cv2.resize(img, self.size)

            img2 = self.cam_1.render(rgb=True, depth=False, segmentation=False, normal=False, use_imshow=False)[0]

            img2 = self.picture_in_picture(img, img2)

            if use_imshow:
                cv2.imshow('Genesis Gym', img)
                cv2.imshow('Genesis Gym 1', img2)
                cv2.waitKey(1)
        return img2
    
if __name__ == '__main__':
    from genesis_sim2real.envs.demo_holder import GenesisDemoHolder
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

            # reset the env
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
            if args.vis: 
                img = next_obs['image']
                cv2.imshow('image', img)
                cv2.waitKey(1)
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
        output_path = pl.Path(f'./inthewild_trials_eef_SB3/{trial_id}_episodes.npy')
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