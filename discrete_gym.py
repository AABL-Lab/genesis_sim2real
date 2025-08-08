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
#from genesis.utils import mesh_utils, morphs, materials, surfaces
import trimesh 
#from genesis.morphs import Mesh
import time
#################### Deafult Args ########################
FINGERTIP_POS = 1.0
KINOVA_START_DOFS_POS = [0.3268500269015339, -1.4471734542578538, 2.3453266624159497, -1.3502152158191212, 2.209384006676201, -1.5125125137062945, -1, 1, FINGERTIP_POS, FINGERTIP_POS]
arm_position_dofs= [-9.4415337e-02, -1.6932747e+00, -4.7102508e-01,  -1.5638263e+00,
  2.0483732e+00,  1.4763145e+00, -6.3851485e-03, -2.4709428e-02,
  7.4626954e-04, -9.4122291e-03]

arm_position2 = [-9.4415337e-02, -1.6932747e+00, -4.7102508e-01,  1.5638263e+00,
  2.0483732e+00,  1.4763145e+00, -6.3851485e-03, -2.4709428e-02,
  7.4626954e-04, -9.4122291e-03]

arm_position2[1] += 0.5  # small increase (in radians)


STATIC_BOTTLE_POSITION = torch.tensor((0.65, -0.225, 0.17))
PX, PZ = 0.465, 0.05
POSITION_0 = (0.5, 0.0, 0.06)
CLAY_RADIUS = 0.06
POSITION_1 = torch.tensor((PX, -0.05, PZ))
POSITION_2 = torch.tensor((PX, -0.2, PZ))

## Default Args
DEFAULT_RADIUS = 0.034
DEFAULT_HEIGHT = 0.09
DEFAULT_RHO = 2000
DEFAULT_FRICTION = 0.5
DEFAULT_STARTING_X = 0.65

class GenesisGym(gym.Env):
    # Locations where the robot can move to?
    # Change action space to use relative movements

    # Actions are 7 continuous actions, 6 dof joint angles, 1 gripper position
    #action_space = spaces.Box(low=np.array([-3.14, -3.14, -3.14, -3.14, -3.14, -3.14, 0]), high=np.array([3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 100.]), shape=(7,), dtype=np.float32)


    def __init__(self):
        ########################## init ##########################
        gs.init(backend=gs.cpu,
        logging_level = 'warning')

        # initialize observation and action space
        obs_dim = 22  # or use len(obs) from _get_obs() if dynamic
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        #self.action_space = spaces.Box(low=np.array([-1, -1, -1, -3.14, -3.14, -3.14, 0]), high=np.array([1, 1, 1, 3.14, 3.14, 3.14, 100.]), shape=(7,), dtype=np.float32)  # [dx, dy, dz, gripper]
        # Discretize action space
        bins = [5, 5, 5, 10]  # dx, dy, dz, gripper
        n_actions = np.prod(bins)  # Total number of discrete actions
        self.action_space = spaces.Discrete(n_actions)
        self.prev_eef_pos = None
        self.is_done = False
        self.trial_number = 0
        self.last_gripper_pos = 0
        
    def unscale_action(self, action_index):
        # Convert single int back to multi-index
        bins = [5, 5, 5, 10]  # dx, dy, dz, gripper
        multi_index = np.unravel_index(action_index, bins)

        # Map each index to real value
        xyz = np.linspace(-1, 1, bins[0])
        gripper = np.linspace(0, 100, bins[3])

        return np.array([
            xyz[multi_index[0]],
            xyz[multi_index[1]],
            xyz[multi_index[2]],
            gripper[multi_index[3]],
        ])


    def init_env(self):
        self.kp = kp = 5
        dt = 3e-3

        ########################## create a scene ##########################
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                # simulation timestep
                dt       = 4e-3,
                substeps = 10,
            ),
            mpm_options=gs.options.MPMOptions(
                dt=dt,
                lower_bound=( -1.0,  -1.0, -1.0),
                upper_bound=( 1.0,  1.0,  1.0),
                gravity=(0, 0, 0), # mimic gravity compensation
                #enable_CPIC=True,
            ),
            vis_options=gs.options.VisOptions(
                visualize_mpm_boundary = True,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_fov=30,
            ),
            show_viewer = True,
        )

        ########################## entities ##########################
        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
            material=gs.materials.Rigid(friction=5),  # very grippy surface
        )

        #add kinova Gen3 lite arm to scene
        self.kinova = self.scene.add_entity(
                    gs.morphs.URDF(
                        file=str('/home/reu_2025/Genesis_Experiments/genesis_sim2real/genesis_sim2real/envs/gen3_lite_2f_robotiq_85.urdf'),
                        fixed=True,
                        convexify=True,
                        pos=(0.0, 0.0, 0.055), # raise to account for table mount
                    ),
                    material=gs.materials.Rigid(friction=1.0),
                    vis_mode="visual"

                    # gs.morphs.MJCF(file="/home/j/workspace/genesis_pickaplace/005_tomato_soup_can/google_512k/kinbody.xml"),
                )

        
        
        # Use ElastoPlastic for clay sphere
        # Temporily changing to rigid for 
        sphere_morph = gs.morphs.Sphere(
                    pos  = POSITION_0,
                    radius = CLAY_RADIUS,
                    collision = True,
                )
        
        sphere_trimesh = trimesh.creation.icosphere(radius=0.05, subdivisions=3)
        components = sphere_trimesh.split(only_watertight=False)
        print(f"Number of connected components: {len(components)}")

        self.obj_plastic = self.scene.add_entity(
            material=gs.materials.MPM.ElastoPlastic(
            ),
            morph=gs.morphs.Sphere(
                pos  = POSITION_0,
                radius = CLAY_RADIUS,
                collision = True,
            ),
            surface=gs.surfaces.Default(
                color    = (0.4, 1.0, 0.4),
                vis_mode = 'particle',
                
            ),
        )
        self.scene.build()
        # self.obj_plastic.pin_particles_by_condition(lambda p: p[2] < 0.02)
        # self.obj_plastic.apply_pinning()
        # self.obj_plastic.mark_pinned()

        state = self.obj_plastic.get_state()
        positions = state.pos.numpy()  # shape: (n_particles, 3)
        threshold = 0.02
        self.indices_to_pin = np.where(positions[:, 2] < threshold)[0]
        print(f"Pining {len(self.indices_to_pin)} particles to plane")
        self.obj_plastic.pin_particles_by_condition(lambda p: p[2] < 0.05)
        # # Get kinova degrees of freedom
        self.kdofs_idx = [self.kinova.get_joint(name).dof_idx_local for name in kinova_joint_names]
        self.eef = self.kinova.get_link(kinova_eef_name)
        
        # print(f"Kinova end effector: {self.eef}")

        ########################## build ##########################
        #self.scene.build()


        self.target_eef_euler = gs.utils.geom.quat_to_xyz(self.eef.get_quat()).cpu().numpy()
        self.prev_pos = [0, 0, 0]
        self.prev_dist = None
    
    def set_viewer(self, on_off):
        self.scene.show_viewer = on_off
    
    def quaternion_multiply(self, q1, q2):
        r1 = R.from_quat(q1)
        r2 = R.from_quat(q2)
        result = r1 * r2  # Applies r2 after r1
        return result.as_quat()
    
    def calc_gripper_force(self, cmd_gripper_pos, threshold=0.03):
        pos = self.last_arm_dofs
        output_force = [0., 0.]

        # print("gripper dofs:", pos[-4], pos[-3])
        motor_cmd = cmd_gripper_pos / 100
        right_error = pos[-4] + motor_cmd
        right_error = right_error if abs(right_error) > threshold else 0.0  # FIXED

        left_error = pos[-3] - motor_cmd
        left_error = left_error if abs(left_error) > threshold else 0.0  # FIXED

        right_fingertip_error = pos[-2] - KINOVA_START_DOFS_POS[-2]
        right_fingertip_error = right_fingertip_error if abs(right_fingertip_error) > threshold else 0.0

        left_fingertip_error = pos[-1] - KINOVA_START_DOFS_POS[-1]
        left_fingertip_error = left_fingertip_error if abs(left_fingertip_error) > threshold else 0.0

        output_force[0] = -self.kp * right_error
        output_force[1] = -self.kp * left_error
        return np.array(output_force)

    
    def apply_action(self, action, use_eef=True):

        print("applying delta eef action")
        delta_pos, gripper_pos = action[:3], action[-1]
        # delta_pos, delta_yaw, gripper_pos = action[:3], action[5], action[-1:]
        
        # Update the current position and euler angle
        current_pos = self.eef.get_pos().cpu().numpy()
        current_quat = self.eef.get_quat().cpu().numpy()

        self.target_eef_pos = current_pos

        self.target_eef_euler = np.array([np.pi, 0.0, 0.0])  # fixed downward
        target_quat = gs.utils.geom.xyz_to_quat(self.target_eef_euler)
        # Use IK to get joint angles
        ik_joints = self.kinova.inverse_kinematics(
            self.eef, 
            pos=self.target_eef_pos, 
            quat=target_quat, 
            rot_mask=[True, True, True]
        )
        arm_pos = ik_joints[:-4]


        self.kinova.control_dofs_position(arm_pos, dofs_idx_local=self.kdofs_idx[:len(arm_pos)])
        self.last_gripper_pos = gripper_pos / 100.0  # normalize to [0, 1]

    
    def get_action(self, gripper_signal):
        # print("Getting gripper open signal")
        return GenesisGym.action_space.sample()
        #return (0.0, 0.0, 0.0, gripper_open_signal)
    
    def step(self, action):
        print("stepping the scene")
        
        #action = np.clip(action, -1.0, 1.0)
        scaled_action = self.unscale_action(action)
        delta = scaled_action[:3] * 0.05  # (dx, dy, dz)
        gripper_control = scaled_action[-1]  # [-1, +1]

        # Scale gripper value from [-1,1] → [0, 100]
        gripper_pos = np.interp(gripper_control, [-1.0, 1.0], [0, 100])

        # Full action = [dx, dy, dz, rx, ry, rz, gripper]
        #full_action = np.concatenate([delta, [0, 0, 0], [gripper_pos]])
        self.apply_action(scaled_action)

        for i in range(10):
            self.obj_plastic.apply_pinning()
            self.scene.step()

        obs = self._get_obs()
        reward = self._compute_reward(obs)
        print("Current reward", reward)
        terminated = self.is_done
        truncated = False

        return obs, reward, terminated, truncated, {}
        
    def _get_obs(self):
        arm_pos = self.kinova.get_dofs_position(dofs_idx_local=self.kdofs_idx).cpu().numpy()
        eef_pos = self.eef.get_pos().cpu().numpy()
        eef_euler = gs.utils.geom.quat_to_xyz(self.eef.get_quat()).cpu().numpy()

        if self.prev_eef_pos is None:
            eef_vel = np.zeros(3)
        else:
            eef_vel = (eef_pos - self.prev_eef_pos) / (self.scene.sim_options.dt * self.scene.sim_options.substeps)

        self.prev_eef_pos = eef_pos

        clay_pos = self.obj_plastic.get_particles()
        mean_clay_pos = np.mean(clay_pos, axis=0)
        std_clay_pos = np.std(clay_pos, axis=0)
        clay_min = np.min(clay_pos, axis=0)
        clay_max = np.max(clay_pos, axis=0)
        clay_extent = clay_max - clay_min

        goal_vec = mean_clay_pos - eef_pos

        obs = np.concatenate([
            eef_pos,               # 3
            eef_euler,             # 3
            eef_vel,               # 3
            mean_clay_pos,         # 3
            std_clay_pos,          # 3
            clay_extent,           # 3
            goal_vec,              # 3
            [self.last_gripper_pos]  # 1
        ])
        
        return obs.astype(np.float32)

    def get_mean_particle_location(self, clay_pos):
        # print(np.shape(clay_pos))
        # print ("x positions", clay_pos[:, 0])
        x_avg = np.sum(clay_pos[:, 0]) / len(clay_pos[:, 0])
        y_avg = np.sum(clay_pos[:, 1]) / len(clay_pos[:, 0])
        z_avg = np.sum(clay_pos[:, 2]) / len(clay_pos[:, 0])
        return [x_avg, y_avg, z_avg]
        #print(" average positions: (", x_avg, y_avg, z_avg, ")")
    
    def reset(self, seed=None, options=None):
        self.is_done = False
        print("resetting scene", "trial number", self.trial_number)
        self.trial_number += 1
        self.scene.reset()

        default_pose = [0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0]

        # print("Going to pose:", default_pose)
        self.kinova.set_dofs_position(np.array(arm_position2), self.kdofs_idx)
        #time.sleep(5)
        action = []

        # run a few steps to stabilize the scene
        for _ in range(10):
            self.scene.step()
        
        obs = self._get_obs()
        return obs, {}
       
    def get_clay_depth(self):
        # Fetch the current scene state
        state = self.obj_plastic.get_state()

        # The returned state usually contains a `pos` tensor for particle positions
        positions = state.pos.cpu().numpy()  # shape: (N_particles, 3)

        # Z-axis min and max
        z_min = positions[:, 2].min()
        z_max = positions[:, 2].max()
        print("min and max z axis data:", z_min, z_max)

        z_width = z_max - z_min
        print(f"Current object Z-width (height/thickness): {z_width:.4f} meters")

        # Get all z-axis particle data:
        z_data = positions[:, 2]
        print(np.shape(z_data))
        return z_width

    def get_visible_top_particles(self, resolution=128):
        """
        Returns Z-values of particles that are visible from above (top-down),
        using a discretized XY grid (like an orthographic camera).
        """
        state = self.obj_plastic.get_state()
        positions = state.pos.cpu().numpy()

        # Project particles onto XY plane
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

        # Normalize coordinates into a grid (bounding box)
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        # Grid resolution (like pixels)
        grid = np.full((resolution, resolution), fill_value=np.nan)

        for xi, yi, zi in zip(x, y, z):
            u = int((xi - x_min) / (x_max - x_min + 1e-8) * (resolution - 1))
            v = int((yi - y_min) /gripper_pos (y_max - y_min + 1e-8) * (resolution - 1))

            # Keep max Z in each XY bin (visible from above)
            if np.isnan(grid[u, v]) or zi > grid[u, v]:
                grid[u, v] = zi

        visible_z = grid[~np.isnan(grid)]
        print(f"Visible top Z-particles: {visible_z.shape[0]} points")

        return visible_z


    def _compute_reward(self, state):
        eef, goal = state[:3], state[3:6]
        horiz_err = np.linalg.norm(goal[:2] - eef[:2])
        vert_err = abs(goal[2] - eef[2])
        current_dist = np.linalg.norm(goal - eef)

        reward = -1.0 * horiz_err - 2.0 * vert_err

        plane_contacts = self.kinova.get_contacts(self.plane)
        # holder_contacts = self.kinova.get_contacts(self.holder)
        # per-step horizontal progress bonus
        if self.prev_dist is not None:
            prev_horiz = np.linalg.norm(self.prev_goal[:2] - self.prev_pos[:2])
            delta_h = prev_horiz - horiz_err
            if delta_h > 0:
                reward += 0.2 * delta_h

        # direction+vertical penalty        # # Penalize contact with the holder
        # if holder_contacts['position'].shape[0] > 0:
        #     print("holder collision")
        #     reward -= 1.0  # Increased penalty
        #     self.is_done = True

        movement = eef - self.prev_pos

        if np.linalg.norm(movement) > 1e-6:
            move_dir = movement / np.linalg.norm(movement)
            goal_dir = (goal - eef) / (current_dist + 1e-6)
            reward += 0.1 * np.dot(move_dir, goal_dir)
            reward -= 0.3 * abs(move_dir[2])

        # success condition
        if current_dist < 0.05:
            reward += 10.0
            self.is_done = True
        
        # # Penalize contact with the holder
        # if holder_contacts['position'].shape[0] > 0:
        #     print("holder collision")
        #     reward -= 1.0  # Increased penalty
        #     self.is_done = True

        # Penalize contact with the plane
        if plane_contacts['position'].shape[0] > 0:
            print("plane collision")
            reward -= 1.0  # Increased penalty
            self.is_done = True

        # reward for being lower than clay depth
        # reward for being closer to center 
        # reward for opening gripper
        # reward for adding contours to clay
        self.prev_pos = eef.copy()
        self.prev_goal = goal.copy()
        self.prev_dist = current_dist
        return reward




