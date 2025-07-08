import numpy as np
import matplotlib.pyplot as plt
import pathlib as pl
import gymnasium
from gymnasium import spaces
import genesis as gs
import torch
from genesis_sim2real.envs.kinova import JOINT_NAMES as kinova_joint_names, EEF_NAME as kinova_eef_name, TRIALS_POSITION_0, TRIALS_POSITION_1, TRIALS_POSITION_2
from genesis_sim2real.envs.actions import KinovaActions
#################### Deafult Args ########################
FINGERTIP_POS = 1.0
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

class GenesisGym():
    # Locations where the robot can move to?
    # Change action space to use relative movements
    action_space = spaces.Box(
        low=np.array([-0.025, -0.025, -0.025, -0.05, -0.05, -0.05, 0]), 
        high=np.array([0.025, 0.025, 0.025, 0.05, 0.05, 0.05, 100.]), 
        shape=(7,), 
        dtype=np.float32
    )

    def __init__(self):
        ########################## init ##########################
        gs.init(backend=gs.cpu,
        logging_level = 'warning')
    
    def init_env(self):
        self.kp = kp = 5
        ########################## create a scene ##########################
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt       = 4e-3,
                substeps = 10,
            ),
            mpm_options=gs.options.MPMOptions(
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
        )

        # add kinova Gen3 arm to scene
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
        self.obj_plastic = self.scene.add_entity(
            material=gs.materials.MPM.ElastoPlastic(),
            morph=gs.morphs.Sphere(
                pos  = (0.5, 0, 0.25),
                radius = 0.05,
            ),
            surface=gs.surfaces.Default(
                color    = (0.4, 1.0, 0.4),
                vis_mode = 'particle',
            ),
        )

        # Get kinova degrees of freedom
        self.kdofs_idx = [self.kinova.get_joint(name).dof_idx_local for name in kinova_joint_names]
        self.eef = self.kinova.get_link(kinova_eef_name)
        print(f"Kinova end effector: {self.eef}")

        ########################## build ##########################
        self.scene.build()

        ############ Optional: set control gains ############
        print("Moving to default position")
        target_eef_pos = self.eef.get_pos().cpu().numpy()
        target_eef_euler = gs.utils.geom.quat_to_xyz(self.eef.get_quat()).cpu().numpy()
        self.kinova.set_dofs_kp(
            kp             = 3*np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100]),
            dofs_idx_local = self.kdofs_idx,
        )
        self.kinova.set_dofs_position(np.array(KINOVA_START_DOFS_POS), self.kdofs_idx)

        for i in range(50):
            self.scene.step()

    def calc_gripper_force(self, cmd_gripper_pos, threshold=0.03):
        # Calculate the gripper force based on the gripper position
        pos = self.last_arm_dofs
        output_force = [0., 0.] #, 0., 0.]
        motor_cmd = (100 - cmd_gripper_pos) / 100
        right_error = pos[-4] + motor_cmd; right_error = right_error if abs(right_error) > threshold else [0.0]
        left_error = pos[-3] - motor_cmd; left_error = left_error if abs(left_error) > threshold else [0.0]
        right_fingertip_error = pos[-2] - KINOVA_START_DOFS_POS[-2]; right_fingertip_error = right_fingertip_error if abs(right_fingertip_error) > threshold else 0.0
        left_fingertip_error = pos[-1] - KINOVA_START_DOFS_POS[-1]; left_fingertip_error = left_fingertip_error if abs(left_fingertip_error) > threshold else 0.0

        print("Right error: ", right_error)
        output_force[0] = -self.kp*right_error
        output_force[1] = -self.kp*left_error
        # print(output_force)
        return np.array(output_force)
    
    def apply_action(self, action, use_eef=True):
        if use_eef: # diff eef action
            # Apply relative changes to current position and orientation
            # print(', '.join([f"{x:+.5f}" for x in action]))
            delta_pos, delta_euler, gripper_pos = action[:3], action[3:6], action[-1]
            # delta_pos, delta_yaw, gripper_pos = action[:3], action[5], action[-1:]
            
            self.target_eef_euler = self.target_eef_euler + delta_euler
            # self.target_eef_euler = self.target_eef_euler + np.array([0, 0, delta_yaw])
            self.target_eef_pos = self.target_eef_pos + delta_pos

            target_quat = gs.utils.geom.xyz_to_quat(self.target_eef_euler)
            
            # Use IK to get joint angles
            ik_joints = self.kinova.inverse_kinematics(
                self.eef, 
                pos=self.target_eef_pos, 
                quat=target_quat, 
                rot_mask=[True, True, True]
            )
            arm_pos = ik_joints[:-4]
        # else:
        #     arm_pos, gripper_pos = action[:6], action[6:]

        print("Gripper position:", gripper_pos)
        gripper_force = self.calc_gripper_force(gripper_pos)

        # Apply controls
        self.kinova.control_dofs_force(gripper_force, dofs_idx_local=np.array(self.kdofs_idx[-4:-2]))
        print("Setting action")
        self.kinova.control_dofs_position(arm_pos, dofs_idx_local=self.kdofs_idx[:len(arm_pos)])

    def get_action(self, gripper_signal):
        print("Getting gripper open signal")
        return (0.0, 0.0, 0.0, gripper_open_signal)
    
    def step(self, action):
        self.apply_action(action)

        for i in range(10):
            self.scene.step()
    
    def get_obs(self):
        arm_pos = self.kinova.get_dofs_position(dofs_idx_local=self.kdofs_idx).cpu().numpy()

        eef_pos = self.eef.get_pos().cpu().numpy()
        eef_euler = gs.utils.geom.quat_to_xyz(self.eef.get_quat()).cpu().numpy()
        finger_joint_pos = [arm_pos[-4]]
        self.last_arm_dofs = arm_pos

    def reset(self):
        # run a few steps to stabilize the scene
        for _ in range(10):
            self.scene.step()
        
        self.last_arm_dofs = self.kinova.get_dofs_position(dofs_idx_local=self.kdofs_idx).cpu().numpy()

        self.target_eef_pos = self.eef.get_pos().cpu().numpy()
        self.target_eef_euler = gs.utils.geom.quat_to_xyz(self.eef.get_quat()).cpu().numpy()

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
            v = int((yi - y_min) / (y_max - y_min + 1e-8) * (resolution - 1))

            # Keep max Z in each XY bin (visible from above)
            if np.isnan(grid[u, v]) or zi > grid[u, v]:
                grid[u, v] = zi

        visible_z = grid[~np.isnan(grid)]
        print(f"Visible top Z-particles: {visible_z.shape[0]} points")

        return visible_z

gripper_open_signal = 0
gripper_close_signal = 100

# Set up genesis environment
simulation = GenesisGym()
simulation.init_env()
simulation.reset()
simulation.get_obs()

# # Open the gripper
# print("Opening the gripper")
# action = simulation.get_action(gripper_open_signal)
# simulation.step(action)
# simulation.reset()

# # Close the gripper
# print("Closing the gripper?")
# action = simulation.get_action(gripper_close_signal)
# simulation.step(action)
# simulation.reset()

# # Get the clay depth
# print("Getting clay depth")
# depth = simulation.get_clay_depth()
# visible_z = simulation.get_visible_top_particles()
#print(visible_z)

# plt.plot(visible_z)
# plt.show()

# Move arm to clay
print("Moving arm to clay")
step_size = 0.1
action = (step_size,  step_size,  step_size, 0.0)
simulation.step(action)