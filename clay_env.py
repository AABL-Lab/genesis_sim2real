import gymnasium as gym
import numpy as np
from gymnasium import spaces
import genesis as gs
import torch

CLAY_RADIUS = 0.04
POSITION_0 = (0.5, 0, 0.1)
KINOVA_START_DOFS_POS = [0.3268, -1.447, 2.345, -1.350, 2.209, -1.512, -1, 1, 1.0, 1.0]
kinova_joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
kinova_eef_name = "tool_frame"
from genesis_sim2real.envs.kinova import JOINT_NAMES as kinova_joint_names, EEF_NAME as kinova_eef_name, TRIALS_POSITION_0, TRIALS_POSITION_1, TRIALS_POSITION_2

class GenesisGym(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        gs.init(backend=gs.cpu, logging_level="warning")

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)  # dx, dy, dz, gripper
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        self.scene = gs.Scene(sim_options=gs.options.SimOptions(dt=4e-3, substeps=10))
        self.plane = self.scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(friction=5))
        
        self.kinova = self.scene.add_entity(
            gs.morphs.URDF(
                file="/home/reu_2025/Genesis_Experiments/genesis_sim2real/genesis_sim2real/envs/gen3_lite_2f_robotiq_85.urdf",
                fixed=True,
                convexify=True,
                pos=(0.0, 0.0, 0.055),
            ),
            material=gs.materials.Rigid(friction=1.0),
        )

        self.clay = self.scene.add_entity(
            material=gs.materials.MPM.ElastoPlastic(),
            morph=gs.morphs.Sphere(pos=POSITION_0, radius=CLAY_RADIUS, collision=True),
            surface=gs.surfaces.Default(color=(0.4, 1.0, 0.4), vis_mode="particle"),
        )

        self.scene.build()

        self.kdofs_idx = [self.kinova.get_joint(name).dof_idx_local for name in kinova_joint_names]
        self.eef = self.kinova.get_link(kinova_eef_name)
        self.target_eef_euler = gs.utils.geom.quat_to_xyz(self.eef.get_quat()).cpu().numpy()

        self.reset()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        delta_pos = action[:3] * 0.02
        gripper_pos = np.interp(action[3], [-1, 1], [0, 100])

        current_pos = self.eef.get_pos().cpu().numpy()
        target_pos = current_pos + delta_pos
        target_quat = gs.utils.geom.xyz_to_quat(self.target_eef_euler)

        ik_joints = self.kinova.inverse_kinematics(self.eef, pos=target_pos, quat=target_quat)
        arm_pos = ik_joints[self.kdofs_idx]

        self.kinova.control_dofs_position(arm_pos, dofs_idx_local=self.kdofs_idx)

        for _ in range(10):
            self.scene.step()

        obs = self._get_obs()
        reward = self._compute_reward(obs)
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {}

    def get_mean_particle_location(self, clay_pos):
        # print(np.shape(clay_pos))
        # print ("x positions", clay_pos[:, 0])
        x_avg = np.sum(clay_pos[:, 0]) / len(clay_pos[:, 0])
        y_avg = np.sum(clay_pos[:, 1]) / len(clay_pos[:, 0])
        z_avg = np.sum(clay_pos[:, 2]) / len(clay_pos[:, 0])
        return [x_avg, y_avg, z_avg]
    
    def _get_obs(self):
        eef_pos = self.eef.get_pos().cpu().numpy()
        eef_euler = gs.utils.geom.quat_to_xyz(self.eef.get_quat()).cpu().numpy()
        clay_pos = self.clay.get_particles()
        #print("clay_pos", clay_pos)
        clay_mean = self.get_mean_particle_location(clay_pos)
        return np.concatenate([eef_pos, eef_euler, clay_mean, [0.0]]).astype(np.float32)

    def _compute_reward(self, obs):
        eef_pos = obs[:3]
        clay_pos = obs[6:9]
        dist = np.linalg.norm(eef_pos - clay_pos)
        return -dist  # Reward = negative distance

    def reset(self, seed=None, options=None):
        print("resetting")
        self.scene.reset()
        self.kinova.set_dofs_position(np.array(KINOVA_START_DOFS_POS), self.kdofs_idx)
        for _ in range(10):
            self.scene.step()
        obs = self._get_obs()
        return obs, {}
