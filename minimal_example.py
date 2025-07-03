import numpy as np
import pathlib as pl

import genesis as gs
import torch
from genesis_sim2real.envs.kinova import JOINT_NAMES as kinova_joint_names, EEF_NAME as kinova_eef_name, TRIALS_POSITION_0, TRIALS_POSITION_1, TRIALS_POSITION_2
from genesis_sim2real.envs.demo_holder import GenesisDemoHolder
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

########################## init ##########################
gs.init(backend=gs.cpu)

########################## create a scene ##########################
scene = gs.Scene(
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
plane = scene.add_entity(
    gs.morphs.Plane(),
)

# add kinova Gen3 arm to scene
kinova = scene.add_entity(
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
obj_plastic = scene.add_entity(
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
kdofs_idx = [kinova.get_joint(name).dof_idx_local for name in kinova_joint_names]
eef = kinova.get_link(kinova_eef_name)
print(f"Kinova end effector: {eef}")

eef_link = kinova.get_link('end_effector_link')


########################## build ##########################
scene.build()

############ Optional: set control gains ############

target_eef_pos = eef_link.get_pos().cpu().numpy()
target_eef_euler = gs.utils.geom.quat_to_xyz(eef_link.get_quat()).cpu().numpy()
kinova.set_dofs_kp(
    kp             = 3*np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100]),
    dofs_idx_local = kdofs_idx,
)
kinova.set_dofs_position(np.array(KINOVA_START_DOFS_POS), kdofs_idx)

# # # Wrist position
# wrist_index = 'joint_6'
# wrist = kinova.get_joint(wrist_index)
# wrist_pos_offset = torch.Tensor([0.0, 0.0, 0.02]).to(device=wrist.get_pos().device)
# #self.update_camera_position()next_action

# self.kinova.control_dofs_force(gripper_force, dofs_idx_local=np.array(self.kdofs_idx[-4:]))
# self.kinova.control_dofs_position(arm_pos, dofs_idx_local=self.kdofs_idx[:len(arm_pos)])

def apply_action(action, target_eef_euler, target_eef_pos):

    delta_pos, delta_euler, gripper_pos = action[:3], action[3:6], action[-1:]
    
    target_eef_euler = target_eef_euler + delta_euler
    target_eef_pos = target_eef_pos + delta_pos

    target_quat = gs.utils.geom.xyz_to_quat(target_eef_euler)
    
    # Use IK to get joint angles
    ik_joints = kinova.inverse_kinematics(
        eef_link, 
        pos=target_eef_pos, 
        quat=target_quat, 
        rot_mask=[True, True, True]
    )
    arm_pos = ik_joints[:-4]

    kinova.control_dofs_force(gripper_force, dofs_idx_local=np.array(kdofs_idx[-4:-2]))
    kinova.control_dofs_position(arm_pos, dofs_idx_local=kdofs_idx[:len(arm_pos)])

def get_action():
    gripper_open_signal = 0
    return (0.0, 0.0, 0.0, gripper_open_signal)

# action = get_action()
# apply_action(action, target_eef_euler, target_eef_pos)

# get the end-effector link

# move to pre-grasp pose
# qpos = franka.inverse_kinematics(
#     link = eef,
#     pos  = np.array([0.65, 0.0, 0.25]),
#     quat = np.array([0, 1, 0, 0]),
# )
# # gripper open pos
# qpos[-2:] = 0.04
# path = franka.plan_path(
#     qpos_goal     = qpos,
#     num_waypoints = 200, # 2s duration
# )
# # execute the planned path
# for waypoint in path:
#     franka.control_dofs_position(waypoint)
#     scene.step()

for i in range(1000):
    scene.step()

