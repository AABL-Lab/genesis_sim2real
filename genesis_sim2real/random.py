# The identity quaternion is [0, 0, 0, 1]
simulation.apply_delta_action([0.05, 0.0, 0.0])
for _ in range(100):
    simulation.scene.step()

simulation.apply_delta_action([0.0, 0.0, -0.2])
for _ in range(100):
    simulation.scene.step()

simulation.apply_delta_action([0.0, 0.0, 0.2])
for _ in range(100):
    simulation.scene.step()

simulation.apply_delta_action([0.0, 0.0, -0.2])
for _ in range(100):
    simulation.scene.step()

simulation.apply_delta_action([0.0, 0.0, 0.2])
for _ in range(100):
    simulation.scene.step()
print("done")
# simulation.apply_delta_action([0.0, 0.0, -0.2, 0.0, 0.0, 0.0, 0.0])
# for _ in range(200):
#     simulation.scene.step()

# simulation.apply_action([0.5, 0.0, 0.3, 0.0, 1.0, 0.0, 0.0])
# for _ in range(100):
#     simulation.scene.step()

# simulation.apply_action([0.5, 0.0, 0.4, 0.0, 1.0, 0.0, 0.0])
# for _ in range(100):
#     simulation.scene.step()

# simulation.apply_action([0.5, 0.0, 0.3, 0.0, 1.0, 0.0, 0.0])
# for _ in range(100):
#     simulation.scene.step()

# simulation.apply_action([0.5, 0.0, 0.4, 0.0, 1.0, 0.0, 0.0])
# for _ in range(100):
############# Test resetting the environment for each trial ############

#simulation.reset()

# #arm_pos = ik_joints[:-4]  # exclude gripper joints
# print("moving to default_pose:", default_pose)
# # default_pose = [0, 0, 0.5, 0.5, 1, 0.0, 0.0]
# simulation.apply_action(default_pose)
# # simulation.kinova.control_dofs_position(arm_pos, dofs_idx_local=simulation.kdofs_idx[:len(arm_pos)])
# action = []
# # Step the scene
# for _ in range(200):
#     simulation.scene.step()

# default_pose = [0.5, 0.0, 0.3, 1, 1, 1, 1]

# #arm_pos = ik_joints[:-4]  # exclude gripper joints
# print("moving to default_pose:", default_pose)
# # default_pose = [0, 0, 0.5, 0.5, 1, 0.0, 0.0]
# simulation.apply_action(default_pose)
# # simulation.kinova.control_dofs_position(arm_pos, dofs_idx_local=simulation.kdofs_idx[:len(arm_pos)])
# action = []
# # Step the scene
# for _ in range(200):
#     simulation.scene.step()

############# Code allows the gripper to open and close #################
# print("moving to open gripper position")
# # half_open_action = simulation.calc_gripper_force(gripper_half_open_signal)
# # print("half_open action", half_open_action)
# default_pose = [0.5, 0.0, 0.5, 0.0, 1, 0.0, 0.0, 100]
# simulation.apply_action(default_pose)

# for _ in range(200):
#     simulation.scene.step()

# print("moving to closed gripper position")
# # half_open_action = simulation.calc_gripper_force(gripper_half_open_signal)
# # print("half_open action", half_open_action)
# default_pose = [0.5, 0.0, 0.5, 0.0, 1, 0.0, 0.0, -100]
# simulation.apply_action(default_pose)

# for _ in range(200):
#     simulation.scene.step()

# print("moving to half open gripper position")
# default_pose = [0.5, 0.0, 0.5, 0.0, 1, 0.0, 0.0, 10]
# # apply a small force to open the gripper and then stop it by sending 0
# simulation.apply_action(default_pose)

# for _ in range(200):
#     simulation.scene.step()
# default_pose = [0.5, 0.0, 0.5, 0.0, 1, 0.0, 0.0, 0]
# simulation.apply_action(default_pose)

# for _ in range(200):
#     simulation.scene.step()
############## RL model ################
# Observation: EEF Position
# Action: Delta movement
# Reward: Negative distance to target
# Policy: Move tward clay


########## Nonsense code idk what im doing ##############

# action = np.concatenate([delta_pos, delta_euler, [gripper_pos]])
# delta_pos: change in end-effector cartesian position in meteres (dx, dy, dz)
# delta_euler: change in eef orientation as euler angles (droll, dpitch, dyaw) in radians
# gripper pos: command to open or close the gripper (0-100)

# for _ in range(20):
#     current_pos = simulation.eef.get_pos().cpu().numpy()
#     delta_pos = np.array([0.5, 0.0, 0.30]) - current_pos
#     delta_pos = np.clip(delta_pos, -0.025, 0.025)  # step size

#     action = np.concatenate([delta_pos, np.zeros(3), [0.0]])
#     simulation.step(action)

# # Close gripper once positioned
# close_grip_action = np.concatenate([np.zeros(3), np.zeros(3), [100.0]])
# simulation.step(close_grip_action)

# # EEF positions -> Inverse Kinematics -> joint position
# # IK calculates joint angles based on desured eef pose

# # Move arm up and over the clay
# delta_euler = np.array([0.0, np.pi / 12, 0.0])  # ~15° pitch forward
# delta_pos = np.array([0.0, 0.0, +0.015])  # move up 1.5 cm

# gripper_pos = 0.0  # Keep gripper open for now
# action = np.concatenate([delta_pos, delta_euler, [gripper_pos]])
# simulation.step(action)

# for _ in range(5):  # or more for gradual motion
#     delta_pos = np.array([0.0, 0.0, 0.015])
#     delta_euler = np.array([0.0, np.pi / 48, 0.0])  # small pitch increment
#     action = np.concatenate([delta_pos, delta_euler, [0.0]])
#     simulation.step(action)

# # Set pitch to π radians → gripper points down
# target_euler = np.array([0.0, np.pi, 0.0])
# print("target_euler", target_euler)
# target_quat = gs.utils.geom.xyz_to_quat(torch.tensor(target_euler))
# print("target_quat", target_quat)
# # Keep current EEF position, only print("moving to open gripper position")
# # half_open_action = simulation.calc_gripper_force(gripper_half_open_signal)
# # print("half_open action", half_open_action)
# simulation.apply_action((0, 0, 0, 100))

# for _ in range(200):
#     simulation.scene.step()

# print("moving to closed gripper position")
# # half_open_action = simulation.calc_gripper_force(gripper_half_open_signal)
# # print("half_open action", half_open_action)
# simulation.apply_action((0, 0, 0, -100))

# for _ in range(200):
#     simulation.scene.step()

# print("moving to half open gripper position")
# # apply a small force to open the gripper and then stop it by sending 0
# simulation.apply_action((0, 0, 0, 10))

# for _ in range(200):
#     simulation.scene.step()

# simulation.apply_action((0, 0, 0, 0))

# for _ in range(200):
#     simulation.scene.step()change orientation
# current_pos = simulation.eef.get_pos().cpu().numpy()
# print("current position", current_pos)
# print("Calculating eef position")
# ik_joints = simulation.kinova.inverse_kinematics(
#     simulation.eef,
#     pos=current_pos,
#     quat=target_quat,
#     rot_mask=[True, True, True]
# )

