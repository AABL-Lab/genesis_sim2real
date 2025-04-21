JOINT_NAMES = [
    'joint_1',
    'joint_2',
    'joint_3',
    'joint_4',
    'joint_5',
    'joint_6',
    'left_finger_bottom_joint',
    'right_finger_bottom_joint',
    'left_finger_tip_joint',
    'right_finger_tip_joint',
]

EEF_NAME = 'end_effector_link'

trials_position_0_successful = [232, 235, 242, 245, 248, 251, 254, 257, 261, 265, 269, 273, 276, 279, 283, 293, 297, 301, 304, 308, 315, 316, 319, 320, 325, 328, 331, 335]
trials_position_0_failed = [238, 249, 260, 268, 282, 288, 307, 312, 324, 331, 334]
trials_position_1_successful = [233, 236, 239, 243, 246, 252, 255, 258, 262, 266, 274, 277, 280, 284, 287, 294, 298, 302, 305, 309, 317, 321, 326, 329, 322]
trials_position_1_failed = [270, 313]
trials_position_2_successful = [327, 330, 333, 300, 303, 306, 311, 318, 278, 281, 286, 290, 295, 299, 256, 259, 263, 267, 275, 234, 237, 244, 247, 250]
trials_position_2_failed = [240, 253, 285, 289, 296, 310, 314, 322]

TRIALS_POSITION_0 = trials_position_0_successful # + trials_position_0_failed
TRIALS_POSITION_1 = trials_position_1_successful # + trials_position_1_failed
TRIALS_POSITION_2 = trials_position_2_successful # + trials_position_2_failed