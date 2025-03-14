import rosbag
from sensor_msgs.msg import Image as RosImage, Joy, JointState
from pathlib import Path
from collections import defaultdict
import numpy as np

def state_from_jointstate(msg: JointState):
    return msg.position[:6] # only first 6 joints, gripper from basefeedback

def action_from_joy(msg):
    return -msg.buttons[4] if msg.buttons[4] else msg.buttons[5]

def state_from_basefeedback(msg):
    gripper_pos = msg.interconnect.oneof_tool_feedback.gripper_feedback[0].motor[0].position
    tool_pose = msg.base.tool_pose_x, msg.base.tool_pose_y, msg.base.tool_pose_z, msg.base.tool_pose_theta_x, msg.base.tool_pose_theta_y, msg.base.tool_pose_theta_z
    # return [*tool_pose, gripper_pos]
    return [gripper_pos]

def read_bag(trial_id):
    path = Path('~').expanduser() / f'user_{trial_id}' / 'trial_data.bag'

    topic_to_fn = {
        '/my_gen3_lite/joint_states': state_from_jointstate,
        '/joy': action_from_joy,
        '/my_gen3_lite/base_feedback': state_from_basefeedback
    }

    filter_zeros = False
    velocity_commands = []
    must_have_keys = [entry for entry in list(topic_to_fn.keys()) if entry not in ['/my_gen3_lite/in/cartesian_velocity', '/joy']]
    msg_topics = set(); t0 = None; hz = 60; frame = defaultdict(list); bag_start = None; total_frames = 0
    bag = rosbag.Bag(path)

    ep_dict = defaultdict(list)
    frame_idx = 0
    for topic, msg, t in bag.read_messages(): # NOTE: we're dropping the last frame
        if not t0: t0 = t.to_sec()
        if not bag_start: bag_start = t.to_sec()
        # print(f'{t.to_sec() - bag_start:1.2f}')
        # if topic == '/my_gen3_lite/in/cartesian_velocity':
        #     print('\\t', msg.twist)

        dt = t.to_sec() - t0
        if dt >= 1/hz:
            if all(len(frame[k]) > 0 for k in must_have_keys):
                joint_vel_mean = np.mean(frame['/my_gen3_lite/joint_states'], axis=0)
                gripper_pos = np.mean(frame['/my_gen3_lite/base_feedback'], axis=0)
                # print(frame_idx, ','.join([f'{c:+1.2f}' for c in joint_vel_mean]), gripper_pos)
                frame_idx += 1
                ep_dict['vel_cmd'].append(joint_vel_mean)
                ep_dict['gripper_pos'].append(gripper_pos)

            # Reset t0
            t0 = t.to_sec()
            frame = defaultdict(list)

        if topic not in msg_topics:
            print(topic, type(msg)) #, msg)
            msg_topics.add(topic)

        if topic in topic_to_fn:
            val = topic_to_fn[topic](msg)
            # if np.isnan(val).any():
            #     print(f'nan in {topic} {val}')
            frame[topic].append(val)

    print(f'bag runtime {t.to_sec() - t0:1.2f} seconds')

    outdir = Path('./inthewild_trials' )
    outdir.mkdir(exist_ok=True)

    # write out the dictionary
    outpath = outdir / f'{trial_id}_episodes.npy'
    np.save(outpath, ep_dict)
    print(f'wrote out to {outpath}')

    return ep_dict['vel_cmd'], ep_dict['gripper_pos']

if __name__ == '__main__':
    trials_position_0_successful = [232, 235, 242, 245, 248, 251, 254, 257, 261, 265, 269, 273, 276, 279, 283, 293, 297, 301, 304, 308, 315, 316, 319, 320, 325, 328, 331, 335]
    trials_position_0_failed = [238, 249, 260, 268, 282, 288, 307, 312, 324, 331, 334]

    trials_position_1_successful = [233, 236, 239, 243, 246, 252, 255, 258, 262, 266, 274, 277, 280, 284, 287, 294, 298, 302, 305, 309, 317, 321, 326, 329, 322]
    trials_position_1_failed = [270, 313]

    trials_position_2_successful = [327, 330, 333, 300, 303, 306, 311, 318, 278, 281, 286, 290, 295, 299, 256, 259, 263, 267, 275, 234, 237, 244, 247, 250]
    trials_position_2_failed = [240, 253, 285, 289, 296, 310, 314, 322]

    trials = trials_position_0_successful + trials_position_1_successful + trials_position_2_successful
    trials += trials_position_0_failed + trials_position_1_failed + trials_position_2_failed

    for trial_id in trials:
        joint_vel, gripper_pos = read_bag(trial_id)
        print(trial_id, f'joint_vel: {len(joint_vel)} gripper_pos: {len(gripper_pos)}')