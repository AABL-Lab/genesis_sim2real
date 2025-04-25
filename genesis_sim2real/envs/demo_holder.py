import pathlib as pl
import numpy as np
import torch
import genesis as gs
from genesis_sim2real.envs.genesis_gym import _normalize_action

class GenesisDemoHolder:
    """
    Class to hold the demo data for the Genesis environment.
    """
    def __init__(self, max_demos=float('inf'), use_eef=False, subsample_ratio=1):

        self.dir = pl.Path('/home/j/workspace/genesis_sim2real/inthewild_trials_eef/') if use_eef else pl.Path('/home/j/workspace/genesis_sim2real/inthewild_trials/')
        self.paths = self.dir.glob('*episodes.npy')
        self.subsample_ratio = subsample_ratio
        self.use_eef = use_eef

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

    def next_action(self, normalize=False, diff_eef=False):
        if not self.NEXT_DEMO_CALLED: self.NEXT_DEMO_CALLED = True # If we take an action, next demo should move us forward
        if self.action_idx >= len(self.demos[self.idx][1]):
            return None
        
        if diff_eef:
            assert self.use_eef, "Cannot use diff eef if use_eef is False"
            if self.action_idx == 0:
                action = np.zeros_like(self.demos[self.idx][1][self.action_idx])
            else:
                action = self.demos[self.idx][1][self.action_idx][:6] - self.demos[self.idx][1][self.action_idx-1][:6]
                action = [*action, self.demos[self.idx][1][self.action_idx][-1]]
        else: action = self.demos[self.idx][1][self.action_idx]

        if normalize: # map from action space to [-1, 1]
            # print(f'original action: {" ".join([f"{x:+.2f}" for x in action])}')
            action = _normalize_action(action)
            # print(f"\tnorm action: {' '.join([f'{x:+.2f}' for x in action])}")

        self.action_idx += 1

        return {'action': action}
    
    def convert_eef_to_diff_eef(self):
        # convert the current demo to a diff eef demo
        # for each action, calculate the difference between the current and previous eef position
        # and store the difference
        diff_eef_demo = []
        for idx, action in enumerate(self.demos[self.idx][1]):
            if idx == 0:
                pass
                # diff_eef_demo.append(action)
            else:
                diff_eef_action = action[:6] - self.demos[self.idx][1][idx-1][:6]
                # diff_eef_demo.append(diff_eef_action + [self.demos[self.idx][1][idx-1][-1]]) # add the gripper position back in
                diff_eef_demo.append([*diff_eef_action, self.demos[self.idx][1][idx-1][-1]]) # add the gripper position back in

        # do a check and print out the difference between the original and the original + diff
        # for idx in range(len(self.demos[self.idx][1])):
        #     if idx == 0:
        #         original = self.demos[self.idx][1][idx]
        #     else:
        #         diff = diff_eef_demo[idx-1] # diff between idx-1 and idx
        #         next_original = self.demos[self.idx][1][idx]
        #         next_with_diff = original + diff
        #         # print(f"Original: {original}, Next original: {next_original}, Next with diff: {next_with_diff}")
        #         if np.linalg.norm(next_original - next_with_diff) > 1e-4:
        #             print(f"Difference between original and original + diff: {np.linalg.norm(next_original - next_with_diff)}")
        #         original = next_original
        return np.array(diff_eef_demo)
        
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

if __name__ == '__main__':
    from genesis_sim2real.envs.genesis_gym import _normalize_action, _unnormalize_action
    demo_holder = GenesisDemoHolder(max_demos=10, use_eef=False)

    for idx, (trial_id, d) in enumerate(demo_holder.demos):
        for i in range(d.shape[0]):
            print(f"{trial_id} {i}: {d[i]}")
    # diff_eef_demo = demo_holder.convert_eef_to_diff_eef()
    # print(diff_eef_demo[-1])


            