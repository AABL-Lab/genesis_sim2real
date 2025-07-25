# genesis_sim2real

Clone this repo.

Install genesis as a submodule. You can use this as your primary Genesis package if you want, but its a submodule so we can grab modules from it. 
$ git submodule init && git submodule update

Install genesis using their instructions https://github.com/Genesis-Embodied-AI/Genesis

I had to make one change in the Genesis repo itself, it was hanging as it tried to show images, that change can be found in the genesis_changes directory. It should be put in the same path in the Genesis project Genesis/genesis/camera/vis.py

The joint states of the in-the-wild study we did are stored in inthewild_trials. 

You can load up the joint states and play them with:
$python main.py 

The logic in here is messy, but it's trying to dynamically figure out where the can should be for the trial to succeed, it saves out those good positions in "trial_can_adjusted.npy" so it the trajectories can be replayed, but with the good can position, into a reinforcement learning system.
