import pathlib as pl
import numpy as np
import torch
import genesis as gs
from genesis_sim2real.envs.genesis_gym import _normalize_action

# --- Define Expanded Discrete Action Indices (29 Actions) ---

# Gripper (Indices 0-1)
ACTION_OPEN_GRIPPER = 0
ACTION_CLOSE_GRIPPER = 1

# Single Axis Movement (Indices 2-7)
ACTION_MOVE_PX = 2  # +X
ACTION_MOVE_NX = 3  # -X
ACTION_MOVE_PY = 4  # +Y
ACTION_MOVE_NY = 5  # -Y
ACTION_MOVE_PZ = 6  # +Z
ACTION_MOVE_NZ = 7  # -Z

# Two Axis Movement (Indices 8-19)
# XY Plane
ACTION_MOVE_PX_PY = 8   # +X+Y
ACTION_MOVE_PX_NY = 9   # +X-Y
ACTION_MOVE_NX_PY = 10  # -X+Y
ACTION_MOVE_NX_NY = 11  # -X-Y
# XZ Plane
ACTION_MOVE_PX_PZ = 12  # +X+Z
ACTION_MOVE_PX_NZ = 13  # +X-Z
ACTION_MOVE_NX_PZ = 14  # -X+Z
ACTION_MOVE_NX_NZ = 15  # -X-Z
# YZ Plane
ACTION_MOVE_PY_PZ = 16  # +Y+Z
ACTION_MOVE_PY_NZ = 17  # +Y-Z
ACTION_MOVE_NY_PZ = 18  # -Y+Z
ACTION_MOVE_NY_NZ = 19  # -Y-Z

# Three Axis Movement (Indices 20-27)
ACTION_MOVE_PX_PY_PZ = 20 # +X+Y+Z
ACTION_MOVE_PX_PY_NZ = 21 # +X+Y-Z
ACTION_MOVE_PX_NY_PZ = 22 # +X-Y+Z
ACTION_MOVE_PX_NY_NZ = 23 # +X-Y-Z
ACTION_MOVE_NX_PY_PZ = 24 # -X+Y+Z
ACTION_MOVE_NX_PY_NZ = 25 # -X+Y-Z
ACTION_MOVE_NX_NY_PZ = 26 # -X-Y+Z
ACTION_MOVE_NX_NY_NZ = 27 # -X-Y-Z

# No Operation (Index 28)
ACTION_NO_OP = 28

# Map indices to names for clarity (optional)
ACTION_NAMES = {
    0: "Open Gripper", 1: "Close Gripper",
    2: "Move +X", 3: "Move -X", 4: "Move +Y", 5: "Move -Y", 6: "Move +Z", 7: "Move -Z",
    8: "Move +X+Y", 9: "Move +X-Y", 10: "Move -X+Y", 11: "Move -X-Y",
    12: "Move +X+Z", 13: "Move +X-Z", 14: "Move -X+Z", 15: "Move -X-Z",
    16: "Move +Y+Z", 17: "Move +Y-Z", 18: "Move -Y+Z", 19: "Move -Y-Z",
    20: "Move +X+Y+Z", 21: "Move +X+Y-Z", 22: "Move +X-Y+Z", 23: "Move +X-Y-Z",
    24: "Move -X+Y+Z", 25: "Move -X+Y-Z", 26: "Move -X-Y+Z", 27: "Move -X-Y-Z",
    28: "No-Op",
}

gripper_threshold = 50
movement_threshold = 0.01
gripper_open_signal = 0.
gripper_close_signal = 100
step_size = movement_threshold

# The Map: Index -> (dx, dy, dz, gripper_signal)
discrete_index_to_vector_map = {
    # Gripper Actions
    ACTION_OPEN_GRIPPER:  (0.0, 0.0, 0.0, gripper_open_signal),
    ACTION_CLOSE_GRIPPER: (0.0, 0.0, 0.0, gripper_close_signal),

    # Single Axis Movement
    ACTION_MOVE_PX: ( step_size,  0.0,  0.0, 0.0),
    ACTION_MOVE_NX: (-step_size,  0.0,  0.0, 0.0),
    ACTION_MOVE_PY: ( 0.0,  step_size,  0.0, 0.0),
    ACTION_MOVE_NY: ( 0.0, -step_size,  0.0, 0.0),
    ACTION_MOVE_PZ: ( 0.0,  0.0,  step_size, 0.0),
    ACTION_MOVE_NZ: ( 0.0,  0.0, -step_size, 0.0),

    # Two Axis Movement (XY)
    ACTION_MOVE_PX_PY: ( step_size,  step_size,  0.0, 0.0),
    ACTION_MOVE_PX_NY: ( step_size, -step_size,  0.0, 0.0),
    ACTION_MOVE_NX_PY: (-step_size,  step_size,  0.0, 0.0),
    ACTION_MOVE_NX_NY: (-step_size, -step_size,  0.0, 0.0),
    # Two Axis Movement (XZ)
    ACTION_MOVE_PX_PZ: ( step_size,  0.0,  step_size, 0.0),
    ACTION_MOVE_PX_NZ: ( step_size,  0.0, -step_size, 0.0),
    ACTION_MOVE_NX_PZ: (-step_size,  0.0,  step_size, 0.0),
    ACTION_MOVE_NX_NZ: (-step_size,  0.0, -step_size, 0.0),
    # Two Axis Movement (YZ)
    ACTION_MOVE_PY_PZ: ( 0.0,  step_size,  step_size, 0.0),
    ACTION_MOVE_PY_NZ: ( 0.0,  step_size, -step_size, 0.0),
    ACTION_MOVE_NY_PZ: ( 0.0, -step_size,  step_size, 0.0),
    ACTION_MOVE_NY_NZ: ( 0.0, -step_size, -step_size, 0.0),

    # Three Axis Movement
    ACTION_MOVE_PX_PY_PZ: ( step_size,  step_size,  step_size, 0.0),
    ACTION_MOVE_PX_PY_NZ: ( step_size,  step_size, -step_size, 0.0),
    ACTION_MOVE_PX_NY_PZ: ( step_size, -step_size,  step_size, 0.0),
    ACTION_MOVE_PX_NY_NZ: ( step_size, -step_size, -step_size, 0.0),
    ACTION_MOVE_NX_PY_PZ: (-step_size,  step_size,  step_size, 0.0),
    ACTION_MOVE_NX_PY_NZ: (-step_size,  step_size, -step_size, 0.0),
    ACTION_MOVE_NX_NY_PZ: (-step_size, -step_size,  step_size, 0.0),
    ACTION_MOVE_NX_NY_NZ: (-step_size, -step_size, -step_size, 0.0),

    # No Operation
    ACTION_NO_OP: (0.0, 0.0, 0.0, 0.0)
}

class KinovaActions():
    def __init__(self, max_demos=float('inf')):
        self.use_eef = True
    
    def next_action(self, normalize=False, diff_eef=True):
        ACTION_MOVE_PX