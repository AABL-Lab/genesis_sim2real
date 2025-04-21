from gymnasium.envs.registration import register

register(
    id="genesis_lift_SB-v0",
    entry_point="genesis_sim2real.envs.genesis_gym:GenesisGym",
    max_episode_steps=800,
    kwargs={
        "env_name": "lift",
        "use_truncated_in_return": True,
        'debug': False,
        'stable_baselines': True,
        'max_steps': 800,
    },
)
register(
    id="genesis_lift-v0",
    entry_point="genesis_sim2real.envs.genesis_gym:GenesisGym",
    max_episode_steps=800,
    kwargs={
        "env_name": "lift",
        "use_truncated_in_return": False,
        'debug': False,
        'stable_baselines': False,
        'max_steps': 800,
    },
)