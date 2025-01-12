import numpy as np
from gymnasium.envs.registration import register

register(
    id='rfrl-gym-abstract-v0',
    entry_point='rfrl_gym.envs:RFRLGymAbstractEnv',
    max_episode_steps = int(1e5)
)

register(
    id='rfrl-gym-iq-v0',
    entry_point='rfrl_gym.envs:RFRLGymIQEnv',
    max_episode_steps = int(1e5)
)

# TODO: Create a way for the user to pass this information in
# so that it is not hardcoded.
repo_root_directory = "/Users/sri/Desktop/rfrl-gym/"
