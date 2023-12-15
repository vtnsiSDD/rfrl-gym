import numpy as np
from gymnasium.envs.registration import register

register(
    id='rfrl-gym-abstract-v0',
    entry_point='rfrl_gym.envs:RFRLGymAbstractEnv',
    max_episode_steps = np.inf
)

register(
    id='rfrl-gym-iq-v0',
    entry_point='rfrl_gym.envs:RFRLGymIQEnv',
    max_episode_steps = np.inf
)