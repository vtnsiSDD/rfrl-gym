import numpy as np
from gymnasium.envs.registration import register

register(
    id='rfrl-gym-abstract-v0',
    entry_point='rfrl_gym.envs:RFRLGymAbstractEnv',
    max_episode_steps = 1000
)

register(
    id='rfrl-gym-iq-v0',
    entry_point='rfrl_gym.envs:RFRLGymIQEnv',
    max_episode_steps = 1000
)