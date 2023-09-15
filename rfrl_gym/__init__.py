import numpy as np
from gym.envs.registration import register

register(
    id='rfrl-gym-v0',
    entry_point='rfrl_gym.envs:RFRLGymEnv',
    max_episode_steps = np.inf
)
