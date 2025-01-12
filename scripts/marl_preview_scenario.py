import gymnasium as gym
import time
import argparse
import numpy as np
from rfrl_gym.envs.ma_rfrl_gym_abstract_env import RFRLGymMultiAgentAbstractEnv

# Parses arguments
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--scenario', default='test_scenario_marl_3.json', type=str, help='The scenario file to be ran with the RFRL Gym\'s MultiAgentEnv.')
parser.add_argument('--no_actions', action='store_true', help='Mark as true if you don\'t want the agents to transmit any signals')
args = parser.parse_args()

# Sets up environment
env = RFRLGymMultiAgentAbstractEnv(scenario_filename=args.scenario)
reset_obs, reset_info = env.reset()

# Steps through a full episode of the scenario
time_start = time.time()
done  = {'__all__': False}
env.render()
while not done['__all__']:
    action_dict = {}
    for agent_id in env.agents_info:
        if args.no_actions:
            action = 0
        else:
            action = np.random.randint(0, env.num_channels + 1)
        action_dict[agent_id] = action
    
    obs, rewards, terminated, truncated, info = env.step(action_dict)
    if(terminated['__all__'] and truncated['__all__']):
        done['__all__'] = True

    env.render()
print('Execution Time: ', time.time()-time_start)
env.close()
