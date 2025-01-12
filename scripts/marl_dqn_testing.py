"""
Multi-agent DQN testing script.
"""

import os
import sys
#sys.stderr = open(os.devnull, 'w')

import argparse
import pickle
import ray
import time
import torch
import numpy as np

from ray.tune import register_env
from ray.rllib.policy.policy import Policy
from ray.rllib.models.modelv2 import flatten

from rfrl_gym.envs.ma_rfrl_gym_abstract_env import RFRLGymMultiAgentAbstractEnv

parser = argparse.ArgumentParser()
parser.add_argument(
    'checkpoint_name',
    type=str,
    help='The name of the folder that the checkpoint will be saved into.'
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--goal-reward",
    type=float,
    help="The intended reward for this episode. This must be provided when using --as-test."
)
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)
parser.add_argument(
    '-s', '--scenario',
    default='test_scenario_marl_3.json',
    type=str,
    help='The scenario file to be ran with the RFRL Gym\'s MultiAgentEnv.'
)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)

if __name__ == "__main__":
    # Parses command-line arguments
    args = parser.parse_args()

    # Initializes the Ray cluster
    ray.init(local_mode=args.local_mode)
    
    # Registers the environment and also makes another one to test with
    register_env(
        name="rfrl_gym_multi_agent_env",
        env_creator=lambda env_creator : RFRLGymMultiAgentAbstractEnv(scenario_filename=args.scenario)
    )
    env = RFRLGymMultiAgentAbstractEnv(scenario_filename=args.scenario)

    # Gets a policy ID from an agent ID
    def policy_mapper(agent_id:str, _episode=None, _worker=None, **_kwargs):
        return f"policy_for_{agent_id}"
    
    # Loads the policies
    loaded_policies = {}
    algo_id = args.checkpoint_name
    for agent_id in env.agents_info:
        policy_state_path = os.path.join("marl_checkpoints", algo_id, "policies",
                                         policy_mapper(agent_id), "policy_state.pkl")
        with open(policy_state_path, "rb") as f:
            policy_state = pickle.load(f)
            loaded_policies[agent_id] = Policy.from_state(policy_state)

    # Uses the saved policies for inference
    obs, info = env.reset()
    env.render()
    time_start = time.time()
    done  = {'__all__': False}
    episode_total_reward = 0
    while not done['__all__']:
        action_dict = {}
        for agent_id in env._agent_ids:
            agent_obs_mode = env.agents_info[agent_id]['observation_mode']
            assert agent_obs_mode in ('classify', 'detect')
            if agent_obs_mode == 'classify':
                observations_per_channel = env.num_entities + env._num_agents + 2
            else:
                observations_per_channel = 2
            one_hot = np.eye(observations_per_channel)[obs[agent_id]]
            flattened = flatten(torch.Tensor(one_hot), args.framework)
            action_dict[agent_id] = loaded_policies[agent_id].compute_single_action(obs=flattened)[0]

        obs, rewards, terminated, truncated, info = env.step(action_dict)
        if(terminated['__all__'] and truncated['__all__']):
            done['__all__'] = True
        episode_total_reward += sum(list(rewards.values()))
        
        env.render()
    env.close()

    # Evaluates the result
    if args.as_test:
        assert args.goal_reward is not None, "Must pass a goal reward (with --goal-reward) when testing the saved policy (with --as-test)."
        if episode_total_reward >= args.goal_reward:
            print(f"Total episode reward ({episode_total_reward}) was greater than/equal to goal reward ({args.goal_reward})!")
        else:
            print(f"Total episode reward ({episode_total_reward}) did not reach goal reward ({args.goal_reward}).")
