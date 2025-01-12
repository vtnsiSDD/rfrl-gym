"""
Multi-agent APPO training script.
"""

import argparse
import os
import ray
import time

from ray.tune import register_env
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.appo.appo import APPOConfig

from rfrl_gym.envs.ma_rfrl_gym_abstract_env import RFRLGymMultiAgentAbstractEnv
from plot_generator import plot_rewards

parser = argparse.ArgumentParser()
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--mixer",
    type=str,
    default="qmix",
    choices=["qmix", "vdn", "none"],
    help="The mixer model to use.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument( # Not commonly used
    "--stop-iters", type=int, default=1e9, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-episodes", type=int, default=600, help="Number of episodes to train."
)
parser.add_argument( # Not commonly used
    "--stop-timesteps", type=int, default=1e9, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=250.0, help="Reward at which we stop training."
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
    '--checkpoint_name',
    default=f'checkpoint{time.time_ns()}',
    type=str,
    help='The name of the folder that the checkpoint will be saved into.'
)

if __name__ == "__main__":
    # Parses command-line arguments
    args = parser.parse_args()

    # Initializes the Ray cluster
    ray.init(local_mode=args.local_mode)

    # Initializes the environment and registers another
    rfrl_gym = RFRLGymMultiAgentAbstractEnv(scenario_filename=args.scenario)
    register_env(
        name="rfrl_gym_multi_agent_env",
        env_creator=lambda env_creator : RFRLGymMultiAgentAbstractEnv(scenario_filename=args.scenario)
    )
    
    # Constructs the config used for building the algorithm
    config = (
        APPOConfig()
        .framework(args.framework)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )
    config = config.framework("torch")
    config = config.rollouts(num_rollout_workers=0, rollout_fragment_length=4)
    config = config.environment(
        env="rfrl_gym_multi_agent_env",
        env_config={
            "separate_state_space": True,
            "one_hot_state_encoding": True,
        }
    )

    # Gets a policy ID from an agent ID
    def policy_mapper(agent_id:str, _episode=None, _worker=None, **_kwargs):
        return f"policy_for_{agent_id}"
    
    # Makes the different policies
    policies = {}
    for agent_id in rfrl_gym._agent_ids:
        policies[policy_mapper(agent_id)] = PolicySpec(
            policy_class=None,
            observation_space=rfrl_gym.observation_space[agent_id],
            action_space=rfrl_gym.action_space[agent_id],
            config = {}
        )
        
    # Maps from each agent to its own policy in the config
    config = config.multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapper
    )

    # Defines stopping conditions
    stop = {
        "episode_reward_mean": args.stop_reward,
        "timesteps_total": args.stop_timesteps,
        "episodes_total": args.stop_episodes,
        "training_iteration": args.stop_iters,
    }

    # Repeatedly trains the algorithm
    algo = config.build()
    episodes_total = 0
    all_episode_rewards = []
    start_time = time.time()
    for i in range(1000000):
        # Trains the algo
        print(f"================ Training iteration {i + 1} ================")
        results = algo.train()

        # Parses results
        episode_reward = results["env_runners"]["hist_stats"]["episode_reward"]
        episodes_this_iter = results["env_runners"]["episodes_this_iter"]
        episodes_total += episodes_this_iter

        # Attaches all episode rewards from this iteration to the main list
        episodes_this_iter *= -1
        rewards_from_this_iter = episode_reward[episodes_this_iter:] # should never be out of range
        all_episode_rewards.extend(rewards_from_this_iter)
        episodes_this_iter *= -1
        print(f"Episodes this iteration = {episodes_this_iter}")
        print(f"Average episode reward during this iteration = {float(sum(rewards_from_this_iter))/(episodes_this_iter)}")
        print(f"Total episodes so far = {episodes_total}\n")

        # Checks if it's time to stop training
        if episodes_total >= stop["episodes_total"]:
            break
    end_time = time.time()
    print(f"Finished training the {type(algo)} algo for {episodes_total} episodes.")
    print(f"Scenario run: {args.scenario}")
    print(f"Total training time: {end_time - start_time}")
    # print(f"\nAll episode rewards: {all_episode_rewards}\n")

    # Makes a plot of the rewards
    algo_id = args.checkpoint_name
    plot_filename = f"marl_checkpoints/{algo_id}.png"
    if os.path.exists(plot_filename):
        os.remove(plot_filename)
    plot_rewards(scenario=args.scenario, episode_rewards={str(algo): all_episode_rewards},
                 smoothing=None, filename=plot_filename)
    
    # Saves the trained algorithm
    checkpoint_dir = f"marl_checkpoints/{algo_id}"
    if os.path.exists(checkpoint_dir):
        import shutil
        shutil.rmtree(checkpoint_dir)
    save_result = algo.save(checkpoint_dir)
    path_to_checkpoint = save_result.checkpoint.path
    print(
        f"An {algo} checkpoint has been created inside the directory: "
        f"'{path_to_checkpoint}'."
    )
    
    # Checks the results
    if args.as_test:
        episode_reward_mean = results['env_runners']['episode_reward_mean'] # from the last 100 episodes
        if episode_reward_mean >= args.stop_reward:
            print(f"episode_reward_mean ({episode_reward_mean}) was greater than/equal to args.stop_reward ({args.stop_reward})")
        else:
            print(f"episode_reward_mean ({episode_reward_mean}) did not reach args.stop_reward ({args.stop_reward})")

    # Terminates all Ray processes
    ray.shutdown()
