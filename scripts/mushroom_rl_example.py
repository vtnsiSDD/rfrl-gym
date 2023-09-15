import argparse
from mushroom_rl.core import Core
from mushroom_rl.environments import Gym
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.algorithms.value import QLearning
from mushroom_rl.utils.parameters import ExponentialParameter, Parameter
import rfrl_gym

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--scenario', default='fixed_test.json', type=str, help='The scenario file to run in the RFRL gym environment.')
args = parser.parse_args()

# Set reinforcement learning parameters.
gamma = 0.99
epsilon_start = 1.0
epsilon_min = 0.0
epsilon_exponent = 0.1
learning_rate = 0.001
num_learning_episodes = 1000

# Initialize the gym environment (note: the horizon is handled by the scenario file for the RFRL gym environment).
mdp = Gym('rfrl-gym-v0', scenario_filename=args.scenario, num_episodes=num_learning_episodes+1, horizon=None, gamma=gamma)

# Initialize the reinforcement learning agent.
epsilon = ExponentialParameter(epsilon_start, exp=epsilon_exponent, min_value=epsilon_min)
policy = EpsGreedy(epsilon)
agent = QLearning(mdp.info, policy, Parameter(learning_rate))
    
# Train and evaluate the reinforcement learning agent.
core = Core(agent, mdp)
core.learn(n_episodes=num_learning_episodes, n_steps_per_fit=1, render=False, quiet=False)
    
agent.policy.set_epsilon(0.0)
_, info = core.evaluate(n_episodes=1, render=True, quiet=True, get_env_info=True)

input('Press any key to end the simulation...')