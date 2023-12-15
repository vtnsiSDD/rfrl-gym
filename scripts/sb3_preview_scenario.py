import gymnasium as gym
import argparse
import matplotlib.pyplot as plt
import rfrl_gym
import time

from stable_baselines3 import DQN

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--scenario', default='sb3_test_scenario.json', type=str, help='The scenario file to preview in the RFRL gym environment.')
parser.add_argument('-m', '--gym_mode', default='abstract', type=str, help='Which type of RFRL gym environment to run.')
parser.add_argument('-e', '--epochs', default=100, type=int, help='Number of training epochs.')
args = parser.parse_args()

if args.gym_mode == 'abstract':
    env = gym.make('rfrl-gym-abstract-v0', scenario_filename=args.scenario)
elif args.gym_mode == 'iq':
    env = gym.make('rfrl-gym-iq-v0', scenario_filename=args.scenario)

model = DQN("MlpPolicy", env, verbose=1, exploration_initial_eps=1.0, exploration_final_eps=0.001,exploration_fraction=0.995)

obs, info = env.reset()
terminated = truncated= False
while not terminated and not truncated:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

del model

time.sleep(5)

model = DQN.load("rfrl_gym_dqn")

obs, info = env.reset()
terminated = truncated= False
while not terminated and not truncated:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()