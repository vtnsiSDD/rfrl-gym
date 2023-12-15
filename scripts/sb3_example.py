import gymnasium as gym
import argparse
import matplotlib.pyplot as plt
import rfrl_gym

from stable_baselines3 import DQN

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--scenario', default='sb3_test_scenario.json', type=str, help='The scenario file to preview in the RFRL gym environment.')
parser.add_argument('-m', '--gym_mode', default='abstract', type=str, help='Which type of RFRL gym environment to run.')
parser.add_argument('-e', '--epochs', default=1000, type=int, help='Number of training epochs.')
args = parser.parse_args()

if args.gym_mode == 'abstract':
    env = gym.make('rfrl-gym-abstract-v0', scenario_filename=args.scenario)
elif args.gym_mode == 'iq':
    env = gym.make('rfrl-gym-iq-v0', scenario_filename=args.scenario)
env.reset()

model = DQN("MlpPolicy", env, verbose=1, exploration_initial_eps=1.0, exploration_final_eps=0.001,exploration_fraction=0.995)
# for _ in range(args.epochs):
model = model.learn(total_timesteps=env.max_steps * args.epochs,log_interval=100, progress_bar=True)
env.reset()
model.save("rfrl_gym_dqn")

del model # remove to demonstrate saving and loading

model = DQN.load("rfrl_gym_dqn")


obs, info = env.reset()
terminated = truncated= False
running_reward = 0
rewards = []

while not terminated and not truncated:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    #env.render()