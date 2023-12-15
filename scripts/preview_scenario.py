import gymnasium as gym
import time
import argparse
import rfrl_gym

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--scenario', default='test_scenario.json', type=str, help='The scenario file to preview in the RFRL gym environment.')
parser.add_argument('-m', '--gym_mode', default='abstract', type=str, help='Which type of RFRL gym environment to run.')
args = parser.parse_args()

if args.gym_mode == 'abstract':
    env = gym.make('rfrl-gym-abstract-v0', scenario_filename=args.scenario)
elif args.gym_mode == 'iq':
    env = gym.make('rfrl-gym-iq-v0', scenario_filename=args.scenario)
env.reset()

done = truncated = False
time_start = time.time()
while not done and not truncated:
    env.render()
    observation, reward, done, truncated, info = env.step(0)

env.render()
print('Execution Time: ', time.time()-time_start)
env.close()
