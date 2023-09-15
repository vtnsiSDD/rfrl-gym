import gym
import argparse
import rfrl_gym

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--scenario', default='fixed_test.json', type=str, help='The scenario file to preview in the RFRL gym environment.')
args = parser.parse_args()

env = gym.make('rfrl-gym-v0', scenario_filename=args.scenario)
env.reset()

done = 0
while done == 0:
    env.render('human')
    observation, reward, done, info = env.step(0)
env.render('human')
input('Press any key to end the simulation...')
env.close()