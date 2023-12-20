# Welcome to the RFRL GYM Python Package!

<p align="center">
  <img width="500" src="https://github.com/vtnsiSDD/rfrl-gym/assets/15094176/2da9506f-8a0e-46d2-9d07-c83f0858cc19" />
</p>

_Note: Pardon our mess as this project is under active development. Please let us know of any feature requests or bugs to be squashed!_

## To install the codebase:

### Linux (verified in Ubuntu)

1. Install necessary prerequist software using the terminal:

`sudo apt install python3 python3-pip python3-venv python3-wheel`

2. Set up a Python virtual environment in the root directory of the repository:

`python3 -m venv rfrl-gym-venv`

3. Ensure that venv is fully updated:

`python3 -m venv --upgrade rfrl-gym-venv`

4. Activate the virtual environment (you will need to do this everytime you being working with the repository in a new terminal):

`source rfrl-gym-venv/bin/activate`

5. Install setuptools:

`pip3 install pip wheel setuptools --upgrade`

6. Install the repository:

`pip3 install --editable .`

## To test installation of the codebase and the renderer:
`python3 scripts/preview_scenario.py -m abstract`

`python3 scripts/preview_scenario.py -m iq`
  
A terminal output should print out showing the observation space upon successful execution. 

## To install with the stable_baselines3 package:
`pip3 install -e ".[rl_packages]"`

## To test installation of the stable_baselines3 package:
`python3 scripts/sb3_example.py -m abstract`

`python3 scripts/sb3_preview_scenario.py -m abstract`

## How to reference

Please use the following bibtex:
```
@inproceedings{rfrlgym,
  Title = {{RFRL Gym: A Reinforcement Learning Testbed for Cognitive Radio Applications}},
  Author = {D. Rosen, I. Rochez, C. McIrvin, J. Lee, K. D’Alessandro, M. Wiecek, N. Hoang, R. Saffarini, S. Philips, V. Jones, W. Ivey, Z. Harris-Smart, Z. Harris-Smart, Z. Chin, A. Johnson, A. Jones, W. C. Headley},
  Booktitle = {{IEEE International Conference on Machine Learning and Applications (ICMLA)}},
  Year = {2023},
  Location = {Jacksonville, USA},
  Month = {December},
  Url = {}
```
