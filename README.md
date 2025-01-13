<img align="right" width="400" src="https://github.com/vtnsiSDD/rfrl-gym/assets/15094176/2da9506f-8a0e-46d2-9d07-c83f0858cc19"> 

# Welcome to the RFRL GYM Python Package!
The RFRL Gym is intended as a training and research environment for wireless communications applications designed to provide comprehensive functionality, such as custom scenario generation, multiple learning settings, and compatibility with third-party RL packages. Additionally, through a gamified mode of the RF spectrum, this tool can be used to teach novices about the fields of AI/ML and RF.

 Jamming Agent before Learning            |  Jamming Agent after Learning
:----------------------------------------:|:-----------------------------:
![initial](https://github.com/vtnsiSDD/rfrl-gym/assets/15094176/aebf248f-b71b-4692-a35f-79091a6e8371) | ![learned](https://github.com/vtnsiSDD/rfrl-gym/assets/15094176/452fefff-0c9d-4d1e-91ac-d722985421ac)

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

## MARL Extension:
To use the MARL extension of the RFRL Gym, please refer to the `marl-dev` branch (https://github.com/vtnsi/rfrl-gym/tree/marl-dev).

## How to reference the single-agent RFRL GYM:
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

## How to reference the multi-agent RFRL GYM:
```
@inproceedings{marl-rfrlgym,
  Title = {{A Multi-Agent Reinforcement Learning Testbed for Cognitive Radio Applications}},
  Author = {S. Vangaru, D. Rosen, D. Green, R. Rodriguez, M. Wiecek, Johnson, A. M. Jones, W. C. Headley},
  Booktitle = {{IEEE Consumer Communications & Networking Conference}},
  Year = {2025},
  Location = {Las Vegas, USA},
  Month = {January},
  Url = {}
```
