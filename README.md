# Welcome to the IC CAE UGRA RFRL GYM Python Package!

<img width="523" alt="logoblack" src="https://github.com/vtnsiSDD/rfrl-gym/assets/15094176/2da9506f-8a0e-46d2-9d07-c83f0858cc19">


Note: Pardon our mess as this project is under active development. Please let us know of any feature requests or bugs to be squashed!

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
