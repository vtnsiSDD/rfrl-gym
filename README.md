# Welcome to the IC CAE UGRA RFRL GYM Python Package!

<img width="523" alt="logo" src="https://github.com/wcheadley/rfrl-gym/assets/15094176/f683967f-8d5a-49f5-ba99-819cd2d50a47">

## To install the codebase:

### Linux 
*Note: A Ubuntu distribution is strongly recommended when working on this project in Linux.*

1. Install necessary prerequisite software using the terminal:

`sudo apt install python3 python3-pip python3-venv python3-wheel`

2. Set up a Python virtual environment in the root directory of the repository:

`python3 -m venv rfrl-gym-venv`

3. Ensure that venv is fully updated:

`python3 -m venv --upgrade rfrl-gym-venv` 

4. Activate the virtual environment (you will need to do this everytime you being working with the repository in a new terminal):

`source rfrl-gym-venv/bin/activate`

5. Install setuptools:

`pip3 install setuptools`

6. Install the repository:

`pip3 install --editable .`

### Windows
*Note: Visual Studio Code is strongly recommended when working on this project in Windows.*

1. Install Python 3.10 from: https://www.python.org/downloads/ (if given the option, chose to add Python commands to PATH)

2. Install Pip for Python 3: 

`curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py`

`python3 get-pip.py` or `py get-pip.py`

`del get-pip.py`

Note: If pip is not being recognized, you may have to add it to your PATH: setx PATH "%PATH%;C:path_to_pip". Then close out of the terminal and reopen.

3. Set up a Python virtual environment in the root directory of the repository:

`cd C:\Users\[PATHTOREPOSITORY]`

`python3 -m venv rfrl-gym-venv` or `py -m venv rfrl-gym-venv`

4. Ensure that venv is fully updated:

`python3 -m venv --upgrade rfrl-gym-venv` or `py -m venv --upgrade rfrl-gym-venv`

5. Activate the virtual environment (you will need to do this everytime you being working with the repository in a new terminal):

`Set-ExecutionPolicy Unrestricted -Scope Process`

`rfrl-gym-venv/Scripts/activate.ps1`

6. Install necessary upgrades and build dependencies to the virtual environment:

`pip install wheel==0.38.4`

7. Install the repository:

`pip install --editable .`

### macOS
1. Install Python 3 from: https://www.python.org/downloads/

2. Install Pip for Python 3:

`curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py`

`python3 get-pip.py`

3. Clone the repository:

`git clone git@github.com:wcheadley/rfrl-gym.git`

The command above uses SSH, but any method will work.

4. Change directory to the repo folder:

`cd rfrl-gym`

5. Set up a Python virtual environment in the root directory of the repository:

`python3 -m venv rfrl-gym-venv`

6. Ensure that venv is fully updated:

`python3 -m venv --upgrade rfrl-gym-venv`

7. Activate the virtual environment:

`source rfrl-gym-venv/bin/activate`

You will need to do this every time you start working with the repository in a new terminal.

8. Define the path variable in order to access the installed packages:

`export PYTHONPATH="ROOT_DIRECTORY/rfrl-gym-venv/lib/PYTHON_VERSION/site-packages"`

Replace `ROOT_DIRECTORY` with the path to the folder containing all the content of this cloned repository, and replace `PYTHON_VERSION` with the version of Python you currently have installed (e.g. `python3.9` for Python 3.9).

This will also need to be done every time you work with the repository in a new terminal window. Alternatively, to avoid this, you could add this command to your local `.bash_profile` or `.zshrc` file, based on which shell you are using.

9. Ensure that pip is upgraded:

`pip3 install --upgrade pip`

10. Make sure that you have the correct version of wheel:

`pip3 install wheel==0.38.4`

11. Install setuptools:

`pip3 install setuptools`

12. Install the repository:

`pip3 install .`

## For testing the single-agent environment:

### To test installation of the codebase:
`python3 scripts/preview_scenario.py`
  
A terminal output should print out showing the observation space upon successful execution. 

### To run with the mushroom-rl package:
`pip3 install -e ".[rl_packages]"`

### To test installation of the mushroom-rl package:
`python3 scripts/mushroom_rl_example.py`

This will instance the packages within "extras_require" entry in setup.py

## For testing the MARL environment:

### To run with the MARL scripts:
`pip3 install "ray[rllib]"==2.31.0`

### To test installation of the MARL extension:
`python3 scripts/marl_preview_scenario.py`

### To train agents:
`python3 scripts/marl_dqn_training.py --checkpoint_name my_saved_checkpoint`

### To load and test the policies of trained agents:
`python3 scripts/marl_dqn_testing.py my_saved_checkpoint`

In general, it's recommended to ensure that the scenario which was used for training a set of agents is the same scenario used when loading the saved policies and evaluating them. This can be done by specifying the `--scenario` argument to both scripts.
