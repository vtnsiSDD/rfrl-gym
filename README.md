# Welcome to the IC CAE UGRA RFRL GYM Python Package!

<img width="523" alt="logo" src="https://github.com/wcheadley/rfrl-gym/assets/15094176/f683967f-8d5a-49f5-ba99-819cd2d50a47">

## To install the codebase:

### Linux 
*Note: A Ubuntu distribution is strongly recommended when working on this project in Linux.*

1. Install necessary prerequist software using the terminal:

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

`python3 get-pip.py`

`del get-pip.py`

3. Install Python Virtual Environment for Python 3:

`pip install virtualenv`

4. Set up a Python virtual environment in the root directory of the repository:

`cd C:\Users\[PATHTOREPOSITORY]`

`python -m venv rfrl-gym-venv`

5. Ensure that venv is fully updated:

`python3 -m venv --upgrade rfrl-gym-venv`

6. Activate the virtual environment (you will need to do this everytime you being working with the repository in a new terminal):

`Set-ExecutionPolicy Unrestricted -Scope Process`

`rfrl-gym-venv/Scripts/activate.ps1`

7. Install the repository:

`pip install --editable .`

### Mac OS
1. Install Python 3 from: https://www.python.org/downloads/

2. Install Pip for Python 3:

`curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py`

`python3 get-pip.py`

3. Install Python Virtual Environment for Python 3:

`pip3 install virtualenv`

4. Set up a Python virtual environment in the root directory of the repository:

`python3 -m venv rfrl-gym-venv`

5. Ensure that venv is fully updated:

`python3 -m venv --upgrade rfrl-gym-venv`

6. Activate the virtual environment (you will need to do this everytime you being working with the repository in a new terminal):

`source rfrl-gym-venv/bin/activate`

7. Install setuptools:

`pip3 install setuptools`

8. Install the repository:

`pip3 install --editable .`

## To test installation of the codebase:
`python3 scripts/gym_test.py`
  
A terminal output should print out showing the observation space upon successful execution. 

## To run with the mushroom-rl package:
`pip3 install -e ".[rl_packages]"`

This will instance the packages within "extras_require" entry in setup.py
