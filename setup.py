from setuptools import setup, find_packages

setup(name='rfrl-gym',
      version='0.1',
      author='Virginia Tech National Security Institute and Morehouse College',
      install_requires=['gym==0.21.0',
                        'numpy',
                        'matplotlib',
                        'argparse',
                        'distinctipy',
                        'pyqtgraph',
                        'pyqt6'],
      extras_require={'rl_packages': ['mushroom_rl', 'stable_baselines3']},
      packages= find_packages())