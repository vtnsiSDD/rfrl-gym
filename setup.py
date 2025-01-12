from setuptools import setup, find_packages

setup(name='rfrl-gym',
      version='0.1',
      author='Virginia Tech National Security Institute and Morehouse College',
      install_requires=['gymnasium',
                        'numpy==1.26.4',
                        'matplotlib',
                        'argparse',
                        'distinctipy',
                        'pyqtgraph',
                        'pyqt6',
                        'scipy',
                        'dm_tree',
                        'pyarrow',
                        'pandas',
                        'torch'],
      extras_require={'rl_packages': ['stable_baselines3']},
      packages= find_packages())