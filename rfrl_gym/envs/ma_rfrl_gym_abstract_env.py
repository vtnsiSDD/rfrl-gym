import gymnasium as gym
import ray
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from rfrl_gym.envs import RFRLGymAbstractEnv
import json
from PyQt6.QtWidgets import QApplication
import numpy as np
import rfrl_gym.renderers
import rfrl_gym.entities
import time
from rfrl_gym import repo_root_directory
from ray.util import log_once
from ray.rllib.utils.annotations import override
import os

class RFRLGymMultiAgentAbstractEnv(MultiAgentEnv):
    metadata = {'render_modes': ['null', 'terminal', 'pyqt'],
                    'reward_modes': ['dsa', 'jam'],
                    'observation_modes': ['detect', 'classify']}

    def __init__(self, scenario_filename, num_episodes=1):
        super().__init__()

        self.num_episodes = num_episodes

        # Load in the JSON scenario file and check for valid entries.
        f_idx = open('scenarios/' + scenario_filename)
        self.scenario_metadata = json.load(f_idx)
        self.__validate_scenario_metadata()
        print(f"Validated scenario metadata.")
        
        # Get the environment parameters from the scenario file.
        self.num_channels = self.scenario_metadata['environment']['num_channels']
        self.max_steps = self.scenario_metadata['environment']['max_steps']
        for agent_id in self.scenario_metadata['environment']['agents']:
            self._agent_ids.add(agent_id)
        self.agents_info = self.scenario_metadata['environment']['agents'] # observation mode, reward mode and target entity *per agent*
        self._num_agents = len(self._agent_ids)

        # Get the entity parameters from the scenario file and initialize the entities.
        entity_idx = 0
        self.entity_list = []
        self.target_idx = {} # Maps from each agent's ID to their target entity index
        for agent_id in self._agent_ids: self.target_idx[agent_id] = None
        for entity in self.scenario_metadata['entities']:  
            entity_idx += 1
            obj_str = f"rfrl_gym.entities.{self.scenario_metadata['entities'][entity]['type']}(entity_label=\'{str(entity)}\', num_channels={str(self.num_channels)}, "
            for param in self.scenario_metadata['entities'][entity]:
                if not param == 'type':
                    obj_str += (param + '=' + str(self.scenario_metadata['entities'][entity][param]) + ', ')
            obj_str += ')'
            self.entity_list.append(eval(obj_str))      
            self.entity_list[-1].set_entity_index(entity_idx)
            for agent_id in self._agent_ids:
                if entity == self.agents_info[agent_id]['target_entity']:
                    self.target_idx[agent_id] = entity_idx
        self.num_entities = len(self.entity_list)

        # Creates a map from agent ID to index
        # Entities range from 1 to num_entities (inclusive), and agents range
        # from (num_entities+1) to (num_entities+num_agents). A collision is
        # represented by (num_entities + num_agents + 1), and a free channel
        # is represented by 0.
        self.agent_id_to_num = {}
        self.num_to_agent_id = {}
        index = self.num_entities
        for agent_id in self._agent_ids:
            index += 1
            self.agent_id_to_num[agent_id] = index
            self.num_to_agent_id[index] = agent_id

        # Get the render parameters from the scenario file and initialize the render if necessary.
        self.render_mode = self.scenario_metadata['render']['render_mode']
        self.render_fps = self.metadata['render_fps'] = self.scenario_metadata['render']['render_fps']
        if self.render_mode == 'pyqt':
            self.pyqt_app = QApplication([])

        # Creates a mapping from each agent's ID to its action space and observation space
        # May change calculations later if incorrect
        # 
        # In gymnasium's core.py, it's stated in the Env class that "action_space" and "observation_space"
        # need to be defined in all subclasses and are of type spaces.Space, and the Space class says:
        # "Different spaces can be combined hierarchically via container spaces (:class:`Tuple`
        # and :class:`Dict`) to build a more expressive space"
        # so this should be fine for MARL.
        self.action_space = {}
        self.observation_base = {}
        self.observation_space = {}
        for agent_id in self._agent_ids:
            # 1 choice is to not transmit, and the remaining num_channels choices are
            # to transmit on any 1 of the num_channels channels
            self.action_space[agent_id] = gym.spaces.Discrete(1+self.num_channels)

            if self.agents_info[agent_id]['observation_mode'] == 'detect':
                # For each channel: either something is there or nothing is there
                self.observation_base[agent_id] = 2
            
            elif self.agents_info[agent_id]['observation_mode'] == 'classify':
                # For each channel: either it contains nothing (1 case), a collision (1 case),
                # a non-learning entity (resulting in num_entities total cases),
                # or one of the learning agents (including itself, causing len(_agent_ids) total cases)
                self.observation_base[agent_id] = 1 + self.num_entities + len(self._agent_ids) + 1

            # the numpy array representing the shape of the MultiDiscrete space (per agent)
            # is of length num_channels, and for each channel, there are observation_base[agent_id]
            # possible observations
            # then, the MultiDiscrete space takes this shape and constructs a gym (gymnasium) Space
            obs_space = self.observation_base[agent_id] + np.zeros(self.num_channels, dtype=int)
            self.observation_space[agent_id] = gym.spaces.MultiDiscrete(nvec=obs_space)
        self.action_space = gym.spaces.Dict(self.action_space)
        self.observation_space = gym.spaces.Dict(self.observation_space)

    # Returns new observations and "auxiliary diagnostic information" for each agent
    # (from RLlib and gymnasium documentation, found at
    # https://docs.ray.io/en/latest/_modules/ray/rllib/env/multi_agent_env.html
    # and https://gymnasium.farama.org/api/env/)
    def reset(self, *, seed=None, options=None):
        # "For Custom environments, the first line of reset should be super().reset(seed=seed) which
        # implements the seeding correctly."
        super().reset(seed=seed)

        # Temporarily store episode specific variables if they exist.
        if hasattr(self, 'info') and options and options['reset_type'] == 'soft':
            episode_number = self.info['episode_number']
            episode_reward = self.info['episode_reward'] 
        else:
            episode_number = -1
            episode_reward = {}
            for agent_id in self._agent_ids:
                episode_reward[agent_id] = np.array([], dtype=float)
            if self.render_mode == 'terminal':
                self.renderer = rfrl_gym.renderers.ma_terminal_renderer.MultiAgentTerminalRenderer(self.num_episodes, self.scenario_metadata)
            if self.render_mode == 'pyqt':
                self.renderer = rfrl_gym.renderers.ma_pyqt_renderer.MultiAgentPyQtRenderer(self.num_episodes, self.scenario_metadata, mode='abstract')
            if self.render_mode != 'null':
                self.renderer.reset()

        # Reset the gym info dictionary and if necessary restore episode variables.
        self.info = {}
        self.info['step_number'] = 0
        self.info['num_entities'] = self.num_entities
        self.info['num_episodes'] = self.num_episodes
        self.info['episode_reward'] = episode_reward
        self.info['episode_number'] = episode_number + 1   
        self.info['true_history'] = np.zeros((self.max_steps+1, self.num_channels), dtype=int)

        # Sets up dictionaries so that each agent has a history.
        # Although the observation mode and reward mode (and target entity, if the reward function
        # incentivizes jamming an entity) are all given in one dict per agent through the JSON
        # representing the scenario, the actual histories of observations and rewards are split up
        # by information type (action, observation, reward, etc.) instead of by agent here.
        self.info['action_history'] = {}
        self.info['observation_history'] = {}
        self.info['reward_history'] = {}
        self.info['cumulative_reward'] = {}
        for agent_id in self._agent_ids:
            self.info['action_history'][agent_id] = -1+np.zeros((1, self.max_steps+1), dtype=int)
            self.info['observation_history'][agent_id] = np.zeros((self.max_steps+1, self.num_channels), dtype=int)
            self.info['reward_history'][agent_id] = np.zeros(self.max_steps+1, dtype=float)
            self.info['cumulative_reward'][agent_id] = np.zeros(self.max_steps+1, dtype=float)
        # To keep indexing consistent (especially for observations), entities are 1-indexed;
        # so, the information at the index 0 is not used
        self.info['action_history']['all_entities'] = -1 + np.zeros((self.num_entities+1, self.max_steps+1), dtype=int)

        # Reset entities, get initial entity actions, and determine initial player observation.
        for entity in self.entity_list:
            entity.reset(self.info)            
        self.info['true_history'][0], observation_histories = self.__get_entity_and_agent_actions_and_observations(action_dict=None)
        for agent_id in self._agent_ids:
            self.info['observation_history'][agent_id][0] = observation_histories[agent_id]

        # Sets return variables
        infos = {}
        for agent_id in self._agent_ids:
            infos[agent_id] = {}
        return observation_histories, infos
    
    # Returns dictionaries (mapping from each agent's ID to the corresponding information)
    # representing the observations, reward values, terminated values (booleans), truncated
    # values (booleans), and auxiliary diagnostic information (from RLlib and gymnasium
    # documentation, found at
    # https://docs.ray.io/en/latest/_modules/ray/rllib/env/multi_agent_env.html
    # and https://gymnasium.farama.org/api/env/)
    def step(self, action_dict : dict[str: int]) -> tuple[dict[str, any], dict[str, any], dict[str, any],
                                         dict[str, any], dict[str, dict[any, any]]]:
        self.info['step_number'] += 1

        # Get entity actions and determine player observation.
        self.info['true_history'][self.info['step_number']], observation_histories = self.__get_entity_and_agent_actions_and_observations(action_dict=action_dict)
        for agent_id in self._agent_ids:
            self.info['observation_history'][agent_id][self.info['step_number']] = observation_histories[agent_id]
    
        # Although the observation mode and reward mode (and target entity, if the reward function
        # incentivizes jamming an entity) are all given in one dict per agent through the JSON
        # representing the scenario, the actual observations and rewards are split up by information
        # type (action, observation, reward, etc.) instead of by agent here
        observation_per_agent = {}
        reward_per_agent = {}
        done_per_agent = {}
        done_per_agent["__all__"] = False
        info_per_agent = {}
        for agent_id in self._agent_ids:
            observation_per_agent[agent_id] = observation_histories[agent_id]
            reward_per_agent[agent_id] = self.info['reward_history'][agent_id][self.info['step_number']]
            done_per_agent[agent_id] = False
            if self.info['step_number'] == self.max_steps:
                self.info['episode_reward'][agent_id] = np.append(self.info['episode_reward'][agent_id], self.info['cumulative_reward'][agent_id][self.info['step_number']])
                done_per_agent[agent_id] = True
                done_per_agent['__all__'] = True
            info_per_agent[agent_id] = {} # may be changed later

        #                                               terminateds     truncateds 
        return observation_per_agent, reward_per_agent, done_per_agent, done_per_agent, info_per_agent
    
    def render(self):
        if self.render_mode != 'null':
            if self.info['step_number'] == 0:
                self.next_frame_time = time.time()
            
            self.renderer.render(self.info, self.agents_info)  
            self.next_frame_time += 1.0/self.render_fps
            time.sleep(1/self.render_fps)

            if time.time() < self.next_frame_time:
                time.sleep(self.next_frame_time - time.time())
        return  

    def close(self):        
        input('Press any key to end the simulation...\n')
        return
    
    def __validate_scenario_metadata(self):
        # Validate scenario environment parameters.
        error_message = 'Environment parameter \'num_channels\' is invalid.'
        assert self.scenario_metadata['environment']['num_channels'] > 0, error_message
        error_message = 'Environment parameter \'max_steps\' is invalid.'
        assert self.scenario_metadata['environment']['max_steps'] > 0, error_message

        # Validates parameters for each agent
        for agent_id in self._agent_ids:
            error_message = 'Invalid observation mode for {}. Must be one of the following options: {}'.format(agent_id, self.metadata["observation_modes"])
            assert self.scenario_metadata['environment']['agents'][agent_id]['observation_mode'] in self.metadata['observation_modes'], error_message
            error_message = 'Invalid reward mode for {}. Must be one of the following options: {}'.format(agent_id, self.metadata["reward_modes"])
            assert self.scenario_metadata['environment']['agents'][agent_id]['reward_mode'] in self.metadata['reward_modes'], error_message
            if self.scenario_metadata['environment']['agents'][agent_id]['reward_mode'] == 'jam':
                error_message = f'Invalid target entity name for agent {agent_id}. Must correspond to the name of one of the entity labels in the scenario file.'
                assert self.scenario_metadata['environment']['agents'][agent_id]['target_entity'] in self.scenario_metadata['entities'].keys() or self.scenario_metadata['environment']['agents'][agent_id]['target_entity'] == None, error_message
        
        # Validate scenario render parameters.
        error_message = 'Invalid render mode. Must be one of the following options: {}'.format(self.metadata["render_modes"])
        assert self.scenario_metadata['render']['render_mode'] is None or self.scenario_metadata['render']['render_mode'] in self.metadata['render_modes'], error_message
        error_message = 'Render parameter \'render_fps\' is invalid.'
        assert self.scenario_metadata['render']['render_fps'] > 0, error_message
        error_message = 'Render parameter \'render_history\' is invalid.'
        assert self.scenario_metadata['render']['render_history'] > 0, error_message
        if 'observation_mode' in self.scenario_metadata['render']:
            error_message = f'Invalid observation mode for rendering. Must be one of the following options: {self.metadata["observation_modes"]}'
            assert self.scenario_metadata['render']['observation_mode'] in self.metadata['observation_modes'], error_message
    
    def __observation_space_encoder(self, observation_vect, agent_id):
        observation_int = 0
        for idx in range(len(observation_vect)):
            observation_int += (self.observation_base[agent_id]**idx)*observation_vect[idx]

        return observation_int
    
    # At the current timestep, uses the given actions per agent (and generates actions
    # for the entities based on their type) in order to create a true observation (which,
    # to be clear, is distinct from the observation space) and an observation per agent.
    def __get_entity_and_agent_actions_and_observations(self, action_dict):
        current_step = self.info["step_number"]

        # Get each entity's action and determine the true observation.
        # All entity actions range from 0 to (num_channels - 1), and a lack of action
        # (no transmission on any channel) is represented by the action equalling -1.
        entity_idx = 0
        true_observation = np.zeros(self.num_channels, dtype=int)
        for entity in self.entity_list:
            entity_idx += 1
            entity_action = entity.get_action(self.info)
            self.info['action_history']['all_entities'][entity_idx][current_step] = entity_action
            # If two or more entities' actions are to choose the same channel, set the observation to the number of entities + 1.
            if entity_action != -1:
                if true_observation[entity_action] == 0:
                    true_observation[entity_action] = entity_idx
                else:
                    true_observation[entity_action] = self.num_entities + self._num_agents + 1
        
        # Gets each agent's action and determines the true observation.
        # All agent actions originally range from 1 to num_channels, where a lack of an action
        # (i.e., no transmission on any channel) is represented by 0. However, all of these
        # values are shifted down by 1 so that they can more seamlessly integrate with
        # the (numpy) arrays in Python.
        if action_dict == None:
            # This condition is only satisfied when calling env.reset(), and is never satisfied
            # when calling env.step().
            for agent_id in self._agent_ids:
                self.info['action_history'][agent_id][0][current_step] = -1
                self.info['reward_history'][agent_id][current_step] = 0
        else:
            # Processes each agent's action, and updates the true observation based on them.
            # This should be done in a separate loop from the calculation of the rewards
            # to avoid the true observation being updated in the middle of calculating
            # the rewards (especially for agents with the `dsa` observation mode).
            for agent_id in self._agent_ids:
                action_dict[agent_id] -= 1
                agent_action = action_dict[agent_id]
                self.info['action_history'][agent_id][0][current_step] = agent_action
                if agent_action != -1:
                    if true_observation[agent_action] == 0:
                        true_observation[agent_action] = self.agent_id_to_num[agent_id]
                    else:
                        true_observation[agent_action] = self.num_entities + self._num_agents + 1

            # Calculates rewards.
            for agent_id in self._agent_ids:
                agent_action = action_dict[agent_id]
                if agent_action == -1:
                    self.info['reward_history'][agent_id][current_step] = 0
                else:
                    if self.agents_info[agent_id]['reward_mode'] == 'dsa':
                        status_of_chosen_channel = true_observation[agent_action]
                        self.info['reward_history'][agent_id][current_step] = 2.0*int(
                            status_of_chosen_channel==self.agent_id_to_num[agent_id] # checks to see that only that agent is in the channel
                        )-1.0
                        # self.info['reward_history'][agent_id][current_step] /= 100.0 # Used for PPO!
                    elif self.agents_info[agent_id]['reward_mode'] == 'jam':
                        target_entity_action = self.info['action_history']['all_entities'][self.target_idx[agent_id]][current_step]
                        self.info['reward_history'][agent_id][current_step] = 2.0*int(
                            agent_action == target_entity_action # checks to see that the agent and its target entity are in the same channel
                        )-1.0
                        # self.info['reward_history'][agent_id][current_step] /= 100.0 # Used for PPO!

                        # The above calculation is used instead of the one below, because a collision
                        # may make it appear that an entity is not in a specific channel, when it actually
                        # may be and it's just part of the collision. However, depending on the use case,
                        # a developer might want to incentivize the algorithm to only target an entity
                        # if it's alone in the channel, which would use the calculation below.

                        # self.info['reward_history'][agent_id][self.info['step_number']] = 2.0*int(
                        #     self.info['true_history'][self.info['step_number']-1][agent_action]==self.target_idx[agent_id]
                        # )-1.0
                self.info['cumulative_reward'][agent_id][current_step] = \
                    self.info['cumulative_reward'][agent_id][current_step - 1] + self.info['reward_history'][agent_id][current_step]

        # Set the observation for each agent.
        player_observation = {}
        for id in self._agent_ids:
            if self.agents_info[id]['observation_mode'] == 'detect':
                player_observation[id] = np.array(true_observation > 0, dtype=int)
            elif self.agents_info[id]['observation_mode'] == 'classify':
                player_observation[id] = true_observation

        # true_observation is a numpy array; player_observation is a dict of numpy arrays,
        # mapping from agent ID to observation
        return true_observation, player_observation
