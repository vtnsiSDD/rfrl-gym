from rfrl_gym.renderers.renderer import Renderer

class TerminalRenderer(Renderer):
    def __init__(self, num_episodes, scenario_metadata):
        super(TerminalRenderer, self).__init__(num_episodes, scenario_metadata)

    def _render(self):
        # Print the channel occupancy to the terminal.
        self.__terminal_divider()
        for channel in range(self.num_channels):
            print('|', end='')
            for step in range(self.render_history):
                if not step > self.info['step_number']:
                    channel_entity = self.info['observation_history'][self.info['step_number']-step][channel]
                    if self.info['action_history'][self.info['step_number']-step] == channel:
                        if channel_entity != 0 and self.observation_mode == 'classify':
                            if channel_entity == len(self.entity_list) + 1:
                                print_element = 'xCx'                                
                            else:
                                print_element = 'x' + str(channel_entity) + 'x'
                        elif channel_entity != 0:
                            print_element = ' X '
                        elif self.observation_mode == 'classify':
                            print_element = ' 0 '
                        else:
                            print_element = ' 1 '
                    else:
                        if self.observation_mode == 'detect' and channel_entity != 0:
                            print_element = ' O '
                        elif self.observation_mode == 'classify' and channel_entity != 0:
                            if channel_entity == self.num_entities + 1:
                                print_element = ' C '
                            else:
                                print_element = ' ' + str(channel_entity) + ' '
                        else:
                            print_element = '   '
                    print(print_element + '|', end='')
            print(' Channel ' + str(channel))
        
        # Print the step numbers to the terminal.
        self.__terminal_divider()        
        for step in range(self.render_history):
            if not step > self.info['step_number']:
                print('|' + ('%03d' % (self.info['step_number']-step,)), end='')
        print('| Step Number')
        self.__terminal_divider()

        # Print the rewards to the terminal.
        print('Step Reward: ' + str(self.info['reward_history'][self.info['step_number']]))
        print('Cummulative Reward: ' + str(self.info['cumulative_reward'][self.info['step_number']]))
        if self.num_episodes > 1:
            print('Episode Rewards: ' + str(self.info['episode_reward']))
        print('')
        
        # Print the legend to the terminal.
        if self.observation_mode == 'classify':
            entity_idx = 0
            print(' 0 : Player')
            for entity in self.entity_list:
                entity_idx += 1
                print(' ' + str(entity_idx) + ' : ' + str(entity) + ' Entity')
            print(' C : ' + 'Multi-Entity Collision')
            print('xNx: Player Collision with Entity N or Multi-Entities (xCx)\n')
        else:
            print('O: Entities')
            print('1: Player')
            print('X: Player Collision with an Entity\n')        

    def _reset(self):
        pass

    def __terminal_divider(self):
        print('=', end='')
        for step in range(self.render_history):
            if not step > self.info['step_number']:
                print('====', end='')
        print('')