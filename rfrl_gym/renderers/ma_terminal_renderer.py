from rfrl_gym.renderers.ma_renderer import MultiAgentRenderer

class MultiAgentTerminalRenderer(MultiAgentRenderer):
    def __init__(self, num_episodes, scenario_metadata):
        super(MultiAgentTerminalRenderer, self).__init__(num_episodes, scenario_metadata)

    def _render(self):
        # # Overwrites the previous print
        # if self.info['step_number'] >= 1:
        #     num_lines_per_step = self.num_channels + 3 * self.num_agents + self.num_entities + 13
        #     for i in range(num_lines_per_step):
        #         print("\033[A\033[K", end="")

        # Maps from agent or entity to integer
        to_num = {}
        idx = 0
        for entity in self.entity_list:
            idx += 1
            to_num[entity] = idx
        for agent_id in self.agents_info:
            idx += 1
            to_num[agent_id] = idx            

        # Prints the channel occupancy to the terminal.
        self.__terminal_divider()
        justification_width = 3
        for channel in range(self.num_channels):
            print('|', end='')
            for step in range(self.render_history):
                if not step > self.info['step_number']:
                    channel_entity = self.info['true_history'][self.info['step_number']-step][channel]
                    print_element = None

                    if channel_entity > 0:
                        if self.observation_mode == 'detect':
                            print_element = str(1).center(justification_width)
                        elif self.observation_mode == 'classify':
                            # No collision
                            if channel_entity <= self.num_entities + self.num_agents:
                                print_element = str(channel_entity).center(justification_width)

                            # Collision
                            else:
                                involves_an_agent = False
                                for agent_id in self.agents_info:
                                    if self.info["action_history"][agent_id][0][self.info['step_number']-step] == channel:
                                        involves_an_agent = True
                                        break
                                if involves_an_agent:
                                    print_element = str(-2).center(justification_width)
                                else:
                                    print_element = str(-1).center(justification_width)
                    else:
                        print_element = '   '

                    print(print_element + '|', end='')
            print(' Channel ' + str(channel))
        
        # Prints the step numbers to the terminal.
        self.__terminal_divider()        
        for step in range(self.render_history):
            if not step > self.info['step_number']:
                print('|' + ('%03d' % (self.info['step_number']-step,)), end='')
        print('| Step Number')
        self.__terminal_divider()

        # Prints the rewards to the terminal.
        print('Step Reward:')
        total = 0
        for agent_id in self.agents_info:
            print('\t{}: {}'.format(agent_id, self.info['reward_history'][agent_id][self.info['step_number']]))
            total += self.info['reward_history'][agent_id][self.info['step_number']]
        print(f'\tTotal: {total}')
        print('Cumulative Reward:')
        cumulative_total = 0
        for agent_id in self.agents_info:
            print('\t{}: {}'.format(agent_id, self.info['cumulative_reward'][agent_id][self.info['step_number']]))
            cumulative_total += self.info['cumulative_reward'][agent_id][self.info['step_number']]
        print(f'\tTotal: {cumulative_total}')
        print('')
        
        # Prints the legend to the terminal.
        print("Legend:")
        if self.observation_mode == 'classify':
            for entity in self.entity_list:
                print(str(to_num[entity]).center(justification_width) + ': ' + str(entity) + ' (Entity)')
            
            for agent_id in self.agents_info:
                print(str(to_num[agent_id]).center(justification_width) + ': ' + str(agent_id) + ' (Agent)')

            print(str(-1).center(justification_width) + ': Collision Between Entities')
            print(str(-2).center(justification_width) + ': Collision Involving Agent(s)')
        elif self.observation_mode == 'detect':
            print(str(1).center(justification_width) + ': Non-Empty Channel')
        print()

    def _reset(self):
        pass

    def __terminal_divider(self):
        print('=', end='')
        for step in range(self.render_history):
            if not step > self.info['step_number']:
                print('====', end='')
        print('')