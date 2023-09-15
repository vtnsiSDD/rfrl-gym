import numpy as np
from rfrl_gym.entities.entity import Entity

# An entity that hops in a repeating pattern over a set of channels.
class AgileFreq(Entity):
    def __init__(self, entity_label, num_channels, channels, onoff=[1,1,0], start=None, stop=None):
        super().__init__(entity_label, num_channels, channels, onoff, start, stop)
    
    def _validate_self(self):
        pass

    def _get_action(self):
        if self.on_flag == 0:
            return -1
        elif self.on_flag == 1:
            occupied_channels = np.zeros(self.num_channels)
            if self.info['action_history'][self.info['step_number']-1] != -1:
                occupied_channels[self.info['action_history'][self.info['step_number']-1]] = 1
            for idx in range(self.num_channels):
                if (self.info['true_history'][self.info['step_number']-1][idx] != self.entity_idx) and (self.info['true_history'][self.info['step_number']-1][idx] != 0):
                    occupied_channels[idx] = self.info['true_history'][self.info['step_number']-1][idx] > 0
            
            if sum(occupied_channels) == self.num_channels:
                return -1
            elif occupied_channels[self.current_channel] == 1:
                self.current_channel = np.random.choice(np.where(occupied_channels == 0)[0])
            return self.current_channel

    def _reset(self):
        self.current_channel = np.random.choice(self.channels)