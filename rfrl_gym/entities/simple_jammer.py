import numpy as np
from rfrl_gym.entities.entity import Entity

# An entity that hops in a repeating pattern over a set of channels.
class SimpleJammer(Entity):
    def __init__(self, entity_label, num_channels, channels, onoff=[1,1,0], start=None, stop=None, modem_params=None, avoid_repeats=True):
        self.avoid_repeats = avoid_repeats
        super().__init__(entity_label, num_channels, channels, onoff, start, stop, modem_params)
    
    def _validate_self(self):
        pass       
    
    def _get_action(self):
        if self.on_flag == 0 or self.info['step_number'] == 0:
            return -1
        elif self.on_flag == 1:  
            occupied_channels = np.zeros(self.num_channels)
            if self.info['action_history'][0][self.info['step_number']-1] != -1:
                occupied_channels[self.info['action_history'][0][self.info['step_number']-1]] = 1
            for idx in range(self.num_channels):
                if (self.info['true_history'][self.info['step_number']-1][idx] != self.entity_idx) and (self.info['true_history'][self.info['step_number']-1][idx] != 0):
                    if occupied_channels[idx] == 1 or self.info['true_history'][self.info['step_number']-1][idx] == self.info["num_entities"]+1:
                        if self.current_channel == idx:
                            occupied_channels[idx] = 2
                        else:
                            occupied_channels[idx] = 0
                    else:
                        occupied_channels[idx] = self.info['true_history'][self.info['step_number']-1][idx] > 0
            
            if sum(occupied_channels) == 0:
                return -1
            else:
                if (self.avoid_repeats == True) and (len(np.where(occupied_channels == 1)[0]) > 0):
                    self.current_channel = np.random.choice(np.where(occupied_channels == 1)[0])
                else:
                    self.current_channel = np.random.choice(np.where(occupied_channels > 0)[0])

                return self.current_channel

    def _reset(self):
       self.current_channel = None