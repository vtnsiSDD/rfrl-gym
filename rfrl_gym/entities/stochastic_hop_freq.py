import numpy as np
from rfrl_gym.entities.entity import Entity

# An entity that hops in a random pattern over a set of channels where the probability of chosing each channel is given by a user-defined weighting.
class StochasticHopFreq(Entity):
    # channel_weights - A vector of probabilities corresponding to the channels vector that determines the probability of chosing each channel.
    def __init__(self, entity_label, num_channels, channels, onoff=[1,1,0], start=None, stop=None, channel_weights=[]):
        self.channel_weights = channel_weights
        if self.channel_weights == []:
            self.channel_weights = (1.0/len(channels))*np.ones(len(channels))
        super().__init__(entity_label, num_channels, channels, onoff, start, stop)
            
    def _validate_self(self):
        if (len(self.channel_weights) != len(self.channels)) or round(np.sum(self.channel_weights),2) != 1.0:
            raise Exception('Parameter \'channel_weights\' of entity \'{}\' is not valid.'.format(self.entity_label))  

    def _get_action(self):
        if self.on_flag == 0:
            return -1
        elif self.on_flag == 1:
            return np.random.choice(self.channels, 1, p=self.channel_weights)

    def _reset(self):
        pass