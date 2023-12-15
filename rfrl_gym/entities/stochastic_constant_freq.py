import numpy as np
from rfrl_gym.entities.entity import Entity

# An entity that hops in a random pattern over a set of channels where the probability of chosing each channel is given by a user-defined weighting.
class StochasticConstantFreq(Entity):
    # channel_weights - A vector of probabilities corresponding to the channels vector that determines the probability of chosing each channel.
    def __init__(self, entity_label, num_channels, channels, start=None, stop=None, modem_params=None, percent_on=1):
        self.percent_on = percent_on
        super().__init__(entity_label, num_channels, channels, [1,1,0], start, stop, modem_params)

    def _validate_self(self):
        pass  

    def _get_action(self):
        rand_num = np.random.uniform(0,1)
        if rand_num > self.percent_on:
            return -1
        else:
            return self.channels[0]

    def _reset(self):
        pass