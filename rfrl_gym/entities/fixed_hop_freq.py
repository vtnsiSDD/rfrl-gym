import numpy as np
from rfrl_gym.entities.entity import Entity

# An entity that hops in a repeating pattern over a set of channels.
class FixedHopFreq(Entity):
    # rand_hop - Toggles whether the channels vector is iterated through sequentially or in a repeating random order.
    def __init__(self, entity_label, num_channels, channels, onoff=[1,1,0], start=None, stop=None, modem_params=None, rand_hop=1):
        self.rand_hop = rand_hop
        super().__init__(entity_label, num_channels, channels, onoff, start, stop, modem_params)

        self.rng = np.random.default_rng()
        if self.rand_hop == 1:
            self.rng.shuffle(self.channels)
    
    def _validate_self(self):
        assert self.rand_hop in [0,1], 'Parameter \'rand_hop\' of entity \'{}\' is not valid.'.format(self.entity_label)

    def _get_action(self):
        if self.on_flag == 0:
            return -1
        elif self.on_flag == 1:
            if self.count == 0:
                self.channel_index += 1
            if self.channel_index == len(self.channels):
                self.channel_index = 0
            return self.channels[self.channel_index]

    def _reset(self):
        self.channel_index = -1