from rfrl_gym.entities.entity import Entity

# An entity that always chooses the same action.
class ConstantFreq(Entity):
    def __init__(self, entity_label, num_channels, channels, onoff=[1,1,0], start=None, stop=None, modem_params=None):
        super().__init__(entity_label, num_channels, channels, onoff, start, stop, modem_params)

    def _validate_self(self):
        pass

    def _get_action(self):
        if self.on_flag == 0:
            action = -1
        else:
            action = self.channels[0]
        return action

    def _reset(self):
        pass