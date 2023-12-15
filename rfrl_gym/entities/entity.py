import numpy as np

class Entity:
    # num_channels - The number of possible channels for the given environment.
    #     channels - A vector listing the set of channels the entity can choose from.
    #        onoff - A vector of three values [X,Y,Z] where:
    #                   X is a value of 0 or 1 that determines whether the entity starts in an off or on position. 
    #                   Y is a positive integer that determines how many steps before turning off (if applicable).
    #                   Z is a positive integer that determines how many steps before turning on (if applicable).
    #        start - The time step number when the entity will be enabled and start executing its get_action() and cycle_onoff() functions.
    #         stop - The time step number when the entity will be disabled and stop executing its get_action() and cycle_onoff() functions.
    def __init__(self, entity_label, num_channels, channels, onoff, start, stop, modem_params):
        self.entity_label = entity_label
        self.num_channels = num_channels
        self.channels = channels
        self.onoff = onoff   
        self.start = start
        self.stop = stop
        self.modem_params = modem_params

        if self.start == None:
            self.start = 0
        if self.stop == None:
            self.stop = np.inf
        self.info = []  

        self.__validate_entity()

    # Set the index of the entity.
    def set_entity_index(self, entity_idx):
        self.entity_idx = entity_idx

    # Check if the entity has valid parameters.
    def __validate_entity(self):
        # Validate channels parameter.
        assert isinstance(self.channels, list), 'Parameter \'channels\' of entity \'{}\' must be of type list.'.format(self.entity_label)
        for channel in self.channels:
            if channel < 0 or channel >= self.num_channels:
                raise Exception('Parameter \'channels\' of entity \'{}\' is not valid for the current gym environment.'.format(self.entity_label))
        # Validate onoff parameter.
        assert self.onoff[0] in [0,1], 'Parameter \'onoff\' of entity \'{}\' is not valid.'.format(self.entity_label)
        if (self.onoff[1] < 0 or str(type(self.onoff[1])) != '<class \'int\'>') or (self.onoff[2] < 0 or str(type(self.onoff[2])) != '<class \'int\'>'):
            raise Exception('Parameter \'onoff\' of entity \'{}\' is not valid.'.format(self.entity_label))
        # Validate start and stop parameters.
        if (str(type(self.start)) != '<class \'int\'>'):
            raise Exception('Parameter \'start\' of entity \'{}\' is not valid.'.format(self.entity_label))
        if (str(type(self.stop)) != '<class \'int\'>') and (self.stop != np.inf):
            raise Exception('Parameter \'stop\' of entity \'{}\' is not valid.'.format(self.entity_label))       

        self._validate_self()
                    
    def _validate_self(self):
        raise Exception('Necessary entity function _validate_self() not implemented for entity type {}'.format(type(self).__name__))
    
    # Get the next action from the entity and check whether signal should turn on/off.
    def get_action(self, info):
        self.info = info
        if (self.info['step_number'] >= self.start) and (self.info['step_number'] <= self.stop):
            action = self._get_action()
            self.__cycle_onoff()
            return action
        else:
            return -1
    
    # Check to make sure the entity has an action function.
    def _get_action(self):
        raise Exception('Necessary entity function _get_action() not implemented for entity type {}'.format(type(self).__name__))
  
    # Reset the entities parameters
    def reset(self, info):
        self.info = info
        self._reset()
        self.count = 0
        self.on_flag = 0

        if self.onoff[0] == 0 and self.onoff[2] != 0:
            self.on_flag = 0
        elif self.onoff[0] == 1 and self.onoff[1] != 0:
            self.on_flag = 1
    
    # Check to make sure the entity has a reset function.
    def _reset(self):
        raise Exception('Necessary entity function _reset() not implemented for entity type {}'.format(type(self).__name__))
    
    # Toggle the on/off flag to tell the entity whether or not to do an action.
    def __cycle_onoff(self):
        self.count += 1
        if (self.on_flag == 1 and self.count == self.onoff[1]):
            if self.onoff[2] != 0:
                self.on_flag = 0
            self.count = 0
        elif (self.on_flag == 0 and self.count == self.onoff[2]):
            if self.onoff[1] != 0:
                self.on_flag = 1
            self.count = 0