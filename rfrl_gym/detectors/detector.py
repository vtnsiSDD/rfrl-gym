import numpy as np

class Detector:
    def __init__(self, num_channels):
        self.num_channels = num_channels
    
    def get_sensing_results(self, info):
        self.info = info
        sensing_results = self._get_sensing_results()
        return sensing_results
    
    # Check to make sure the detector has a sensing function
    def _get_sensing_results(self):
        raise Exception('Necessary function _get_sensing_results() not implemented for detector')

    # Check to make sure the detector has a reset function.
    def _reset(self):
        raise Exception('Necessary detector function _reset() not implemented for detector')
