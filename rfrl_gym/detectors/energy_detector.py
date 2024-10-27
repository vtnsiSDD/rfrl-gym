import numpy as np
from rfrl_gym.detectors.detector import Detector

class EnergyDetector(Detector):
    def __init__(self, num_channels):
        super().__init__(num_channels)

    def _validate_self(self):
        pass

    # Note: when we get sensing results, they are the results for the frame rendered
    # immediately before the environment's step() method was called
    # Since the previous frame's sensing results were calculated in pyqt_renderer.py,
    # we can just fetch it from self.info['sensing_energy_history']
    def _get_sensing_results(self):
        for k in range(self.num_channels):
            sensed_result = self.info['sensing_energy_history'][self.info['step_number']][k]
            if sensed_result > 0.001:
                self.info['sensing_history'][self.info['step_number']][k] = 1
            else:
                self.info['sensing_history'][self.info['step_number']][k] = 0
        return self.info
    
    def _reset(self):
        pass

