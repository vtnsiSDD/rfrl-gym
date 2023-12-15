import numpy as np
import rfrl_gym.datagen.modems as modems

class IQ_Gen:
    def __init__(self, num_channels, num_entities, history, entity_list):
        # Placeholder for data generation.
        self.num_channels = num_channels
        self.num_entities = num_entities + 1
        self.history = history
        self.entity_list = entity_list
        self.entity_modems = self.__init_entity_modems(entity_list)

        self.samples_per_step = 10000
        self.noise_std = 0.01

        self.fc = np.linspace(-0.5, 0.5, self.num_channels+1)+1/self.num_channels/2
        self.rng=np.random.default_rng()

    def gen_iq(self, actions):
        self.samples = np.roll(self.samples, self.samples_per_step, axis=0)
        self.samples[0:self.samples_per_step] =  self.rng.normal(0.0, self.noise_std/np.sqrt(2.0), self.samples_per_step) + 1.0j*(self.rng.normal(0.0, self.noise_std/np.sqrt(2.0), self.samples_per_step))
        for k in range(self.num_channels):
            for kk in range(self.num_entities):
                if actions[kk] == k:
                    if kk != 0:
                        entity = self.entity_list[kk-1]

                        cent_freq = self.rng.uniform(entity.modem_params['center_frequency'][0],entity.modem_params['center_frequency'][1])
                        start = entity.modem_params['start']
                        duration = entity.modem_params['duration']
                        modem = self.entity_modems[entity]

                        data = modem.gen_samps(int(duration*self.samples_per_step))
                        self.t = np.linspace(0, int(duration*self.samples_per_step), int(duration*self.samples_per_step))
                        data = data * np.exp(1j*2*np.pi*(self.fc[k]+cent_freq/self.num_channels)*self.t)
                        self.samples[int(start*self.samples_per_step):int((start+duration)*self.samples_per_step)] += data
                    else:
                        self.t = np.linspace(0, self.samples_per_step, self.samples_per_step)
                        data = (1.0/10.0)*np.exp(1j*2*np.pi*(1/self.num_channels)/2*((self.t-self.samples_per_step/2)**2)/self.samples_per_step)
                        data = data * np.exp(1j*2*np.pi*self.fc[k]*self.t)
                        data[0:100] = np.linspace(0,1,100)*data[0:100]
                        data[-100:] = np.linspace(1,0,100)*data[-100:]
                        self.samples[0:self.samples_per_step] += data
        return self.samples

    def reset(self):
        self.samples = self.rng.normal(0.0, self.noise_std/np.sqrt(2.0), self.samples_per_step*self.history) + 1.0j*(self.rng.normal(0.0, self.noise_std/np.sqrt(2.0), self.samples_per_step*self.history))

    # This function will create a dict that maps an entity within the scenario to its corresponding modem
    def __init_entity_modems(self, entity_list):
        entity_modems = dict()

        for entity in entity_list:
            bandwidth = entity.modem_params['bandwidth']
            sps = int(self.num_channels / bandwidth)

            # Construct this entity's modem
            if entity.modem_params['type'] == 'psk' or entity.modem_params['type'] == 'qam' or entity.modem_params['type'] == 'ask':
                beta = 0.35
                span = 10
                trim = 0
                modem = modems.LDAPM(sps=sps, mod_type=entity.modem_params['type'], mod_order=entity.modem_params['order'], filt_type=entity.modem_params['filter'], beta=beta, span=span, trim=trim)
            else:
                modem = modems.Tone(sps=sps, mod_type=entity.modem_params['type'])

            entity_modems[entity] = modem

        return entity_modems
