import numpy as np
import rfrl_gym.datagen.filters as filters

class LDAPM:
    def __init__(self, sps=2, symb_table=None, mod_type='psk', mod_order=4, filt_type='rrc', beta=0.35, span=10, trim=1):
        self.symb_table = symb_table
        if self.symb_table == None:
            self.mod_type = mod_type
            self.mod_order = mod_order
            self.symb_table = []
            self.__symb_table_create()
        else:
            self.mod_type = 'custom'
            self.mod_order = np.log2(len(symb_table))
        self.__symb_table_norm()
        self.set_filter(sps, filt_type, beta, span, trim)

    def __symb_table_create(self):
        if self.mod_type == 'ask':
            for x in range(0, self.mod_order):
                self.symb_table.append(x)
        elif self.mod_type == 'psk':
            self.symb_table = [(np.cos((2.0*np.pi*x)/self.mod_order+(np.log2(self.mod_order)-1.0)*(np.pi/4.0))+1.0j*np.sin((2.0*np.pi*x)/self.mod_order+(np.log2(self.mod_order)-1.0)*(np.pi/4.0))) for x in range(0,self.mod_order)]
        elif self.mod_type == 'qam':
            max_val = int(self.mod_order**(1.0/2.0)-1.0)
            for x in range(-max_val,max_val+1, 2):
                for y in range(-max_val,max_val+1, 2):
                    self.symb_table.append(x+1.0j*y)

    def __symb_table_norm(self):
        self.symb_table = np.divide(self.symb_table, np.mean(np.abs(self.symb_table)**2.0)**(1.0/2.0))

    def set_filter(self, sps, filt_type, beta, span, trim):
        class_method = getattr(filters, filt_type)
        if filt_type == 'SQR':
            self.filt = class_method(sps, span=span, trim=trim)
        else:
            self.filt = class_method(sps, beta=beta, span=span, trim=trim)

    def gen_symbs(self, total_symbs):
        return np.random.choice(self.symb_table, total_symbs)

    def get_samps(self, symbs):
        return self.filt.filter(symbs, 'int')

    def gen_samps(self, total_samps):
        if self.filt.trim == 1:
            total_symbs = int(np.ceil((total_samps-int(self.filt.span/2))/self.filt.sps))
        else:
            total_symbs = int(np.ceil((total_samps-self.filt.span*self.filt.sps+1)/self.filt.sps))

        if total_symbs < 1:
            return np.array([])

        samps = self.filt.filter(self.gen_symbs(total_symbs), 'int')

        diff = len(samps)-total_samps
        if diff == 0:
            return samps
        else:
            return samps[int(diff/2):-(int(diff/2)+1*(diff%2))]

    def get_symbs(self, samps):
        return np.array(self.filt.filter(samps, 'dec'))

class Tone:
    def __init__(self, sps=2, ramp_per=0.1, mod_type='constant'):
        self.sps = sps
        self.mod_type = mod_type
        self.ramp_per = ramp_per

    def gen_samps(self, total_samps):
        ramp_up = np.linspace(0, 1, int(self.ramp_per*total_samps), endpoint=True)
        ramp_down = np.linspace(1, 0, int(self.ramp_per*total_samps), endpoint=True)

        samps = np.concatenate((ramp_up, np.ones(total_samps-len(ramp_up)-len(ramp_down)), ramp_down))
        if self.mod_type == 'p_fmcw' or self.mod_type == 'n_fmcw':
            t = np.linspace(0, total_samps, total_samps)
            bw = 1/self.sps

            if self.mod_type == 'p_fmcw':
                samps = (1.0/10.0)*samps*np.exp(1j*2*np.pi*bw/2*((t-total_samps/2)**2)/total_samps)
            elif self.mod_type == 'n_fmcw':
                samps = (1.0/10.0)*samps*np.exp(-1j*2*np.pi*bw/2*((t-total_samps/2)**2)/total_samps)
        return samps