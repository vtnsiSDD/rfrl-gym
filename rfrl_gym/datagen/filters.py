import numpy as np

class _filterbase:
    def __init__(self, sps, span=10, trim=1):
        self.sps = sps
        self.span = span
        self.trim = trim

        self.taps = []
        self.gen_taps()
        self.delay = len(self.taps)

    def __upsample(self, symbs):
        samps = np.zeros(self.sps*len(symbs)).astype(np.csingle)
        for x in range(0, len(symbs)):
            samps[x*self.sps] = symbs[x]
        return samps

    def __downsample(self, samps, delay, cutoff):
        return [samps[x] for x in range(delay, cutoff, self.sps)]

    def filter(self, data, type):
        if type == 'int':
            samps = np.convolve(self.__upsample(data), self.taps, 'full')
            if self.trim == 1:
                return samps[(int(self.delay/2)-self.sps):-(int(self.delay/2))]
            else:
                return samps
        elif type == 'dec':
            samps = np.convolve(data, self.taps, 'full')
            if self.trim == 1:
                return self.__downsample(samps, int(self.delay/2)+self.sps, len(samps)-int(self.delay/2))
            else:
                return self.__downsample(samps, self.delay-1, len(samps)-self.delay)

class RC(_filterbase):
    def __init__(self, sps, beta=0.35, span=10, trim=1):
        self.beta = beta
        super().__init__(sps, span, trim)

    def gen_taps(self):
        for x in range(-int(self.span*self.sps/2), int(self.span*self.sps/2)+1):
            if x == 0:
                self.taps.append(1.0/(self.sps**(1.0/2.0)))
            elif abs(x) == self.sps/(2.0*self.beta):
                self.taps.append((1.0/(self.sps**(1.0/2.0)))*(self.beta/2.0)*np.sin(np.pi/(2.0*self.beta)))
            else:
                self.taps.append((1.0/(self.sps**(1.0/2.0)))*(np.sin((np.pi*x)/self.sps)/((np.pi*x)/self.sps))*(np.cos((np.pi*self.beta*x)/self.sps)/(1.0-((2.0*self.beta*x)/self.sps)**2.0)))

class RRC(RC):
    def __init__(self, sps, beta=0.35, span=10, trim=1):
        super().__init__(sps, beta, span, trim)

    def gen_taps(self):
        for x in range(-int(self.span*self.sps/2), int(self.span*self.sps/2)+1):
            if x == 0:
                self.taps.append((1.0/(self.sps**(1.0/2.0)))*((1.0-self.beta)+((4.0*self.beta)/np.pi)))
            elif abs(x) == self.sps/(4.0*self.beta):
                self.taps.append((self.beta/((2.0*self.sps)**(1.0/2.0)))*((1.0+2.0/np.pi)*np.sin(np.pi/(4.0*self.beta))+(1.0-2.0/np.pi)*np.cos(np.pi/(4.0*self.beta))))
            else:
                self.taps.append((1.0/((self.sps)**(1.0/2.0)))*((np.sin((np.pi*x*(1.0-self.beta))/self.sps)+(((4.0*self.beta*x)/self.sps)*np.cos((np.pi*x*(1.0+self.beta))/self.sps)))/(((np.pi*x)/self.sps)*(1.0-((4.0*self.beta*x)/self.sps)**(2.0)))))

class SQR(_filterbase):
    def __init__(self, sps, span=10, trim=1):
        super().__init__(sps, span, trim)

    def gen_taps(self):
        self.taps = np.zeros(2*int(self.span*self.sps/2)+1)
        self.taps[int((len(self.taps)/2-(self.sps-1)/2)):int(((len(self.taps)/2+(self.sps-1)/2+1)))] = 1.0/(self.sps**(1.0/2.0))
