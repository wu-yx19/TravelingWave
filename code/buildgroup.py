from brian2 import *
import pickle
import os
import math
import matplotlib.pyplot as plt

tau = 6*ms
El = -55*mV
Vth = 0*mV
Cm = 0.1875*nF
te = 5*ms
ti = 8.75*ms


@implementation('cython', '''
cdef double position_round(double r1, double r2):
    temp = r1-r2-floor(r1-r2)
    sig = 2*(temp > 0)-1
    temp = temp*sig
    sig = temp > 0.5
    u = sig-temp*sig
    l = temp*(1-sig)
    return u+l
''')
@check_units(r1=1, r2=1, result=1)
def position_round(r1, r2):
    temp = r1-r2-floor(r1-r2)
    sig = 2*(temp > 0)-1
    temp = temp*sig
    sig = temp > 0.5
    u = sig-temp*sig
    l = temp*(1-sig)
    return u+l


class Exc_group:

    def __init__(self, E_num=100, E_l_mean=0.03, E_l_var=0, E_I0=1*nA, E_gain=3, E_I_rand=0*nA, E_w0=1*nA, E_cp=0.25) -> None:
        self.num = E_num
        self.cp = E_cp
        self.eqs = '''
        r : 1 (constant)
        l = l_mean+(-0.5+rand())*l_var*2 : 1 (constant over dt)
        I_st = I0*(1+gain*exp(-(position_round(r,position(t)))**2/(2*l**2))) : amp
        dv/dt = (El-v)/tau+(I_exc-I_inh+I_st+I_rand*(tau**0.5)*xi_exc)/Cm : volt 
        dI_exc/dt = -I_exc/te : amp
        dI_inh/dt = -I_inh/ti : amp
        '''
        self.group = NeuronGroup(self.num, self.eqs,
                                 threshold='v > Vth', reset='v = El', method='euler')

        self.group.namespace['l_mean'] = E_l_mean
        self.group.namespace['l_var'] = E_l_var
        self.group.r = 'i/N'
        self.group.v = 'El'
        self.group.namespace['I0'] = E_I0
        self.group.namespace['gain'] = E_gain
        self.group.namespace['I_rand'] = E_I_rand
        self.group.I_exc = 0*nA
        self.group.I_inh = 0*nA

        self.ceqs = '''
        w = w0*exp(-(position_round(r_pre,r_post))**2/(2*ls**2)) : amp
        '''
        # ls : strength
        # cp connection probability
        self.connection = Synapses(
            self.group, self.group, model=self.ceqs, on_pre='I_exc += w')
        self.connection.connect(condition='i != j', p=self.cp)
        self.connection.namespace['ls'] = 0.15
        self.connection.namespace['w0'] = E_w0


    def get_item(self):
        return self.group, self.connection

    def plot_connection(self):
        gain = 1000
        plt.scatter(self.connection.i, self.connection.j,
                    abs(self.connection.w_[:])*gain)
        plt.xlabel('Source neuron index')
        plt.ylabel('Target neuron index')
        plt.show()

    def plot_spike(self, timerange, monitor):
        plt.plot(monitor.t/second, monitor.i, marker='.',
                 color='orange', linestyle='None', ms=1)
        xlim(timerange[0], timerange[1])
        ylim(0, self.num)
        xlabel('Time (s)')
        ylabel('Neuron index')
        plt.show()

    def save_data(self, filename, monitor):

        if filename != '':
            filename = filename + '_'

        if type(monitor) == SpikeMonitor:
            data = monitor.get_states(['t', 'i'])
            with open('.\\data\\{}{}Spike.pickle'.format(filename, self.name), 'wb') as f:
                pickle.dump(data, f)
        else:
            data = monitor.get_states(['t', 'v', 'I_st'])
            with open('.\\data\\{}{}State.pickle'.format(filename, self.name), 'wb') as f:
                pickle.dump(data, f)


class Inh_group:

    def __init__(self, I_num=25, I_freq=5*Hz, I_I0=1*nA, I_theta=0, I_gain=2, I_I_rand=0*nA, I_w0=0.2*nA, I_wv=0.1*nA, I_cp=0.75):

        self.num = I_num
        self.cp = I_cp
        self.eqs = '''
        I_st = I0*(1+gain*sin(2*pi*freq*t+theta)):amp
        dv/dt = (El-v)/tau+(I_exc-I_inh+I_st+I_rand*(tau**0.5)*xi_exc)/Cm : volt 
        dI_exc/dt = -I_exc/te : amp
        dI_inh/dt = -I_inh/ti : amp
        '''
        self.group = NeuronGroup(self.num, self.eqs, threshold='v > Vth',
                                 reset='v = El', method='euler')
        self.group.v = 'El'
        self.group.namespace['I0'] = I_I0
        self.group.namespace['gain'] = I_gain
        self.group.namespace['I_rand'] = I_I_rand
        self.group.namespace['freq'] = I_freq
        self.group.namespace['theta'] = I_theta
        self.group.I_exc = 0*nA
        self.group.I_inh = 0*nA

        self.ceqs = '''
        w = w0 + wv/3*randn(): amp (constant over dt)
        '''
        self.connection = Synapses(
            self.group, self.group, model=self.ceqs, on_pre='I_inh += w')
        self.connection.connect(condition='i != j', p=self.cp)
        self.connection.namespace['w0'] = I_w0
        self.connection.namespace['wv'] = I_wv

    def get_item(self):
        return self.group, self.connection

    def plot_connection(self):
        gain = 1000
        plt.scatter(self.connection.i, self.connection.j,
                    abs(self.connection.w_[:])*gain)
        plt.xlabel('Source neuron index')
        plt.ylabel('Target neuron index')
        plt.show()

    def plot_spike(self, timerange, monitor):
        plt.plot(monitor.t/second, monitor.i, marker='.',
                 color='dodgerblue', linestyle='None', ms=1)
        xlim(timerange[0], timerange[1])
        ylim(0, self.num)
        xlabel('Time (s)')
        ylabel('Neuron index')
        plt.show()

    def save_data(self, filename, monitor):

        if filename != '':
            filename = filename + '_'

        if type(monitor) == SpikeMonitor:
            data = monitor.get_states(['t', 'i'])
            with open('.\\data\\{}{}Spike.pickle'.format(filename, self.name), 'wb') as f:
                pickle.dump(data, f)
        else:
            data = monitor.get_states(['t', 'v', 'I_st'])
            with open('.\\data\\{}{}State.pickle'.format(filename, self.name), 'wb') as f:
                pickle.dump(data, f)


def build_I_syn(G1, G2, S_w0, S_wv, S_cp, S_delay=0*ms):

    eqs = '''
    w = w0 + wv/3*randn(): amp (constant over dt)
    '''

    S = Synapses(G1, G2, model=eqs, on_pre='I_inh += w', delay=S_delay)
    S.connect(p=S_cp)
    
    S.namespace['w0'] = S_w0
    S.namespace['wv'] = S_wv

    return S


def build_E_syn(G1, G2, S_w0, S_wv, S_cp, S_delay=0*ms):

    eqs = '''
    w = w0 + wv/3*randn(): amp (constant over dt)
    '''

    S = Synapses(G1, G2, model=eqs, on_pre='I_exc += w', delay=S_delay)
    S.connect(p=S_cp)
    
    S.namespace['w0'] = S_w0
    S.namespace['wv'] = S_wv

    return S


class Block:

    def __init__(self, E_num=45, E_l_mean=0.03, E_l_var=0, E_I0=1*nA, E_gain=2, E_I_rand=0*nA, E_w0=1*nA, E_cp=0.1,
                 I_num=15, I_freq=5*Hz, I_I0=1*nA, I_theta=0, I_gain=2, I_I_rand=0*nA, I_w0=0.2*nA, I_wv=0.1*nA, I_cp=0.2,
                 IE_w0=0.2*nA, IE_wv=0.1*nA, IE_cp=0.5, IE_delay=0.2*ms, 
                 EI_w0=0.5*nA, EI_wv=0.25*nA, EI_cp=0.1, EI_delay=0.2*ms, x=0, y=0):

        self.exc = Exc_group(E_num, E_l_mean, E_l_var,
                             E_I0, E_gain, E_I_rand, E_w0, E_cp)
        self.inh = Inh_group(I_num, I_freq, I_I0, I_theta,
                             I_gain, I_I_rand, I_w0, I_wv, I_cp)

        self.i2e = build_I_syn(self.inh.group, self.exc.group, IE_w0, IE_wv, IE_cp, IE_delay)
        self.e2i = build_E_syn(self.exc.group, self.inh.group, EI_w0, EI_wv, EI_cp, EI_delay)

        self.x = x
        self.y = y

        self.Estate = StateMonitor(
            self.exc.group, ['v', 'I_st'], record=True)
        self.Istate = StateMonitor(
            self.inh.group, ['v', 'I_st'], record=True)
        self.Espike = SpikeMonitor(self.exc.group)
        self.Ispike = SpikeMonitor(self.inh.group)

    def get_item(self):
        return self.exc.get_item(), self.inh.get_item(), self.i2e, self.e2i, self.Estate, self.Istate, self.Espike, self.Ispike

    def plot_connection(self):
        gain = 1
        E_num = len(self.e2i.source)
        I_num = len(self.e2i.target)
        plt.figure(figsize=(8, 8))

        plt.subplot(5, 5, (1, 16))
        plt.scatter(self.i2e.i, self.i2e.j, abs(self.i2e.w/nA)*gain)
        # xlabel('Source neuron index')
        plt.ylabel('Target E')
        plt.xticks([])
        plt.xlim(-1, I_num)
        plt.ylim(-1, E_num)

        plt.subplot(5, 5, (2, 20))
        plt.scatter(self.exc.connection.i, self.exc.connection.j,
                    abs(self.exc.connection.w/nA)*gain)
        plt.xticks([])
        plt.yticks([])
        plt.xlim(-1, E_num)
        plt.ylim(-1, E_num)

        plt.subplot(5, 5, 21)
        plt.scatter(self.inh.connection.i, self.inh.connection.j,
                    abs(self.inh.connection.w/nA)*gain)
        plt.xlabel('Source I')
        plt.ylabel('Target I')
        plt.ylim(-1, I_num)
        plt.xlim(-1, I_num)

        plt.subplot(5, 5, (22, 25))
        plt.scatter(self.e2i.i, self.e2i.j, abs(self.e2i.w/nA)*gain)
        plt.xlabel('Source E')
        plt.yticks([])
        plt.xlim(-1, E_num)
        plt.ylim(-1, I_num)
        plt.subplots_adjust(left=None, bottom=None, right=None,
                            top=None, wspace=None, hspace=None)

        plt.show()

    def plot_spike(self, timerange=[]):

        plt.subplot(5, 1, (1, 4))
        plt.plot(self.Espike.t/second, np.array(self.Espike.i), marker='.',
                 color='orange', linestyle='None', ms=1)
        plt.ylabel('E index')
        plt.xticks([])
        ylim(0, self.exc.num)
        if timerange != []:
            xlim(timerange[0], timerange[1])

        plt.subplot(5, 1, 5)
        plt.plot(self.Ispike.t/second, np.array(self.Ispike.i), marker='.',
                 color='dodgerblue', linestyle='None', ms=1)
        plt.xlabel('Time (s)')
        plt.ylabel('I index')
        if timerange != []:
            xlim(timerange[0], timerange[1])
        ylim(0, self.inh.num)
        plt.show()

    def save_state(self, name, folder):

        datapath = '..\\data\\{}'.format(folder)
        os.makedirs(datapath, exist_ok=True)

        data = self.Estate.get_states(['t', 'v', 'I_st'])
        with open(datapath + '\\{}{}{}_EState.pickle'.format(name, self.x, self.y), 'wb') as f:
            pickle.dump(data, f)
        data = self.Istate.get_states(['t', 'v', 'I_st'])
        with open(datapath + '\\{}{}{}_IState.pickle'.format(name, self.x, self.y), 'wb') as f:
            pickle.dump(data, f)

    def save_spike(self, name, folder):

        datapath = '..\\data\\{}'.format(folder)
        os.makedirs(datapath, exist_ok=True)

        data = self.Espike.get_states(['t', 'i'])
        with open(datapath + '\\{}{}{}_ESpike.pickle'.format(name, self.x, self.y), 'wb') as f:
            pickle.dump(data, f)
        data = self.Ispike.get_states(['t', 'i'])
        with open(datapath + '\\{}{}{}_ISpike.pickle'.format(name, self.x, self.y), 'wb') as f:
            pickle.dump(data, f)

    def plot_average(self, window=0):

        Estate = self.Estate.get_states(['t', 'v'])
        Istate = self.Istate.get_states(['t', 'v'])
        t = Estate['t']/second
        data = np.hstack((Estate['v']/mV, Istate['v']/mV))
        # data = np.array(Estate['v']/mV)
        average_n = np.mean(data, axis=1)
        if window != 0:
            result = np.zeros([math.ceil(t[-1]/window)])
            for n in range(len(t)-1):
                index = math.floor(t[n]/window)
                result[index] += (average_n[n])*(t[n+1]-t[n])/window
            time = np.array(
                range(math.ceil(Estate['t'][-1]/window)))*window+window
        else:
            result = average_n
            time = t
        plt.plot(time, result)
        plt.xlabel('Time(s)')
        plt.ylabel('Averaged membrane potential')
        plt.show()
