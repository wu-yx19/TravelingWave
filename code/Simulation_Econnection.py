
from brian2 import *
import time
from buildgroup import *
import os
import numpy as np
os.system("cls")

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
    temp = r1 - r2 - floor(r1 - r2)
    sig = 2 * (temp > 0) - 1
    temp = temp * sig
    sig = temp > 0.5
    u = sig - temp * sig
    l = temp * (1 - sig)
    return u + l


print('Start')

tau = 6 * ms
El = -55 * mV
Vth = 0 * mV
Cm = 0.1875 * nF
te = 5 * ms
ti = 8.75 * ms

circuit = 2 * metre
velocity = 0.2 * metre / second
lap = 0.5
total_time = lap * circuit / velocity

defaultclock.dt = 0.05 * ms
position = TimedArray(np.linspace(0, lap, int(
    circuit / velocity / defaultclock.dt * lap)), dt=defaultclock.dt)

name = 'block'

scale = [12, 5]
rf_range = [0.05, 0.25]
II0_range = [0.6, 0.4]
EI0_range = [0.8, 0.5]

l_mean_temp = np.linspace(rf_range[0],rf_range[1],scale[0]+scale[1]-1)
II0_temp = np.linspace(II0_range[0],II0_range[1],scale[0])
EI0_temp = np.linspace(EI0_range[0],EI0_range[1],scale[0]+scale[1]-1)

S_delay = 8 * ms 
all_blocks = []
net = Network()

for i in range(scale[0]):
    blocks = []
    for j in range(scale[1]):
        block = Block(E_num=45, E_l_mean=l_mean_temp[i+j], E_l_var=0, E_I0=EI0_temp[i+j] * nA, 
        E_gain=2, E_I_rand=0 * nA, E_w0=0.3 * nA, E_cp=0.05,
        I_num=15, I_freq=8 * Hz, I_I0=II0_temp[i] * nA, I_theta=0, I_gain=2, 
        I_I_rand=0 * nA, I_w0=0.2 * nA, I_wv=0.1*nA, I_cp=0.25,
        IE_w0=0.2*nA, IE_wv=0.1*nA, IE_cp=0.5, IE_delay=0.2*ms, 
        EI_w0=0.5*nA, EI_wv=0.25*nA, EI_cp=0.1, EI_delay=0.2*ms, x=i, y=j)   
        net.add(block.get_item())
        blocks.append(block)
    all_blocks.append(blocks)


all_syns = []
for i in range(scale[0]):
    syns = []
    for j in range(scale[1]):
        syn = dict()
        if i + 1 < scale[0]:
            temp = build_E_syn(
                all_blocks[i][j].exc.group, all_blocks[i + 1][j].exc.group, 0.4 * nA, 0.32 * nA, 0.1, 10*ms)
            net.add(temp)
            syn['lg_f'] = temp
        if j + 1 < scale[1]:
            temp = build_E_syn(
                all_blocks[i][j].exc.group, all_blocks[i][j + 1].exc.group, 0.2 * nA, 0.16 * nA, 0.1, 10*ms)
            net.add(temp)
            syn['ts_f'] = temp
        if i > 0:
            temp = build_E_syn(
                all_blocks[i][j].exc.group, all_blocks[i - 1][j].exc.group, 0.1 * nA, 0.08 * nA, 0.05, 10*ms)
            net.add(temp)
            syn['lg_b'] = temp
        if j > 0 :
            temp = build_E_syn(
                all_blocks[i][j].exc.group, all_blocks[i][j - 1].exc.group, 0.2 * nA, 0.16 * nA, 0.05, 10*ms)
            net.add(temp)
            syn['ts_b'] = temp                
        syns.append(syn)
    all_syns.append(syns)  


folder = 'Econnection'
print(folder) 
start = time.time()

print('Simulation Start')
net.run(total_time)    
for i in range(scale[0]):
    for j in range(scale[1]):
        block = all_blocks[i][j]
        block.save_state(name, folder)  
        
print('Simulation End')
end = time.time()
print(end - start)
print('Saved!')

