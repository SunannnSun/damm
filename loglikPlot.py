import matplotlib.pyplot as plt
import numpy as np
import argparse, subprocess, os, sys, csv, random
from scipy.interpolate import interp1d
import matplotlib as mpl
mpl.rc('font',family='Times New Roman')

def moving_average(data, window_size):
    weights = np.repeat(1.0, window_size) / window_size
    smoothed = np.convolve(data, weights, mode='void')
    return smoothed


filepath = os.path.dirname(os.path.realpath(__file__))
logLik_1core        = np.genfromtxt(filepath + '/data/logLik2_1core_3.15.csv', dtype=float, delimiter=',')
logLik_2core        = np.genfromtxt(filepath + '/data/logLik2_2core_2.55.csv', dtype=float, delimiter=',')
logLik_4core        = np.genfromtxt(filepath + '/data/logLik2_4core_2.02.csv', dtype=float, delimiter=',')
logLik_8core        = np.genfromtxt(filepath + '/data/logLik2_8core_1.98.csv', dtype=float, delimiter=',')

# logLogLik = moving_average(logLogLik, 100)

terminal_time = 1000
start_time = 0.1

t1 = 32
dt1 = t1/logLik_1core.shape[0]
x1 = np.arange(start_time, terminal_time, dt1) # unit is time
y1 = np.zeros((x1.shape[0]))
y1[0:logLik_1core.shape[0]] = logLik_1core
y1[logLik_1core.shape[0]:] = logLik_1core[-1]


t2 = 21
dt2 = t2/logLik_2core.shape[0]
x2 = np.arange(start_time, terminal_time, dt2) # unit is time
y2 = np.zeros((x2.shape[0]))
y2[0:logLik_2core.shape[0]] = logLik_2core
y2[logLik_2core.shape[0]:] = logLik_2core[-1]



t3 = 11.2
dt3 = t3/logLik_4core.shape[0]
x3 = np.arange(start_time, terminal_time, dt3) # unit is time
y3 = np.zeros((x3.shape[0]))
y3[0:logLik_4core.shape[0]] = logLik_4core
y3[logLik_4core.shape[0]:] = logLik_4core[-1]




t4 = 10.5
dt4 = t4/logLik_8core.shape[0]
x4 = np.arange(start_time, terminal_time, dt4) # unit is time
y4 = np.zeros((x4.shape[0]))
y4[0:logLik_8core.shape[0]] = logLik_8core
y4[logLik_8core.shape[0]:] = logLik_8core[-1]




t5 = 999
dt5 = t5/logLik_2core.shape[0]
x5 = np.arange(start_time, terminal_time, dt5) # unit is time
y5 = np.zeros((x5.shape[0]))
y5[0:logLik_2core.shape[0]] = logLik_2core
y5[logLik_2core.shape[0]:] = logLik_2core[-1]


fig, ax = plt.subplots()
# ax.plot(x_smooth, y_smooth, label='DAMM', c='blue')

ax.plot(x1, y1, label='DAMM 1 Core ', c='blue')
ax.plot(x2, y2, label='DAMM 2 Cores', c='magenta')
ax.plot(x3, y3, label='DAMM 4 Cores', c='lime')
ax.plot(x4, y4, label='DAMM 8 Cores', c='cyan')
ax.plot(x5, y5, label='PC-GMM', c ='black')
ax.set_xscale('log')
legend= ax.legend()
legend.set_draggable(True)


# # Set the x-axis ticks to span from 10^-1 to 10^4
ax.set_xticks([ 0.1, 1, 10, 100, 1000])
ax.set_xticklabels([ r'$10^{-1}$' ,'1', '10', r'$10^{2}$' , r'$10^{3}$' ])
ax.set_xlabel('Time (secs) log scale', fontname='Times New Roman', fontsize=12)
ax.set_ylabel('Log Likelihood' , fontname='Times New Roman', fontsize=12)
ax.set_title('Computation Time', fontname='Times New Roman', fontsize=12)
ax.set_aspect('auto')


resolution_value = 1200
plt.savefig("myImage.png", format="png", dpi=resolution_value)

# # Show the plot
plt.show()

