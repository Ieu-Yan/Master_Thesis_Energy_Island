

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker


#%% Data inputs


# Investment cost
I_ofw = [1.93e6, 1.81e6, 1.78e6]
I_onw = [1.04e6, 9.8e5, 9.6e5]
I_pv = [3.8e5, 3.3e5, 3e5]
I_h2 = [4.5e5, 3e5, 2.5e5]
I_nh3 = [1.3e6, 1.1e6, 8e5]
I_b = [1.01e6, 6.64e5, 5.1e5]  

# FOM
F_ofw = [36053, 33169, 32448]
F_onw = [12600, 11592, 11340]
F_pv = [7250, 6625, 6250]
F_h2 = [9000, 6000, 5000]
F_nh3 = [39000, 32000, 24000]
F_b = [540, 540, 540]  

# VOM
V_ofw = [2.7, 2.5, 2.4]
V_onw = [1.35, 1.24, 1.22]
V_pv = [0.01, 0.01, 0.01]
V_h2 = [0.01, 0.01, 0.01]
V_nh3 = [0.02, 0.02, 0.02]
V_b = [1.8, 1.7, 1.6] 

# Expected lifetime
n_ofw = [30, 30, 30]
n_onw = [30, 30, 30]
n_pv = [40, 40, 40]
n_h2 = [30, 32, 35]
n_nh3 = [30, 30, 30]
n_b = [25, 30, 30] 


#%% Plots


# Plot of investment cost
index = np.arange(3)
X = np.arange(3)
fig = plt.figure(figsize = (12,8), dpi = 150)
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, I_ofw, color = 'blue', width = 0.1, label = 'Offshore Wind')
ax.bar(X + 0.1, I_onw, color = 'deepskyblue', width = 0.1, label = 'Onshore Wind')
ax.bar(X + 0.2, I_pv, color = 'darkorange', width = 0.1, label = 'Solar PV')
ax.bar(X + 0.3, I_b, color = 'gold', width = 0.1, label = 'Battery')
ax.bar(X + 0.4, I_nh3, color = 'green', width = 0.1, label = 'NH3')
ax.bar(X + 0.5, I_h2, color = 'limegreen', width = 0.1, label = 'H2')
plt.rcParams.update({'font.size': 20})
plt.xticks(index + 0.25, ('2030', '2040', '2050'))
plt.tick_params(axis=u'both', which=u'both',length=0) 
plt.xlim(-0.2,2.9)
plt.ylabel('Investment cost [€/MW]')
yfmt = ticker.ScalarFormatter(useMathText=True)
yfmt.set_powerlimits((0, 0))
ax.yaxis.set_major_formatter(yfmt)
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.get_yaxis().get_offset_text().set_visible(False)
ax_max = max(ax.get_yticks())
exponent_axis = np.floor(np.log10(ax_max)).astype(int)
ax.annotate(r'$\times$10$^{%i}$'%(exponent_axis),
             xy=(0, 1.005), xycoords='axes fraction')
plt.legend()

# Plot of FOM
X = np.arange(3)
fig = plt.figure(figsize = (12,8), dpi = 150)
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, F_ofw, color = 'blue', width = 0.1, label = 'Offshore Wind')
ax.bar(X + 0.1, F_onw, color = 'deepskyblue', width = 0.1, label = 'Onshore Wind')
ax.bar(X + 0.2, F_pv, color = 'darkorange', width = 0.1, label = 'Solar PV')
ax.bar(X + 0.3, F_b, color = 'gold', width = 0.1, label = 'Battery')
ax.bar(X + 0.4, F_nh3, color = 'green', width = 0.1, label = 'NH3')
ax.bar(X + 0.5, F_h2, color = 'limegreen', width = 0.1, label = 'H2')
plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
plt.xticks(index + 0.25, ('2030', '2040', '2050'))
plt.tick_params(axis=u'both', which=u'both',length=0) 
plt.xlim(-0.2,2.9)
plt.ylabel('FOM cost [€/MW/year]')   
yfmt = ticker.ScalarFormatter(useMathText=True)
yfmt.set_powerlimits((3, 4))
ax.yaxis.set_major_formatter(yfmt)
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.get_yaxis().get_offset_text().set_visible(False)
ax_max = max(ax.get_yticks())
exponent_axis = np.floor(np.log10(ax_max)).astype(int)
ax.annotate(r'$\times$10$^{%i}$'%(exponent_axis), xy=(0, 1.005), xycoords='axes fraction')
plt.legend()

# Plot of VOM
X = np.arange(3)
fig = plt.figure(figsize = (12,8), dpi = 150)
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, V_ofw, color = 'blue', width = 0.1, label = 'Offshore Wind')
ax.bar(X + 0.1, V_onw, color = 'deepskyblue', width = 0.1, label = 'Onshore Wind')
ax.bar(X + 0.2, V_pv, color = 'darkorange', width = 0.1, label = 'Solar PV')
ax.bar(X + 0.3, V_b, color = 'gold', width = 0.1, label = 'Battery')
ax.bar(X + 0.4, V_nh3, color = 'green', width = 0.1, label = 'NH3')
ax.bar(X + 0.5, V_h2, color = 'limegreen', width = 0.1, label = 'H2')
plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
plt.xticks(index + 0.25, ('2030', '2040', '2050'))
plt.tick_params(axis=u'both', which=u'both',length=0) 
plt.xlim(-0.2,2.9)
plt.ylabel('VOM cost [€/MWh]')   
plt.legend()

# Plot of lifetime
X = np.arange(3)
fig = plt.figure(figsize = (12,8), dpi = 150)
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, n_ofw, color = 'blue', width = 0.1, label = 'Offshore Wind')
ax.bar(X + 0.1, n_onw, color = 'deepskyblue', width = 0.1, label = 'Onshore Wind')
ax.bar(X + 0.2, n_pv, color = 'darkorange', width = 0.1, label = 'Solar PV')
ax.bar(X + 0.3, n_b, color = 'gold', width = 0.1, label = 'Battery')
ax.bar(X + 0.4, n_nh3, color = 'green', width = 0.1, label = 'NH3')
ax.bar(X + 0.5, n_h2, color = 'limegreen', width = 0.1, label = 'H2')
plt.xticks(index + 0.25, ('2030', '2040', '2050'))
plt.tick_params(axis=u'both', which=u'both',length=0) 
plt.xlim(-0.2,3.2)
plt.ylim(0,55)
plt.ylabel('Lifetime [years]')   
plt.legend()

