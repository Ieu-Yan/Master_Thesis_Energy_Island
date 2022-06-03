

import numpy as np
import pandas as pd
import pypsa
import matplotlib.pyplot as plt
from matplotlib import dates


#%% Efficiencies


eff = pd.Series(index = ['H2 Plant','NH3 Plant'], dtype = float)
eff['H2 Plant'] = 141.8/(50*3.6)
eff['NH3 Plant'] = 23/(9.9*3.6)


#%% Plot of weekly power flows:
    
    
network = pypsa.Network('offgrid_2030_NH3_a.nc') 
network1 = pypsa.Network('offgrid_2030_H2_a.nc') 
network2 = pypsa.Network('offgrid_2030_NH3_w.nc') 
network3 = pypsa.Network('offgrid_2030_H2_w.nc') 


# January - NH3
fig, ax = plt.subplots(figsize = (12,8), dpi = 150)
ax.plot(network.links_t.p0['NH3 Plant']['2030-01-01 00:00':'2030-01-07 23:00'],
         c = 'black', label = 'Electricity Generation')
ax.plot(network.loads_t.p['Fuel Demand']['2030-01-01 00:00':'2030-01-07 23:00']/eff['NH3 Plant'], 
         c='black', linestyle='dashed', label='Electricity Demand')
ax.plot(network.generators_t.p['Offshore Wind']['2030-01-01 00:00':'2030-01-07 23:00'],
         c = 'blue', label = 'Offshore Wind')
ax.plot(network.generators_t.p['Solar PV']['2030-01-01 00:00':'2030-01-07 23:00'],
         c = 'darkorange', label = 'Solar PV')
ax.plot(network.links_t.p0['H2 Plant']['2030-01-01 00:00':'2030-01-07 23:00'],
         c = 'gold', label = 'Battery')
plt.rcParams.update({'font.size': 20})  
ax.set_xticks(['2030-01-0%s 12:00'%i for i in range(1,8)])
ax.set_xticklabels(['01-0%s'%i for i in range(1,8)])
ax.set_ylim(-150,5500)
ax.set_ylabel('Power [MW]')
ax.legend(loc = 'upper right')
plt.show()


# July - NH3
fig, ax = plt.subplots(figsize = (12,8), dpi = 150)
ax.plot(network.links_t.p0['NH3 Plant']['2030-07-01 00:00':'2030-07-07 23:00'],
         c = 'black', label = 'Electricity Generation')
ax.plot(network.loads_t.p['Fuel Demand']['2030-07-01 00:00':'2030-07-07 23:00']/eff['NH3 Plant'], 
         c='black', linestyle='dashed', label='Electricity Demand')
ax.plot(network.generators_t.p['Offshore Wind']['2030-07-01 00:00':'2030-07-07 23:00'],
         c = 'blue', label = 'Offshore Wind')
ax.plot(network.generators_t.p['Solar PV']['2030-07-01 00:00':'2030-07-07 23:00'],
         c = 'darkorange', label = 'Solar PV')
ax.plot(network.links_t.p0['H2 Plant']['2030-07-01 00:00':'2030-07-07 23:00'],
         c = 'gold', label = 'Battery')
plt.rcParams.update({'font.size': 20})
ax.set_xticks(['2030-07-0%s 12:00'%i for i in range(1,8)])
ax.set_xticklabels(['07-0%s'%i for i in range(1,8)])
ax.set_ylim(-150,5500)
ax.set_ylabel('Power [MW]')
ax.legend(loc = 'upper right')
plt.show()


# January - H2
fig, ax = plt.subplots(figsize = (12,8), dpi = 150)
ax.plot(network1.links_t.p0['H2 Plant']['2030-01-01 00:00':'2030-01-07 23:00'],
         c = 'black', label = 'Electricity Generation')
ax.plot(network1.loads_t.p['Fuel Demand']['2030-01-01 00:00':'2030-01-07 23:00']/eff['H2 Plant'], 
         c='black', linestyle='dashed', label='Electricity Demand')
ax.plot(network1.generators_t.p['Offshore Wind']['2030-01-01 00:00':'2030-01-07 23:00'],
         c = 'blue', label = 'Offshore Wind')
ax.plot(network1.generators_t.p['Solar PV']['2030-01-01 00:00':'2030-01-07 23:00'],
         c = 'darkorange', label = 'Solar PV')
ax.plot(network1.links_t.p0['NH3 Plant']['2030-01-01 00:00':'2030-01-07 23:00'],
         c = 'gold', label = 'Battery')
plt.rcParams.update({'font.size': 20})  
ax.set_xticks(['2030-01-0%s 12:00'%i for i in range(1,8)])
ax.set_xticklabels(['01-0%s'%i for i in range(1,8)])
ax.set_ylim(-150,5500)
ax.set_ylabel('Power [MW]')
ax.legend(loc = 'upper right')
plt.show()


# July - H2
fig, ax = plt.subplots(figsize = (12,8), dpi = 150)
ax.plot(network1.links_t.p0['H2 Plant']['2030-07-01 00:00':'2030-07-07 23:00'],
         c = 'black', label = 'Electricity Generation')
ax.plot(network1.loads_t.p['Fuel Demand']['2030-07-01 00:00':'2030-07-07 23:00']/eff['H2 Plant'], 
         c='black', linestyle='dashed', label='Electricity Demand')
ax.plot(network1.generators_t.p['Offshore Wind']['2030-07-01 00:00':'2030-07-07 23:00'],
         c = 'blue', label = 'Offshore Wind')
ax.plot(network1.generators_t.p['Solar PV']['2030-07-01 00:00':'2030-07-07 23:00'],
         c = 'darkorange', label = 'Solar PV')
ax.plot(network1.links_t.p0['NH3 Plant']['2030-07-01 00:00':'2030-07-07 23:00'],
         c = 'gold', label = 'Battery')
plt.rcParams.update({'font.size': 20})
ax.set_xticks(['2030-07-0%s 12:00'%i for i in range(1,8)])
ax.set_xticklabels(['07-0%s'%i for i in range(1,8)])
ax.set_ylim(-150,5500)
ax.set_ylabel('Power [MW]')
ax.legend(loc = 'upper right')
plt.show()


#%% Plot of hourly dispatch and duration curves


# Duration curve - NH3
t = pd.date_range('2030-01-01 00:00', '2030-12-31 23:00', freq = 'H')
x = np.arange(np.shape(t)[0])
plt.figure(figsize = (12,8), dpi = 150)
plt.plot(x, network.generators_t.p['Offshore Wind'].sort_values(ascending = False), c = 'blue', label = 'Offshore Wind')
plt.plot(x, network.generators_t.p['Solar PV'].sort_values(ascending = False), c = 'darkorange', label = 'Solar PV')
plt.plot(x, network.links_t.p0['NH3 Plant'].sort_values(ascending = False), c = 'black', label = 'Electricity Generation')
plt.plot(x, network.links_t.p0['H2 Plant'].sort_values(ascending = False), c = 'gold', label = 'Battery')
plt.rcParams.update({'font.size': 20})
plt.ylim(-150,5500)
plt.xlabel('Hours')
plt.ylabel('Power [MW]')
plt.legend(loc = 'upper right')
plt.show()

# Hourly dispatch - NH3
fig, ax=plt.subplots(1, 1, figsize=(12, 8), dpi = 150)
plt.plot(network.loads_t.p['Fuel Demand']/eff['NH3 Plant'],
         c = 'black', linestyle='dashed', label = 'Electricity Demand')
plt.plot(network.links_t.p0['NH3 Plant'],
         c = 'black', label = 'Electricity Generation')
plt.plot(network.generators_t.p['Offshore Wind'],
         c = 'blue', label = 'Offshore Wind')    
plt.plot(network.generators_t.p['Solar PV'],
         c = 'darkorange', label = 'Solar PV')
plt.plot(network.links_t.p0['H2 Plant'],
         c = 'gold', label = 'Battery')
plt.rcParams.update({'font.size': 20})
ax.xaxis.set_major_locator(dates.MonthLocator(bymonthday=11))
ax.xaxis.set_major_formatter(dates.DateFormatter('%b'))  
ax.set_xticks(ax.get_xticks()[:12])
plt.ylim(-150,5500)
plt.ylabel('Power [MW]')
plt.legend(loc = 'upper right')
plt.show()


# Duration curve - H2
t = pd.date_range('2030-01-01 00:00', '2030-12-31 23:00', freq = 'H')
x = np.arange(np.shape(t)[0])
plt.figure(figsize = (12,8), dpi = 150)
plt.plot(x, network1.generators_t.p['Offshore Wind'].sort_values(ascending = False), c = 'blue', label = 'Offshore Wind')
plt.plot(x, network1.generators_t.p['Solar PV'].sort_values(ascending = False), c = 'darkorange', label = 'Solar PV')
plt.plot(x, network1.links_t.p0['H2 Plant'].sort_values(ascending = False), c = 'black', label = 'Electricity Generation')
plt.plot(x, network1.links_t.p0['NH3 Plant'].sort_values(ascending = False), c = 'gold', label = 'Battery')
plt.rcParams.update({'font.size': 20})
plt.ylim(-150,5500)
plt.xlabel('Hours')
plt.ylabel('Power [MW]')
plt.legend(loc = 'upper right')
plt.show()

# Hourly dispatch - H2
fig, ax=plt.subplots(1, 1, figsize=(12, 8), dpi = 150)
plt.plot(network1.loads_t.p['Fuel Demand']/eff['NH3 Plant'],
         c = 'black', linestyle='dashed', label = 'Electricity Demand')
plt.plot(network1.links_t.p0['H2 Plant'],
         c = 'black', label = 'Electricity Generation')
plt.plot(network1.generators_t.p['Offshore Wind'],
         c = 'blue', label = 'Offshore Wind')    
plt.plot(network1.generators_t.p['Solar PV'],
         c = 'darkorange', label = 'Solar PV')
plt.plot(network1.links_t.p0['NH3 Plant'],
         c = 'gold', label = 'Battery')
plt.rcParams.update({'font.size': 20})
ax.xaxis.set_major_locator(dates.MonthLocator(bymonthday=11))
ax.xaxis.set_major_formatter(dates.DateFormatter('%b'))
ax.set_xticks(ax.get_xticks()[:12])  
plt.ylim(-150,5500)
plt.ylabel('Power [MW]')
plt.legend(loc = 'upper right')
plt.show()


#%% Electricity mix & cost-composition (Pie chart)
        

# Plots with NH3   
plt.figure(figsize = (3, 3), dpi = 150)
plt.pie([network.generators_t.p['Offshore Wind'].sum(),
         network.generators_t.p['Solar PV'].sum()], 
        colors = ['blue', 'darkorange'], 
        labels = ['Offshore Wind','Solar PV'], 
        autopct = '%1.1f%%')
plt.axis('equal')
plt.show()

labels = []
sizes = []
colors = []

labels.append('Offshore Wind')
sizes.append(network.generators.capital_cost['Offshore Wind']*network.generators.p_nom_opt['Offshore Wind']
             +network.generators.marginal_cost['Offshore Wind']*network.generators_t.p.sum()['Offshore Wind'])
colors.append('blue')
labels.append('Solar PV')
sizes.append(network.generators.capital_cost['Solar PV']*network.generators.p_nom_opt['Solar PV']
             +network.generators.marginal_cost['Solar PV']*network.generators_t.p.sum()['Solar PV'])
colors.append('darkorange')
labels.append('NH3 Plant')
sizes.append(network.links.capital_cost['NH3 Plant']*network.links.p_nom_opt['NH3 Plant']
             +network.links.marginal_cost['NH3 Plant']*network.links_t.p0.sum()['NH3 Plant'])
colors.append('green')

plt.figure(figsize = (3, 3), dpi = 150)
plt.pie(sizes, 
        colors = colors, 
        labels = labels, 
        autopct = '%1.1f%%')
plt.axis('equal')
plt.show()


# Plots with H2  
plt.figure(figsize = (3, 3), dpi = 150)
plt.pie([network1.generators_t.p['Offshore Wind'].sum(),
         network1.generators_t.p['Solar PV'].sum()], 
        colors = ['blue', 'darkorange'], 
        labels = ['Offshore Wind','Solar PV'], 
        autopct = '%1.1f%%')
plt.axis('equal')
plt.show()

labels = []
sizes = []
colors = []

labels.append('Offshore Wind')
sizes.append(network1.generators.capital_cost['Offshore Wind']*network1.generators.p_nom_opt['Offshore Wind']
             +network1.generators.marginal_cost['Offshore Wind']*network1.generators_t.p.sum()['Offshore Wind'])
colors.append('blue')
labels.append('Solar PV')
sizes.append(network1.generators.capital_cost['Solar PV']*network1.generators.p_nom_opt['Solar PV']
             +network1.generators.marginal_cost['Solar PV']*network1.generators_t.p.sum()['Solar PV'])
colors.append('darkorange')
labels.append('H2 Plant')
sizes.append(network1.links.capital_cost['H2 Plant']*network1.links.p_nom_opt['H2 Plant']
             +network1.links.marginal_cost['H2 Plant']*network1.links_t.p0.sum()['H2 Plant'])
colors.append('limegreen')

plt.figure(figsize = (3, 3), dpi = 150)
plt.pie(sizes, 
        colors = colors, 
        labels = labels, 
        autopct = '%1.1f%%')
plt.axis('equal')
plt.show()


#%% Electricity mix & cost-composition using only offshore wind for power generation (Pie chart)
        

# Plots with NH3  
labels = []
sizes = []
colors = []
 
labels.append('Offshore Wind')
sizes.append(network2.generators.capital_cost['Offshore Wind']*network2.generators.p_nom_opt['Offshore Wind']
             +network2.generators.marginal_cost['Offshore Wind']*network2.generators_t.p.sum()['Offshore Wind'])
colors.append('blue')
labels.append('NH3 Plant')
sizes.append(network2.links.capital_cost['NH3 Plant']*network2.links.p_nom_opt['NH3 Plant']
             +network2.links.marginal_cost['NH3 Plant']*network2.links_t.p0.sum()['NH3 Plant'])
colors.append('green')

plt.figure(figsize = (3, 3), dpi = 150)
plt.pie(sizes, 
        colors = colors, 
        labels = labels, 
        autopct = '%1.1f%%')
plt.axis('equal')
plt.show()


# Plots with H2  
labels = []
sizes = []
colors = []

labels.append('Offshore Wind')
sizes.append(network3.generators.capital_cost['Offshore Wind']*network3.generators.p_nom_opt['Offshore Wind']
             +network3.generators.marginal_cost['Offshore Wind']*network3.generators_t.p.sum()['Offshore Wind'])
colors.append('blue')
labels.append('H2 Plant')
sizes.append(network3.links.capital_cost['H2 Plant']*network3.links.p_nom_opt['H2 Plant']
             +network3.links.marginal_cost['H2 Plant']*network3.links_t.p0.sum()['H2 Plant'])
colors.append('limegreen')

plt.figure(figsize = (3, 3), dpi = 150)
plt.pie(sizes, 
        colors = colors, 
        labels = labels, 
        autopct = '%1.1f%%')
plt.axis('equal')
plt.show()


#%% Sensitivity analysis:
    
    
# Calculations:      
x = np.arange(-6,7)
xt = []
for i in range(np.shape(x)[0]):
    xt.append('%1.0f'%(x[i]*10)+'%')
x = 1+0.1*x

a = pd.read_csv(r'Sensitivity_nh3.csv', sep = ';', header = None) 
b = pd.read_csv(r'Sensitivity_h2.csv', sep = ';', header = None) 

  
# NH3
plt.figure(figsize = (12,8), dpi = 200)
plt.plot(x, a[1][0:13], c = 'blue', label = 'Offshore Wind')
plt.plot(x, a[2][0:13], c = 'darkorange', label = 'Solar PV')
plt.plot(x, a[3][0:13], c = 'green', label = 'NH3 Plant')
plt.plot(x, a[4][0:13], c = 'gold', label = 'Battery')
plt.xticks(x, xt)
plt.rcParams.update({'font.size': 19})
axes = plt.gca()
axes.xaxis.label.set_size(24)
axes.yaxis.label.set_size(24)
plt.xlabel('Cost Change of Offshore Wind')
plt.ylabel('Installed Capacity [MW]')
plt.xlim(0.4,1.6)
plt.ylim(-150,5500)
plt.legend(loc = 'upper right')

plt.figure(figsize = (12,8), dpi = 200)
plt.plot(x, a[1][14:27], c = 'blue', label = 'Offshore Wind')
plt.plot(x, a[2][14:27], c = 'darkorange', label = 'Solar PV')
plt.plot(x, a[3][14:27], c = 'green', label = 'NH3 Plant')
plt.plot(x, a[4][14:27], c = 'gold', label = 'Battery')
plt.xticks(x, xt)
plt.rcParams.update({'font.size': 19})
axes = plt.gca()
axes.xaxis.label.set_size(24)
axes.yaxis.label.set_size(24)
plt.xlabel('Cost Change of Solar PV')
plt.ylabel('Installed Capacity [MW]')
plt.xlim(0.4,1.6)
plt.ylim(-150,5500)
plt.legend(loc = 'upper right')

plt.figure(figsize = (12,8), dpi = 200)
plt.plot(x, a[1][28:41], c = 'blue', label = 'Offshore Wind')
plt.plot(x, a[2][28:41], c = 'darkorange', label = 'Solar PV')
plt.plot(x, a[3][28:41], c = 'green', label = 'NH3 Plant')
plt.plot(x, a[4][28:41], c = 'gold', label = 'Battery')
plt.xticks(x, xt)
plt.rcParams.update({'font.size': 19})
axes = plt.gca()
axes.xaxis.label.set_size(24)
axes.yaxis.label.set_size(24)
plt.xlabel('Cost Change of NH3 Plant')
plt.ylabel('Installed Capacity [MW]')
plt.xlim(0.4,1.6)
plt.ylim(-150,5500)
plt.legend(loc = 'upper right')

plt.figure(figsize = (12,8), dpi = 200)
plt.plot(x, a[1][42:55], c = 'blue', label = 'Offshore Wind')
plt.plot(x, a[2][42:55], c = 'darkorange', label = 'Solar PV')
plt.plot(x, a[3][42:55], c = 'green', label = 'NH3 Plant')
plt.plot(x, a[4][42:55], c = 'gold', label = 'Battery')
plt.xticks(x, xt)
plt.rcParams.update({'font.size': 19})
axes = plt.gca()
axes.xaxis.label.set_size(24)
axes.yaxis.label.set_size(24)
plt.xlabel('Cost Change of Battery')
plt.ylabel('Installed Capacity [MW]')
plt.xlim(0.4,1.6)
plt.ylim(-150,5500)
plt.legend(loc = 'upper right')


# H2
plt.figure(figsize = (12,8), dpi = 200)
plt.plot(x, b[1][0:13], c = 'blue', label = 'Offshore Wind')
plt.plot(x, b[2][0:13], c = 'darkorange', label = 'Solar PV')
plt.plot(x, b[3][0:13], c = 'limegreen', label = 'H2 Plant')
plt.plot(x, b[4][0:13], c = 'gold', label = 'Battery')
plt.xticks(x, xt)
plt.rcParams.update({'font.size': 19})
axes = plt.gca()
axes.xaxis.label.set_size(24)
axes.yaxis.label.set_size(24)
plt.xlabel('Cost Change of Offshore Wind')
plt.ylabel('Installed Capacity [MW]')
plt.xlim(0.4,1.6)
plt.ylim(-150,7000)
plt.legend(loc = 'upper right')

plt.figure(figsize = (12,8), dpi = 200)
plt.plot(x, b[1][14:27], c = 'blue', label = 'Offshore Wind')
plt.plot(x, b[2][14:27], c = 'darkorange', label = 'Solar PV')
plt.plot(x, b[3][14:27], c = 'limegreen', label = 'H2 Plant')
plt.plot(x, b[4][14:27], c = 'gold', label = 'Battery')
plt.xticks(x, xt)
plt.rcParams.update({'font.size': 19})
axes = plt.gca()
axes.xaxis.label.set_size(24)
axes.yaxis.label.set_size(24)
plt.xlabel('Cost Change of Solar PV')
plt.ylabel('Installed Capacity [MW]')
plt.xlim(0.4,1.6)
plt.ylim(-150,7000)
plt.legend(loc = 'upper right')

plt.figure(figsize = (12,8), dpi = 200)
plt.plot(x, b[1][28:41], c = 'blue', label = 'Offshore Wind')
plt.plot(x, b[2][28:41], c = 'darkorange', label = 'Solar PV')
plt.plot(x, b[3][28:41], c = 'limegreen', label = 'H2 Plant')
plt.plot(x, b[4][28:41], c = 'gold', label = 'Battery')
plt.xticks(x, xt)
plt.rcParams.update({'font.size': 19})
axes = plt.gca()
axes.xaxis.label.set_size(24)
axes.yaxis.label.set_size(24)
plt.xlabel('Cost Change of H2 Plant')
plt.ylabel('Installed Capacity [MW]')
plt.xlim(0.4,1.6)
plt.ylim(-150,7000)
plt.legend(loc = 'upper right')

plt.figure(figsize = (12,8), dpi = 200)
plt.plot(x, b[1][42:55], c = 'blue', label = 'Offshore Wind')
plt.plot(x, b[2][42:55], c = 'darkorange', label = 'Solar PV')
plt.plot(x, b[3][42:55], c = 'limegreen', label = 'H2 Plant')
plt.plot(x, b[4][42:55], c = 'gold', label = 'Battery')
plt.xticks(x, xt)
plt.rcParams.update({'font.size': 19})
axes = plt.gca()
axes.xaxis.label.set_size(24)
axes.yaxis.label.set_size(24)
plt.xlabel('Cost Change of Battery')
plt.ylabel('Installed Capacity [MW]')
plt.xlim(0.4,1.6)
plt.ylim(-150,7000)
plt.legend(loc = 'upper right')


#%% Installed capacities 2030-2050


# System with offshore wind + solar PV + battery + fuel plant
# Capacities of NH3 [MW] (Greenfield - Fixed)
w_nh3 = [3600, 3600, 3500]
s_nh3 = [1700, 1900, 2000]
p_nh3 = [3300, 3300, 3400]
# Capacities of H2 [MW]
w_h2 = [3000, 3000, 2900]
s_h2 = [700, 800, 1000]
p_h2 = [2900, 2900, 2900]

# Capacities of NH3 [MW] (Greenfield - Increased)
w_nh3i = [3600, 4100, 4800]
s_nh3i = [1700, 2200, 2600]
p_nh3i = [3300, 3900, 4600]
# Capacities of H2 [MW]
w_h2i = [3000, 3400, 4000]
s_h2i = [700, 900, 1300]
p_h2i = [2900, 3400, 3900]


# NH3 Plant (Fixed)
X = np.arange(3)
index = np.arange(3)
plt.figure(figsize = (12,8), dpi = 150)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, w_nh3, color = 'blue', width = 0.15, label = 'Offshore Wind')
ax.bar(X + 0.15, s_nh3, color = 'darkorange', width = 0.15, label = 'Solar PV')
ax.bar(X + 0.30, p_nh3, color = 'green', width = 0.15, label = 'NH3 Plant')
plt.xticks(index + 0.15, ('2030', '2040', '2050'))
plt.tick_params(axis=u'both', which=u'both',length=0)
plt.ylim(0,7700)
plt.ylabel('Installed Capacity [MW]')
plt.grid(axis="y",linewidth = "0.5")
plt.legend()

# H2 Plant (Fixed)
X = np.arange(3)
index = np.arange(3)
plt.figure(figsize = (12,8), dpi = 150)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, w_h2, color = 'blue', width = 0.15, label = 'Offshore Wind')
ax.bar(X + 0.15, s_h2, color = 'darkorange', width = 0.15, label = 'Solar PV')
ax.bar(X + 0.30, p_h2, color = 'limegreen', width = 0.15, label = 'H2 Plant')
plt.xticks(index + 0.15, ('2030', '2040', '2050'))
plt.tick_params(axis=u'both', which=u'both',length=0)
plt.ylim(0,7700)
plt.ylabel('Installed Capacity [MW]')
plt.grid(axis="y",linewidth = "0.5")
plt.legend()


# NH3 Plant (Increased)
X = np.arange(3)
index = np.arange(3)
plt.figure(figsize = (12,8), dpi = 150)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, w_nh3i, color = 'blue', width = 0.15, label = 'Offshore Wind')
ax.bar(X + 0.15, s_nh3i, color = 'darkorange', width = 0.15, label = 'Solar PV')
ax.bar(X + 0.30, p_nh3i, color = 'green', width = 0.15, label = 'NH3 Plant')
plt.xticks(index + 0.15, ('2030', '2040', '2050'))
plt.tick_params(axis=u'both', which=u'both',length=0)
plt.ylim(0,7700)
plt.ylabel('Installed Capacity [MW]')
plt.grid(axis="y",linewidth = "0.5")
plt.legend()

# H2 Plant (Increased)
X = np.arange(3)
index = np.arange(3)
plt.figure(figsize = (12,8), dpi = 150)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, w_h2i, color = 'blue', width = 0.15, label = 'Offshore Wind')
ax.bar(X + 0.15, s_h2i, color = 'darkorange', width = 0.15, label = 'Solar PV')
ax.bar(X + 0.30, p_h2i, color = 'limegreen', width = 0.15, label = 'H2 Plant')
plt.xticks(index + 0.15, ('2030', '2040', '2050'))
plt.tick_params(axis=u'both', which=u'both',length=0)
plt.ylim(0,7700)
plt.ylabel('Installed Capacity [MW]')
plt.grid(axis="y",linewidth = "0.5")
plt.legend()


# System with offshore wind + battery + fuel plant
# Capacities of NH3 [MW] (Greenfield - Fixed)
w_nh3 = [3900, 3900, 3800]
p_nh3 = [3400, 3500, 3500]
# Capacities of H2 [MW]
w_h2 = [3100, 3100, 3100]
p_h2 = [3000, 3000, 3000]

# Capacities of NH3 [MW] (Greenfield - Increased)
w_nh3i = [3900, 4500, 5200]
p_nh3i = [3400, 4000, 4700]
# Capacities of H2 [MW]
w_h2i = [3100, 3600, 4200]
p_h2i = [3000, 3500, 4100]


# NH3 Plant (Fixed)
X = np.arange(3)
plt.figure(figsize = (12,8), dpi = 150)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, w_nh3, color = 'blue', width = 0.15, label = 'Offshore Wind')
ax.bar(X + 0.15, p_nh3, color = 'green', width = 0.15, label = 'NH3 Plant')
plt.xticks(index + 0.075, ('2030', '2040', '2050'))
plt.tick_params(axis=u'both', which=u'both',length=0)
plt.ylim(0,7200)
plt.ylabel('Installed Capacity [MW]')
plt.grid(axis="y",linewidth = "0.5")
plt.legend()

# H2 Plant (Fixed)
X = np.arange(3)
plt.figure(figsize = (12,8), dpi = 150)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, w_h2, color = 'blue', width = 0.15, label = 'Offshore Wind')
ax.bar(X + 0.15, p_h2, color = 'limegreen', width = 0.15, label = 'H2 Plant')
plt.xticks(index + 0.075, ('2030', '2040', '2050'))
plt.tick_params(axis=u'both', which=u'both',length=0)
plt.ylim(0,7200)
plt.ylabel('Installed Capacity [MW]')
plt.grid(axis="y",linewidth = "0.5")
plt.legend()


# NH3 Plant (Increased)
X = np.arange(3)
plt.figure(figsize = (12,8), dpi = 150)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, w_nh3i, color = 'blue', width = 0.15, label = 'Offshore Wind')
ax.bar(X + 0.15, p_nh3i, color = 'green', width = 0.15, label = 'NH3 Plant')
plt.xticks(index + 0.075, ('2030', '2040', '2050'))
plt.tick_params(axis=u'both', which=u'both',length=0)
plt.ylim(0,7200)
plt.ylabel('Installed Capacity [MW]')
plt.grid(axis="y",linewidth = "0.5")
plt.legend()

# H2 Plant (Increased)
X = np.arange(3)
plt.figure(figsize = (12,8), dpi = 150)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, w_h2i, color = 'blue', width = 0.15, label = 'Offshore Wind')
ax.bar(X + 0.15, p_h2i, color = 'limegreen', width = 0.15, label = 'H2 Plant')
plt.xticks(index + 0.075, ('2030', '2040', '2050'))
plt.tick_params(axis=u'both', which=u'both',length=0)
plt.ylim(0,7200)
plt.ylabel('Installed Capacity [MW]')
plt.grid(axis="y",linewidth = "0.5")
plt.legend()


#%% Levelized cost of fuel (2030-2050)


LFC_nh3 = [94, 84, 75]
LFC_h2 = [57, 50, 48]
LFC_s_nh3 = [246, 220, 196]
LFC_s_h2 = [103, 91, 87]

# LFC
X = np.arange(3)
plt.figure(figsize = (12,6), dpi = 150)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, LFC_nh3, color = 'green', width = 0.25, label = 'NH3')
ax.bar(X + 0.25, LFC_h2, color = 'limegreen', width = 0.25, label = 'H2')
plt.xticks(index + 0.123, ('2030', '2040', '2050'))
plt.ylim(0,270)
plt.ylabel('Cost per energy produced [EUR/MWh]')
plt.title('Levelized cost of fuel')
plt.grid(axis="y",linewidth = "0.5")
plt.legend()

# LFC - Ship
X = np.arange(3)
plt.figure(figsize = (12,6), dpi = 150)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, LFC_s_nh3, color = 'green', width = 0.25, label = 'NH3')
ax.bar(X + 0.25, LFC_s_h2, color = 'limegreen', width = 0.25, label = 'H2')
plt.xticks(index + 0.123, ('2030', '2040', '2050'))
plt.ylim(0,270)
plt.ylabel('Cost per energy produced [EUR/MWh]')
plt.title('Levelized cost of fuel - Ship')
plt.grid(axis="y",linewidth = "0.5")
plt.legend()


#%% Levelized cost of fuel - Configurations (Appendix)


# Cost excluding battery [NH3, H2]
LFC_w = np.array([94, 57]) # Only wind
LFC_s = np.array([172, 80]) # Only solar
LFC_ws = np.array([94, 57]) # Wind + Solar

# Cost including battery [NH3, H2]
LFC_wb =np.array([94, 57])
LFC_sb =np.array([169, 80])
LFC_wsb = np.array([94, 57])

# Directly applied on ship [NH3, H2]
LFCs_w = np.array([248, 103]) # Only wind
LFCs_s = np.array([453, 146]) # Only solar
LFCs_ws = np.array([246, 103]) # Wind + Solar

# Directly applied on ship including battery [NH3, H2]
LFCs_wb =np.array([248, 103])
LFCs_sb =np.array([445, 146])
LFCs_wsb = np.array([246, 103])
 
# Bar chart (Excluding battery)
plt.figure(figsize = (12,6), dpi = 150)
data = [[LFC_w[0], LFC_s[0], LFC_ws[0]],
        [LFC_w[1], LFC_s[1], LFC_ws[1]]]
plt.rcParams.update({'font.size': 13.5})
index = np.arange(3)
fig, ax = plt.subplots()
bar_width = 0.25
opacity = 0.8

rects1 = plt.bar(index, data[0], bar_width,
alpha=opacity,
color='green',
label='NH3')

rects2 = plt.bar(index + bar_width, data[1], bar_width,
alpha=opacity,
color='limegreen',
label='H2')

plt.ylim(0,185)
plt.ylabel('Cost per energy produced [EUR/MWh]')
plt.title('Levelized cost of fuel')
plt.xticks(index + 0.12, ('Wind', 'Solar', 'Wind + Solar'))
plt.tick_params(axis=u'both', which=u'both',length=0)
plt.legend()
plt.tight_layout()
plt.grid(axis="y",linewidth = "0.5")
plt.show()


# Bar chart (Including battery)
plt.figure(figsize = (12,6), dpi = 150)
data = [[LFC_wb[0], LFC_sb[0], LFC_wsb[0]],
        [LFC_wb[1], LFC_sb[1], LFC_wsb[1]]]
index = np.arange(3)
fig, ax = plt.subplots()
bar_width = 0.25
opacity = 0.8

rects1 = plt.bar(index, data[0], bar_width,
alpha=opacity,
color='green',
label='NH3')

rects2 = plt.bar(index + bar_width, data[1], bar_width,
alpha=opacity,
color='limegreen',
label='H2')
plt.ylim(0,185)
plt.ylabel('Cost per energy produced [EUR/MWh]')
plt.title('Levelized cost of fuel')
plt.xticks(index + 0.12, ('Wind', 'Solar', 'Wind + Solar'))
plt.tick_params(axis=u'both', which=u'both',length=0)
plt.legend()
plt.tight_layout()
plt.grid(axis="y",linewidth = "0.5")
plt.show()

# Bar chart [Applied directly on ship]
plt.figure(figsize = (12,6), dpi = 150)
data = [[LFCs_w[0], LFCs_s[0], LFCs_ws[0]],
        [LFCs_w[1], LFCs_s[1], LFCs_ws[1]]]
index = np.arange(3)
fig, ax = plt.subplots()
bar_width = 0.25
opacity = 0.8

rects1 = plt.bar(index, data[0], bar_width,
alpha=opacity,
color='green',
label='NH3')

rects2 = plt.bar(index + bar_width, data[1], bar_width,
alpha=opacity,
color='limegreen',
label='H2')
plt.ylim(0,500)
plt.ylabel('Cost per energy produced [EUR/MWh]')
plt.title('Levelized cost of fuel - Ship')
plt.xticks(index + 0.12, ('Wind', 'Solar', 'Wind + Solar'))
plt.tick_params(axis=u'both', which=u'both',length=0)
plt.legend()
plt.tight_layout()
plt.grid(axis="y",linewidth = "0.5")
plt.show()

# Bar char (Including battery)
plt.figure(figsize = (12,6), dpi = 150)
data = [[LFCs_wb[0], LFCs_sb[0], LFCs_wsb[0]],
        [LFCs_wb[1], LFCs_sb[1], LFCs_wsb[1]]]
index = np.arange(3)
fig, ax = plt.subplots()
bar_width = 0.25
opacity = 0.8

rects1 = plt.bar(index, data[0], bar_width,
alpha=opacity,
color='green',
label='NH3')

rects2 = plt.bar(index + bar_width, data[1], bar_width,
alpha=opacity,
color='limegreen',
label='H2')
plt.tick_params(axis=u'both', which=u'both',length=0)
plt.ylim(0,500)
plt.ylabel('Cost per energy produced [EUR/MWh]')
plt.title('Levelized cost of fuel - Ship')
plt.xticks(index + 0.12, ('Wind', 'Solar', 'Wind + Solar'))
plt.legend()
plt.tight_layout()
plt.grid(axis="y",linewidth = "0.5")
plt.show()