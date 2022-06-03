

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pypsa


#%% Efficiencies


eff = pd.Series(index = ['H2 Plant','NH3 Plant'], dtype = float)
eff['H2 Plant'] = 141.8/(50*3.6)
eff['NH3 Plant'] = 23/(9.9*3.6)

    
#%% Read network files:  
    
    
locvar = locals()
for i in [2030,2040,2050]:
    for j in ['H2','NH3']:
        for k in ['DK Wind','DK Solar','DE Wind','DE Solar','NO Wind','NO Solar','NL Wind','NL Solar','GB Wind','GB Solar']:
            locvar['network_%d_%s_%s'%(i,j,k)] = pypsa.Network('results/%d_%s_%s.nc'%(i,j,k))  
                
lfc = []
lfc_t = []
p_opt = []
p_w = []
p_s = []
p_fp = []
p_ft = []
p_b = []
cc_pp = []
p_trans = []

for year in [2030,2040,2050]:
    for fuel in ['H2','NH3']:
        for location in ['DK Wind','DK Solar','DE Wind','DE Solar','NO Wind','NO Solar','NL Wind','NL Solar','GB Wind','GB Solar']:
            lfc.append(int(locvar['network_%d_%s_%s'%(year,fuel,location)].buses_t.marginal_price[fuel].mean()))
            lfc_t.append(int(sum((locvar['network_%d_%s_%s'%(year,fuel,location)].buses_t.marginal_price['Truck'])*(locvar['network_%d_%s_%s'%(year,fuel,location)].links_t.p0['%s Transport'%(fuel)]))/sum((locvar['network_%d_%s_%s'%(year,fuel,location)].links_t.p0['%s Transport'%(fuel)]))))
            p_w.append(int(locvar['network_%d_%s_%s'%(year,fuel,location)].generators.p_nom_opt['Onshore Wind']))
            p_s.append(int(locvar['network_%d_%s_%s'%(year,fuel,location)].generators.p_nom_opt['Solar PV']))
            p_fp.append(int(locvar['network_%d_%s_%s'%(year,fuel,location)].links.p_nom_opt['%s Plant'%(fuel)]))
            p_ft.append(int(locvar['network_%d_%s_%s'%(year,fuel,location)].links.p_nom_opt['%s Transport'%(fuel)]))
            p_b.append(int(locvar['network_%d_%s_%s'%(year,fuel,location)].storage_units.p_nom_opt['Battery']))
p_opt = pd.DataFrame(np.array([p_w, p_s, p_fp, p_ft, p_b]).T.tolist(), columns=['Onshore Wind', 'Solar PV', '%s Plant'%(fuel),'%s Transport'%(fuel), 'Battery'])


#%% Levelized cost of fuel at different locations   


# Data input    
# H2
DK_w = [lfc[0], lfc[20], lfc[40]]
DK_s = [lfc[1], lfc[21], lfc[41]]
DE_w = [lfc[2], lfc[22], lfc[42]]
DE_s = [lfc[3], lfc[23], lfc[43]]
NO_w = [lfc[4], lfc[24], lfc[44]]
NO_s = [lfc[5], lfc[25], lfc[45]]
NL_w = [lfc[6], lfc[26], lfc[46]]
NL_s = [lfc[7], lfc[27], lfc[47]]
GB_w = [lfc[8], lfc[28], lfc[48]]
GB_s = [lfc[9], lfc[29], lfc[49]]


# NH3
DK_w1 = [lfc[10], lfc[30], lfc[50]]
DK_s1 = [lfc[11], lfc[31], lfc[51]]
DE_w1 = [lfc[12], lfc[32], lfc[52]]
DE_s1 = [lfc[13], lfc[33], lfc[53]]
NO_w1 = [lfc[14], lfc[34], lfc[54]]
NO_s1 = [lfc[15], lfc[35], lfc[55]]
NL_w1 = [lfc[16], lfc[36], lfc[56]]
NL_s1 = [lfc[17], lfc[37], lfc[57]]
GB_w1 = [lfc[18], lfc[38], lfc[58]]
GB_s1 = [lfc[19], lfc[39], lfc[59]]


# H2 - Ship
DK_ws = DK_w/eff['H2 Plant']
DK_ss = DK_s/eff['H2 Plant']
DE_ws = DE_w/eff['H2 Plant']
DE_ss = DE_s/eff['H2 Plant']
NO_ws = NO_w/eff['H2 Plant']
NO_ss = NO_s/eff['H2 Plant']
NL_ws = NL_w/eff['H2 Plant']
NL_ss = NL_s/eff['H2 Plant']
GB_ws = GB_w/eff['H2 Plant']
GB_ss = GB_s/eff['H2 Plant']


# NH3 - Ship
DK_ws1 = DK_w1/eff['NH3 Plant']
DK_ss1 = DK_s1/eff['NH3 Plant']
DE_ws1 = DE_w1/eff['NH3 Plant']
DE_ss1 = DE_s1/eff['NH3 Plant']
NO_ws1 = NO_w1/eff['NH3 Plant']
NO_ss1 = NO_s1/eff['NH3 Plant']
NL_ws1 = NL_w1/eff['NH3 Plant']
NL_ss1 = NL_s1/eff['NH3 Plant']
GB_ws1 = GB_w1/eff['NH3 Plant']
GB_ss1 = GB_s1/eff['NH3 Plant']


# Plot of levelized cost of fuels
index = np.arange(3)
X = np.arange(3)
plt.figure(figsize = (12,8), dpi = 150)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, DK_w, width = 0.05, label = 'DK - Wind')
ax.bar(X + 0.05, DK_s, width = 0.05, label = 'DK - Solar')
ax.bar(X + 0.10, DE_w, width = 0.05, label = 'DE - Wind')
ax.bar(X + 0.15, DE_s, width = 0.05, label = 'DE - Solar')
ax.bar(X + 0.20, NO_w, width = 0.05, label = 'NO - Wind')
ax.bar(X + 0.25, NO_s, width = 0.05, label = 'NO - Solar')
ax.bar(X + 0.30, NL_w, width = 0.05, label = 'NL - Wind')
ax.bar(X + 0.35, NL_s, width = 0.05, label = 'NL - Solar')
ax.bar(X + 0.40, GB_w, width = 0.05, label = 'GB - Wind')
ax.bar(X + 0.45, GB_s, width = 0.05, label = 'GB - Solar')
plt.xticks(index + 0.225, ('2030', '2040', '2050'))
plt.ylim(0,170)
plt.ylabel('Cost per energy produced [EUR/MWh]')
plt.title('Levelized cost of fuel - H2')
plt.grid(axis="y",linewidth = "0.5")
plt.rcParams.update({'font.size': 16})
plt.tick_params(axis=u'both', which=u'both',length=0) 
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


index = np.arange(3)
X = np.arange(3)
plt.figure(figsize = (12,8), dpi = 150)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, DK_w1, width = 0.05, label = 'DK - Wind')
ax.bar(X + 0.05, DK_s1, width = 0.05, label = 'DK - Solar')
ax.bar(X + 0.10, DE_w1, width = 0.05, label = 'DE - Wind')
ax.bar(X + 0.15, DE_s1, width = 0.05, label = 'DE - Solar')
ax.bar(X + 0.20, NO_w1, width = 0.05, label = 'NO - Wind')
ax.bar(X + 0.25, NO_s1, width = 0.05, label = 'NO - Solar')
ax.bar(X + 0.30, NL_w1, width = 0.05, label = 'NL - Wind')
ax.bar(X + 0.35, NL_s1, width = 0.05, label = 'NL - Solar')
ax.bar(X + 0.40, GB_w1, width = 0.05, label = 'GB - Wind')
ax.bar(X + 0.45, GB_s1, width = 0.05, label = 'GB - Solar')
plt.xticks(index + 0.225, ('2030', '2040', '2050'))
plt.ylim(0,170)
plt.ylabel('Cost per energy produced [EUR/MWh]')
plt.title('Levelized cost of fuel - NH3')
plt.grid(axis="y",linewidth = "0.5")
plt.rcParams.update({'font.size': 16})
plt.tick_params(axis=u'both', which=u'both',length=0) 
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


# Plot of levelized cost of fuels (Ship)
index = np.arange(3)
X = np.arange(3)
plt.figure(figsize = (12,8), dpi = 150)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, DK_ws, width = 0.05, label = 'DK - Wind')
ax.bar(X + 0.05, DK_ss, width = 0.05, label = 'DK - Solar')
ax.bar(X + 0.10, DE_ws, width = 0.05, label = 'DE - Wind')
ax.bar(X + 0.15, DE_ss, width = 0.05, label = 'DE - Solar')
ax.bar(X + 0.20, NO_ws, width = 0.05, label = 'NO - Wind')
ax.bar(X + 0.25, NO_ss, width = 0.05, label = 'NO - Solar')
ax.bar(X + 0.30, NL_ws, width = 0.05, label = 'NL - Wind')
ax.bar(X + 0.35, NL_ss, width = 0.05, label = 'NL - Solar')
ax.bar(X + 0.40, GB_ws, width = 0.05, label = 'GB - Wind')
ax.bar(X + 0.45, GB_ss, width = 0.05, label = 'GB - Solar')
plt.xticks(index + 0.225, ('2030', '2040', '2050'))
plt.ylim(0,240)
plt.ylabel('Cost per energy produced [EUR/MWh]')
plt.title('Levelized cost of fuel - H2 (ship)')
plt.grid(axis="y",linewidth = "0.5")
plt.rcParams.update({'font.size': 16})
plt.tick_params(axis=u'both', which=u'both',length=0) 
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


index = np.arange(3)
X = np.arange(3)
plt.figure(figsize = (12,8), dpi = 150)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(X + 0.00, DK_ws1, width = 0.05, label = 'DK - Wind')
ax.bar(X + 0.05, DK_ss1, width = 0.05, label = 'DK - Solar')
ax.bar(X + 0.10, DE_ws1, width = 0.05, label = 'DE - Wind')
ax.bar(X + 0.15, DE_ss1, width = 0.05, label = 'DE - Solar')
ax.bar(X + 0.20, NO_ws1, width = 0.05, label = 'NO - Wind')
ax.bar(X + 0.25, NO_ss1, width = 0.05, label = 'NO - Solar')
ax.bar(X + 0.30, NL_ws1, width = 0.05, label = 'NL - Wind')
ax.bar(X + 0.35, NL_ss1, width = 0.05, label = 'NL - Solar')
ax.bar(X + 0.40, GB_ws1, width = 0.05, label = 'GB - Wind')
ax.bar(X + 0.45, GB_ss1, width = 0.05, label = 'GB - Solar')
plt.xticks(index + 0.225, ('2030', '2040', '2050'))
plt.ylim(0,240)
plt.ylabel('Cost per energy produced [EUR/MWh]')
plt.title('Levelized cost of fuel - NH3 (ship)')
plt.grid(axis="y",linewidth = "0.5")
plt.rcParams.update({'font.size': 16})
plt.tick_params(axis=u'both', which=u'both',length=0) 
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    
#%% Installed Capacity components at different locations    


# Installed capacity in 2030 (NH3)   
fuel= 'NH3'
i = 0
DKW = np.array([p_opt['Onshore Wind'][10-i],p_opt['Solar PV'][10-i],p_opt['%s Plant'%(fuel)][10-i],p_opt['%s Transport'%(fuel)][10-i],p_opt['Battery'][10-i]])
DKS = np.array([p_opt['Onshore Wind'][11-i],p_opt['Solar PV'][11-i],p_opt['%s Plant'%(fuel)][11-i],p_opt['%s Transport'%(fuel)][11-i],p_opt['Battery'][11-i]])
DEW = np.array([p_opt['Onshore Wind'][12-i],p_opt['Solar PV'][12-i],p_opt['%s Plant'%(fuel)][12-i],p_opt['%s Transport'%(fuel)][12-i],p_opt['Battery'][12-i]])
DES = np.array([p_opt['Onshore Wind'][13-i],p_opt['Solar PV'][13-i],p_opt['%s Plant'%(fuel)][13-i],p_opt['%s Transport'%(fuel)][13-i],p_opt['Battery'][13-i]])
NOW = np.array([p_opt['Onshore Wind'][14-i],p_opt['Solar PV'][14-i],p_opt['%s Plant'%(fuel)][14-i],p_opt['%s Transport'%(fuel)][14-i],p_opt['Battery'][14-i]])
NOS = np.array([p_opt['Onshore Wind'][15-i],p_opt['Solar PV'][15-i],p_opt['%s Plant'%(fuel)][15-i],p_opt['%s Transport'%(fuel)][15-i],p_opt['Battery'][15-i]])
NLW = np.array([p_opt['Onshore Wind'][16-i],p_opt['Solar PV'][16-i],p_opt['%s Plant'%(fuel)][16-i],p_opt['%s Transport'%(fuel)][16-i],p_opt['Battery'][16-i]])
NLS = np.array([p_opt['Onshore Wind'][17-i],p_opt['Solar PV'][17-i],p_opt['%s Plant'%(fuel)][17-i],p_opt['%s Transport'%(fuel)][17-i],p_opt['Battery'][17-i]])
GBW = np.array([p_opt['Onshore Wind'][18-i],p_opt['Solar PV'][18-i],p_opt['%s Plant'%(fuel)][18-i],p_opt['%s Transport'%(fuel)][18-i],p_opt['Battery'][18-i]])
GBS = np.array([p_opt['Onshore Wind'][19-i],p_opt['Solar PV'][19-i],p_opt['%s Plant'%(fuel)][19-i],p_opt['%s Transport'%(fuel)][19-i],p_opt['Battery'][19-i]])

#fuel= 'H2'
index = np.arange(5)
X = np.arange(5)
fig = plt.figure(figsize = (12,8), dpi = 150)
ax = fig.add_axes([0,0,1,1])
ax.bar(X+0.05, DKW, width = 0.05, label = 'DK - Wind')
ax.bar(X+0.1, DKS, width = 0.05, label = 'DK - Solar')
ax.bar(X+0.15, DEW, width = 0.05, label = 'DE - Wind')
ax.bar(X+0.2, DES, width = 0.05, label = 'DE - Solar')
ax.bar(X+0.25, NOW, width = 0.05, label = 'NO - Wind')
ax.bar(X+0.3, NOS, width = 0.05, label = 'NO - Solar')
ax.bar(X+0.35, NLW, width = 0.05, label = 'NL - Wind')
ax.bar(X+0.4, NLS, width = 0.05, label = 'NL - Solar')
ax.bar(X+0.45, GBW, width = 0.05, label = 'GB - Wind')
ax.bar(X+0.5, GBS, width = 0.05, label = 'GB - Solar')
plt.xticks(index + 0.25, ('Onshore Wind', 'Solar PV', '%s Plant'%(fuel), '%s Transport'%(fuel), 'Battery'), rotation=90)
plt.ylabel('Installed Capacity [MW]')
plt.ylim(0,12000)
plt.title('Optimal Power Capacity - 2030')
plt.grid(axis="y",linewidth = "0.5")
plt.rcParams.update({'font.size': 16})
plt.tick_params(axis=u'both', which=u'both',length=0) 
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


# Installed capacity in 2040 (NH3)  
i = -20
DKW = np.array([p_opt['Onshore Wind'][10-i],p_opt['Solar PV'][10-i],p_opt['%s Plant'%(fuel)][10-i],p_opt['%s Transport'%(fuel)][10-i],p_opt['Battery'][10-i]])
DKS = np.array([p_opt['Onshore Wind'][11-i],p_opt['Solar PV'][11-i],p_opt['%s Plant'%(fuel)][11-i],p_opt['%s Transport'%(fuel)][11-i],p_opt['Battery'][11-i]])
DEW = np.array([p_opt['Onshore Wind'][12-i],p_opt['Solar PV'][12-i],p_opt['%s Plant'%(fuel)][12-i],p_opt['%s Transport'%(fuel)][12-i],p_opt['Battery'][12-i]])
DES = np.array([p_opt['Onshore Wind'][13-i],p_opt['Solar PV'][13-i],p_opt['%s Plant'%(fuel)][13-i],p_opt['%s Transport'%(fuel)][13-i],p_opt['Battery'][13-i]])
NOW = np.array([p_opt['Onshore Wind'][14-i],p_opt['Solar PV'][14-i],p_opt['%s Plant'%(fuel)][14-i],p_opt['%s Transport'%(fuel)][14-i],p_opt['Battery'][14-i]])
NOS = np.array([p_opt['Onshore Wind'][15-i],p_opt['Solar PV'][15-i],p_opt['%s Plant'%(fuel)][15-i],p_opt['%s Transport'%(fuel)][15-i],p_opt['Battery'][15-i]])
NLW = np.array([p_opt['Onshore Wind'][16-i],p_opt['Solar PV'][16-i],p_opt['%s Plant'%(fuel)][16-i],p_opt['%s Transport'%(fuel)][16-i],p_opt['Battery'][16-i]])
NLS = np.array([p_opt['Onshore Wind'][17-i],p_opt['Solar PV'][17-i],p_opt['%s Plant'%(fuel)][17-i],p_opt['%s Transport'%(fuel)][17-i],p_opt['Battery'][17-i]])
GBW = np.array([p_opt['Onshore Wind'][18-i],p_opt['Solar PV'][18-i],p_opt['%s Plant'%(fuel)][18-i],p_opt['%s Transport'%(fuel)][18-i],p_opt['Battery'][18-i]])
GBS = np.array([p_opt['Onshore Wind'][19-i],p_opt['Solar PV'][19-i],p_opt['%s Plant'%(fuel)][19-i],p_opt['%s Transport'%(fuel)][19-i],p_opt['Battery'][19-i]])

index = np.arange(5)
X = np.arange(5)
fig = plt.figure(figsize = (12,8), dpi = 150)
ax = fig.add_axes([0,0,1,1])
ax.bar(X+0.05, DKW, width = 0.05, label = 'DK - Wind')
ax.bar(X+0.1, DKS, width = 0.05, label = 'DK - Solar')
ax.bar(X+0.15, DEW, width = 0.05, label = 'DE - Wind')
ax.bar(X+0.2, DES, width = 0.05, label = 'DE - Solar')
ax.bar(X+0.25, NOW, width = 0.05, label = 'NO - Wind')
ax.bar(X+0.3, NOS, width = 0.05, label = 'NO - Solar')
ax.bar(X+0.35, NLW, width = 0.05, label = 'NL - Wind')
ax.bar(X+0.4, NLS, width = 0.05, label = 'NL - Solar')
ax.bar(X+0.45, GBW, width = 0.05, label = 'GB - Wind')
ax.bar(X+0.5, GBS, width = 0.05, label = 'GB - Solar')
plt.xticks(index + 0.25, ('Onshore Wind', 'Solar PV', '%s Plant'%(fuel), '%s Transport'%(fuel), 'Battery'), rotation=90)
plt.ylabel('Installed Capacity [MW]')
plt.ylim(0,20000)
plt.title('Optimal Power Capacity - 2040')
plt.grid(axis="y",linewidth = "0.5")
plt.rcParams.update({'font.size': 16})
plt.tick_params(axis=u'both', which=u'both',length=0) 
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


# Installed capacity in 2050 (NH3)  
i = -40
DKW = np.array([p_opt['Onshore Wind'][10-i],p_opt['Solar PV'][10-i],p_opt['%s Plant'%(fuel)][10-i],p_opt['%s Transport'%(fuel)][10-i],p_opt['Battery'][10-i]])
DKS = np.array([p_opt['Onshore Wind'][11-i],p_opt['Solar PV'][11-i],p_opt['%s Plant'%(fuel)][11-i],p_opt['%s Transport'%(fuel)][11-i],p_opt['Battery'][11-i]])
DEW = np.array([p_opt['Onshore Wind'][12-i],p_opt['Solar PV'][12-i],p_opt['%s Plant'%(fuel)][12-i],p_opt['%s Transport'%(fuel)][12-i],p_opt['Battery'][12-i]])
DES = np.array([p_opt['Onshore Wind'][13-i],p_opt['Solar PV'][13-i],p_opt['%s Plant'%(fuel)][13-i],p_opt['%s Transport'%(fuel)][13-i],p_opt['Battery'][13-i]])
NOW = np.array([p_opt['Onshore Wind'][14-i],p_opt['Solar PV'][14-i],p_opt['%s Plant'%(fuel)][14-i],p_opt['%s Transport'%(fuel)][14-i],p_opt['Battery'][14-i]])
NOS = np.array([p_opt['Onshore Wind'][15-i],p_opt['Solar PV'][15-i],p_opt['%s Plant'%(fuel)][15-i],p_opt['%s Transport'%(fuel)][15-i],p_opt['Battery'][15-i]])
NLW = np.array([p_opt['Onshore Wind'][16-i],p_opt['Solar PV'][16-i],p_opt['%s Plant'%(fuel)][16-i],p_opt['%s Transport'%(fuel)][16-i],p_opt['Battery'][16-i]])
NLS = np.array([p_opt['Onshore Wind'][17-i],p_opt['Solar PV'][17-i],p_opt['%s Plant'%(fuel)][17-i],p_opt['%s Transport'%(fuel)][17-i],p_opt['Battery'][17-i]])
GBW = np.array([p_opt['Onshore Wind'][18-i],p_opt['Solar PV'][18-i],p_opt['%s Plant'%(fuel)][18-i],p_opt['%s Transport'%(fuel)][18-i],p_opt['Battery'][18-i]])
GBS = np.array([p_opt['Onshore Wind'][19-i],p_opt['Solar PV'][19-i],p_opt['%s Plant'%(fuel)][19-i],p_opt['%s Transport'%(fuel)][19-i],p_opt['Battery'][19-i]])

index = np.arange(5)
X = np.arange(5)
fig = plt.figure(figsize = (12,8), dpi = 150)
ax = fig.add_axes([0,0,1,1])
ax.bar(X+0.05, DKW, width = 0.05, label = 'DK - Wind')
ax.bar(X+0.1, DKS, width = 0.05, label = 'DK - Solar')
ax.bar(X+0.15, DEW, width = 0.05, label = 'DE - Wind')
ax.bar(X+0.2, DES, width = 0.05, label = 'DE - Solar')
ax.bar(X+0.25, NOW, width = 0.05, label = 'NO - Wind')
ax.bar(X+0.3, NOS, width = 0.05, label = 'NO - Solar')
ax.bar(X+0.35, NLW, width = 0.05, label = 'NL - Wind')
ax.bar(X+0.4, NLS, width = 0.05, label = 'NL - Solar')
ax.bar(X+0.45, GBW, width = 0.05, label = 'GB - Wind')
ax.bar(X+0.5, GBS, width = 0.05, label = 'GB - Solar')
plt.xticks(index + 0.25, ('Onshore Wind', 'Solar PV', '%s Plant'%(fuel), '%s Transport'%(fuel), 'Battery'), rotation=90)
plt.ylabel('Installed Capacity [MW]')
plt.ylim(0,18500)
plt.title('Optimal Power Capacity - 2050')
plt.grid(axis="y",linewidth = "0.5")
plt.rcParams.update({'font.size': 16})
plt.tick_params(axis=u'both', which=u'both',length=0) 
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


# Installed capacity in 2030 (H2)   
fuel= 'NH3'
i = 10
DKW = np.array([p_opt['Onshore Wind'][10-i],p_opt['Solar PV'][10-i],p_opt['%s Plant'%(fuel)][10-i],p_opt['%s Transport'%(fuel)][10-i],p_opt['Battery'][10-i]])
DKS = np.array([p_opt['Onshore Wind'][11-i],p_opt['Solar PV'][11-i],p_opt['%s Plant'%(fuel)][11-i],p_opt['%s Transport'%(fuel)][11-i],p_opt['Battery'][11-i]])
DEW = np.array([p_opt['Onshore Wind'][12-i],p_opt['Solar PV'][12-i],p_opt['%s Plant'%(fuel)][12-i],p_opt['%s Transport'%(fuel)][12-i],p_opt['Battery'][12-i]])
DES = np.array([p_opt['Onshore Wind'][13-i],p_opt['Solar PV'][13-i],p_opt['%s Plant'%(fuel)][13-i],p_opt['%s Transport'%(fuel)][13-i],p_opt['Battery'][13-i]])
NOW = np.array([p_opt['Onshore Wind'][14-i],p_opt['Solar PV'][14-i],p_opt['%s Plant'%(fuel)][14-i],p_opt['%s Transport'%(fuel)][14-i],p_opt['Battery'][14-i]])
NOS = np.array([p_opt['Onshore Wind'][15-i],p_opt['Solar PV'][15-i],p_opt['%s Plant'%(fuel)][15-i],p_opt['%s Transport'%(fuel)][15-i],p_opt['Battery'][15-i]])
NLW = np.array([p_opt['Onshore Wind'][16-i],p_opt['Solar PV'][16-i],p_opt['%s Plant'%(fuel)][16-i],p_opt['%s Transport'%(fuel)][16-i],p_opt['Battery'][16-i]])
NLS = np.array([p_opt['Onshore Wind'][17-i],p_opt['Solar PV'][17-i],p_opt['%s Plant'%(fuel)][17-i],p_opt['%s Transport'%(fuel)][17-i],p_opt['Battery'][17-i]])
GBW = np.array([p_opt['Onshore Wind'][18-i],p_opt['Solar PV'][18-i],p_opt['%s Plant'%(fuel)][18-i],p_opt['%s Transport'%(fuel)][18-i],p_opt['Battery'][18-i]])
GBS = np.array([p_opt['Onshore Wind'][19-i],p_opt['Solar PV'][19-i],p_opt['%s Plant'%(fuel)][19-i],p_opt['%s Transport'%(fuel)][19-i],p_opt['Battery'][19-i]])

fuel= 'H2'
index = np.arange(5)
X = np.arange(5)
fig = plt.figure(figsize = (12,8), dpi = 150)
ax = fig.add_axes([0,0,1,1])
ax.bar(X+0.05, DKW, width = 0.05, label = 'DK - Wind')
ax.bar(X+0.1, DKS, width = 0.05, label = 'DK - Solar')
ax.bar(X+0.15, DEW, width = 0.05, label = 'DE - Wind')
ax.bar(X+0.2, DES, width = 0.05, label = 'DE - Solar')
ax.bar(X+0.25, NOW, width = 0.05, label = 'NO - Wind')
ax.bar(X+0.3, NOS, width = 0.05, label = 'NO - Solar')
ax.bar(X+0.35, NLW, width = 0.05, label = 'NL - Wind')
ax.bar(X+0.4, NLS, width = 0.05, label = 'NL - Solar')
ax.bar(X+0.45, GBW, width = 0.05, label = 'GB - Wind')
ax.bar(X+0.5, GBS, width = 0.05, label = 'GB - Solar')
plt.xticks(index + 0.25, ('Onshore Wind', 'Solar PV', '%s Plant'%(fuel), '%s Transport'%(fuel), 'Battery'), rotation=90)
plt.ylabel('Installed Capacity [MW]')
plt.ylim(0,12000)
plt.title('Optimal Power Capacity - 2030')
plt.grid(axis="y",linewidth = "0.5")
plt.rcParams.update({'font.size': 16})
plt.tick_params(axis=u'both', which=u'both',length=0) 
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


# Installed capacity in 2040 (H2)   
fuel= 'NH3'
i = -10
DKW = np.array([p_opt['Onshore Wind'][10-i],p_opt['Solar PV'][10-i],p_opt['%s Plant'%(fuel)][10-i],p_opt['%s Transport'%(fuel)][10-i],p_opt['Battery'][10-i]])
DKS = np.array([p_opt['Onshore Wind'][11-i],p_opt['Solar PV'][11-i],p_opt['%s Plant'%(fuel)][11-i],p_opt['%s Transport'%(fuel)][11-i],p_opt['Battery'][11-i]])
DEW = np.array([p_opt['Onshore Wind'][12-i],p_opt['Solar PV'][12-i],p_opt['%s Plant'%(fuel)][12-i],p_opt['%s Transport'%(fuel)][12-i],p_opt['Battery'][12-i]])
DES = np.array([p_opt['Onshore Wind'][13-i],p_opt['Solar PV'][13-i],p_opt['%s Plant'%(fuel)][13-i],p_opt['%s Transport'%(fuel)][13-i],p_opt['Battery'][13-i]])
NOW = np.array([p_opt['Onshore Wind'][14-i],p_opt['Solar PV'][14-i],p_opt['%s Plant'%(fuel)][14-i],p_opt['%s Transport'%(fuel)][14-i],p_opt['Battery'][14-i]])
NOS = np.array([p_opt['Onshore Wind'][15-i],p_opt['Solar PV'][15-i],p_opt['%s Plant'%(fuel)][15-i],p_opt['%s Transport'%(fuel)][15-i],p_opt['Battery'][15-i]])
NLW = np.array([p_opt['Onshore Wind'][16-i],p_opt['Solar PV'][16-i],p_opt['%s Plant'%(fuel)][16-i],p_opt['%s Transport'%(fuel)][16-i],p_opt['Battery'][16-i]])
NLS = np.array([p_opt['Onshore Wind'][17-i],p_opt['Solar PV'][17-i],p_opt['%s Plant'%(fuel)][17-i],p_opt['%s Transport'%(fuel)][17-i],p_opt['Battery'][17-i]])
GBW = np.array([p_opt['Onshore Wind'][18-i],p_opt['Solar PV'][18-i],p_opt['%s Plant'%(fuel)][18-i],p_opt['%s Transport'%(fuel)][18-i],p_opt['Battery'][18-i]])
GBS = np.array([p_opt['Onshore Wind'][19-i],p_opt['Solar PV'][19-i],p_opt['%s Plant'%(fuel)][19-i],p_opt['%s Transport'%(fuel)][19-i],p_opt['Battery'][19-i]])

fuel= 'H2'
index = np.arange(5)
X = np.arange(5)
fig = plt.figure(figsize = (12,8), dpi = 150)
ax = fig.add_axes([0,0,1,1])
ax.bar(X+0.05, DKW, width = 0.05, label = 'DK - Wind')
ax.bar(X+0.1, DKS, width = 0.05, label = 'DK - Solar')
ax.bar(X+0.15, DEW, width = 0.05, label = 'DE - Wind')
ax.bar(X+0.2, DES, width = 0.05, label = 'DE - Solar')
ax.bar(X+0.25, NOW, width = 0.05, label = 'NO - Wind')
ax.bar(X+0.3, NOS, width = 0.05, label = 'NO - Solar')
ax.bar(X+0.35, NLW, width = 0.05, label = 'NL - Wind')
ax.bar(X+0.4, NLS, width = 0.05, label = 'NL - Solar')
ax.bar(X+0.45, GBW, width = 0.05, label = 'GB - Wind')
ax.bar(X+0.5, GBS, width = 0.05, label = 'GB - Solar')
plt.xticks(index + 0.25, ('Onshore Wind', 'Solar PV', '%s Plant'%(fuel), '%s Transport'%(fuel), 'Battery'), rotation=90)
plt.ylabel('Installed Capacity [MW]')
plt.ylim(0,20000)
plt.title('Optimal Power Capacity - 2040')
plt.grid(axis="y",linewidth = "0.5")
plt.rcParams.update({'font.size': 16})
plt.tick_params(axis=u'both', which=u'both',length=0) 
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


# Installed capacity in 2050 (H2)   
fuel= 'NH3'
i = -30
DKW = np.array([p_opt['Onshore Wind'][10-i],p_opt['Solar PV'][10-i],p_opt['%s Plant'%(fuel)][10-i],p_opt['%s Transport'%(fuel)][10-i],p_opt['Battery'][10-i]])
DKS = np.array([p_opt['Onshore Wind'][11-i],p_opt['Solar PV'][11-i],p_opt['%s Plant'%(fuel)][11-i],p_opt['%s Transport'%(fuel)][11-i],p_opt['Battery'][11-i]])
DEW = np.array([p_opt['Onshore Wind'][12-i],p_opt['Solar PV'][12-i],p_opt['%s Plant'%(fuel)][12-i],p_opt['%s Transport'%(fuel)][12-i],p_opt['Battery'][12-i]])
DES = np.array([p_opt['Onshore Wind'][13-i],p_opt['Solar PV'][13-i],p_opt['%s Plant'%(fuel)][13-i],p_opt['%s Transport'%(fuel)][13-i],p_opt['Battery'][13-i]])
NOW = np.array([p_opt['Onshore Wind'][14-i],p_opt['Solar PV'][14-i],p_opt['%s Plant'%(fuel)][14-i],p_opt['%s Transport'%(fuel)][14-i],p_opt['Battery'][14-i]])
NOS = np.array([p_opt['Onshore Wind'][15-i],p_opt['Solar PV'][15-i],p_opt['%s Plant'%(fuel)][15-i],p_opt['%s Transport'%(fuel)][15-i],p_opt['Battery'][15-i]])
NLW = np.array([p_opt['Onshore Wind'][16-i],p_opt['Solar PV'][16-i],p_opt['%s Plant'%(fuel)][16-i],p_opt['%s Transport'%(fuel)][16-i],p_opt['Battery'][16-i]])
NLS = np.array([p_opt['Onshore Wind'][17-i],p_opt['Solar PV'][17-i],p_opt['%s Plant'%(fuel)][17-i],p_opt['%s Transport'%(fuel)][17-i],p_opt['Battery'][17-i]])
GBW = np.array([p_opt['Onshore Wind'][18-i],p_opt['Solar PV'][18-i],p_opt['%s Plant'%(fuel)][18-i],p_opt['%s Transport'%(fuel)][18-i],p_opt['Battery'][18-i]])
GBS = np.array([p_opt['Onshore Wind'][19-i],p_opt['Solar PV'][19-i],p_opt['%s Plant'%(fuel)][19-i],p_opt['%s Transport'%(fuel)][19-i],p_opt['Battery'][19-i]])

fuel= 'H2'
index = np.arange(5)
X = np.arange(5)
fig = plt.figure(figsize = (12,8), dpi = 150)
ax = fig.add_axes([0,0,1,1])
ax.bar(X+0.05, DKW, width = 0.05, label = 'DK - Wind')
ax.bar(X+0.1, DKS, width = 0.05, label = 'DK - Solar')
ax.bar(X+0.15, DEW, width = 0.05, label = 'DE - Wind')
ax.bar(X+0.2, DES, width = 0.05, label = 'DE - Solar')
ax.bar(X+0.25, NOW, width = 0.05, label = 'NO - Wind')
ax.bar(X+0.3, NOS, width = 0.05, label = 'NO - Solar')
ax.bar(X+0.35, NLW, width = 0.05, label = 'NL - Wind')
ax.bar(X+0.4, NLS, width = 0.05, label = 'NL - Solar')
ax.bar(X+0.45, GBW, width = 0.05, label = 'GB - Wind')
ax.bar(X+0.5, GBS, width = 0.05, label = 'GB - Solar')
plt.xticks(index + 0.25, ('Onshore Wind', 'Solar PV', '%s Plant'%(fuel), '%s Transport'%(fuel), 'Battery'), rotation=90)
plt.ylabel('Installed Capacity [MW]')
plt.ylim(0,18500)
plt.title('Optimal Power Capacity - 2050')
plt.grid(axis="y",linewidth = "0.5")
plt.rcParams.update({'font.size': 16})
plt.tick_params(axis=u'both', which=u'both',length=0) 
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
