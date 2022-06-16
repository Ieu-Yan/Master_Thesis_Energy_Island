

import numpy as np
import pandas as pd
import pypsa


# In[ ]:


switch = pd.Series(
             index = ['year','location','fuel','fuel_transport','increase_demand','myopic','battery','wind','solar','connection'],
             dtype = object
         )
switch['year'] = 2030 # 2030, 2040, 2050
switch['location'] = 'DK Wind' # 'Energy Island', 'CountryCode Wind/Solar'
switch['fuel'] = 'NH3' # 'NH3', 'H2'
switch['fuel_transport'] = True
switch['increase_demand'] = False
switch['myopic'] = False
switch['battery'] = True
switch['wind'] = True
switch['solar'] = True
switch['connection'] = True


# In[ ]:


def annuity(n,r): # annuity factor
    return r/(1.-1./(1.+r)**n)*(r>0)+1/n*(r<=0)

# parameters for different locations
plants = pd.read_csv('data/plants/plants.csv', sep = ';', index_col = 0, comment = '#')


# In[ ]:


def get_time_index(year):
    t = pd.date_range('%d-01-01 00:00'%year, '%d-12-31 23:00'%year, freq = 'H')
    # leap year
    if (np.mod(year,100) != 0 and np.mod(year,4) == 0) or np.mod(year,400) == 0:
        t = t[:1416].union(t[1440:]) # drop 2/29 due to lack of data
    return t

# weather data from refinement
def get_capacity_factor(t,lat_lon):
    cf = pd.DataFrame(index = t, columns = ['Wind','PV'])
    cf.Wind = pd.read_csv('data/cf/wind_%s.csv'%lat_lon, index_col = 0).to_numpy()
    cf.PV = pd.read_csv('data/cf/pv_%s.csv'%lat_lon, index_col = 0).to_numpy()
    return cf


# In[ ]:


def get_efficiency():
    eff = pd.Series(index = ['H2 Plant','NH3 Plant','H2 Engine','NH3 Engine','Battery','HVDC'], dtype = float)
    eff['H2 Plant'] = 141.8/(50*3.6)
    eff['NH3 Plant'] = 23/(9.9*3.6)
    eff['H2 Engine'] = 0.55
    eff['NH3 Engine'] = 0.38
    eff['Battery'] = 0.9216
    eff['HVDC'] = 0.97 # per 1000 km
    return eff


# In[ ]:


# energy island fuel demand
def get_load(switch): # require year, increase_demand
    load = 0.026*1.9e12/3.6/1000/8760 # MWh in every hour
    if switch['increase_demand']: load *= 1.015**(switch['year']-2030)
    return load


# In[ ]:


def get_cost_data(switch): # require year
    # battery max charging hours
    h_b = 6
    # cost data
    # offwind: 21 large turbines, off-shore
    # onwind: 20 Large wind turbines on land
    # pv: large scale utility systems (NO axis-tracking)
    # H2: AEC 100MW
    # NH3: no electrolyzer, ASU ?
    if switch['year'] == 2030:
        n = np.array([30,30,40,30,30,25]) # expected lifetime
        r = np.array([0.07]) # constant discount rate
        costdata = pd.DataFrame(
            np.array([[1.93e6,1.04e6,3.8e5,4.5e5,1.3e6,1.42e5*h_b+1.6e5], # €/MW
                      [36053,12600,7250,9000,39000,540], # €/MW/year
                      [2.7,1.35,0.01,0.01,0.02,1.8]]), # €/MWh
            index = ['Investment','FOM','VOM'],
            columns = ['Offshore Wind','Onshore Wind','Solar PV','H2 Plant','NH3 Plant','Battery']
        )
        
    elif switch['year'] == 2040: 
        n = np.array([30,30,40,32,30,30]) # expected lifetime
        r = np.array([0.07]) # constant discount rate
        costdata = pd.DataFrame(
            np.array([[1.81e6,9.8e5,3.3e5,3e5,1.1e6,9.4e4*h_b+1e5], # €/MW
                      [33169,11592,6625,6000,32000,540], # €/MW/year
                      [2.5,1.24,0.01,0.01,0.02,1.7]]), # €/MWh
            index = ['Investment','FOM','VOM'],
            columns = ['Offshore Wind','Onshore Wind','Solar PV','H2 Plant','NH3 Plant','Battery']
        )
        
    elif switch['year'] == 2050: 
        n = np.array([30,30,40,35,30,30]) # expected lifetime
        r = np.array([0.07]) # constant discount rate
        costdata = pd.DataFrame(
            np.array([[1.78e6,9.6e5,3e5,2.5e5,8e5,7.5e4*h_b+6e4], # €/MW
                      [32448,11340,6250,5000,24000,540], # €/MW/year
                      [2.4,1.22,0.01,0.01,0.02,1.6]]), # €/MWh
            index = ['Investment','FOM','VOM'],
            columns = ['Offshore Wind','Onshore Wind','Solar PV','H2 Plant','NH3 Plant','Battery']
        )
        
    ccost = annuity(n,r)*costdata.loc['Investment']+costdata.loc['FOM'] # €/MW
    return h_b, costdata, ccost


# In[ ]:


def get_distance(switch): # require location
    # detour factor for electricity cables
    dfr = 1.2
   
    # onshore and offshore straight distance
    if switch['location'] == 'Energy Island':
        distance = {'DK': (158.09,81.5), # Thorsminde-DK https://www.mapdevelopers.com/distance_from_to.php
                    'DE': (447.02,192.3), # Sylt-DE
                    'NO': (290.47,185.12), # Mandal-NO
                    'NL': (125.38,354), # Groningen-NL
                    'GB': (75.8,550)} # Newcastle_upon_Tyne-GB

    elif switch['location'] == 'DK Wind':
        distance = {'DK': (163.49,0)} # Distance to country grid [onshore_distance,offshore_distance]
        distance2 = 142.37 # Onshore distance to container port: Esbjerg Hafen

    elif switch['location'] == 'DK Solar':
        distance = {'DK': (112.01,0)} #
        distance2 = 183.37 # Copenhagen Malmö Port
    
    elif switch['location'] == 'DE Wind':
        distance = {'DE': (312.18,0)} # 
        distance2 = 30.67 # Bremerhaven Container-Terminal

    elif switch['location'] == 'DE Solar':
        distance = {'DE': (382.30,0)} #
        distance2 = 461.82 # Intermodal Terminal Venice

    elif switch['location'] == 'NO Wind':
        distance = {'NO': (278.10,0)} #
        distance2 = 49.15 # Westport - Risavika Terminal

    elif switch['location'] == 'NO Solar':
        distance = {'NO': (174.44,0)}
        distance2 = 11.82 # Larvik Havn

    elif switch['location'] == 'NL Wind':
        distance = {'NL': (107.84,0)} # 
        distance2 = 114.01 # CTU Flevokust

    elif switch['location'] == 'NL Solar':
        distance = {'NL': (85.19,0)} #
        distance2 = 44.40 # TMA Terminals B.V.

    elif switch['location'] == 'GB Wind':
        distance = {'GB': (62.58,0)} # 
        distance2 = 42.18 # Teesport Container Terminal

    elif switch['location'] == 'GB Solar':
        distance = {'GB': (412.24,0)} #
        distance2 = 13.82 # Dover Cargo Terminal     
        
    return dfr, distance, distance2


# In[ ]:


def get_country_data(switch,t): # require year
    cprice = pd.read_csv('data/market/price_%d.csv'%switch['year'], index_col = 0).set_index(t)
    cload = pd.read_csv('data/market/load_%d.csv'%switch['year'], index_col = 0).set_index(t)
    return cprice, cload


# In[ ]:


def get_capacity_limits(switch): # require location
    capacitylimits = pd.DataFrame(
        index = ['H2 Plant','NH3 Plant','Wind','Solar PV','Battery','DK','DE','NO','NL','GB'],
        columns = ['Minimum','Maximum'],
        dtype = float
    )
    capacitylimits.Maximum = np.inf
    capacitylimits.Minimum = 0
    
    #capacitylimits.loc['Battery','Maximum'] = 1000
    if switch['location'] == 'Energy Island':
        capacitylimits.loc['DK','Maximum'] = 5000
        capacitylimits.loc['DE','Maximum'] = 2000
        capacitylimits.loc['NO','Maximum'] = 2000
        capacitylimits.loc['NL','Maximum'] = 2000
        capacitylimits.loc['GB','Maximum'] = 1000
    elif switch['location'] == 'DK Wind':
        capacitylimits.loc['DK','Maximum'] = 600
    elif switch['location'] == 'DK Solar':
        capacitylimits.loc['DK','Maximum'] = 600
    elif switch['location'] == 'DE Wind':
        capacitylimits.loc['DE','Maximum'] = 2000
    elif switch['location'] == 'DE Solar':
        capacitylimits.loc['DE','Maximum'] = 2000
    elif switch['location'] == 'NO Wind':
        capacitylimits.loc['NO','Maximum'] = 460
    elif switch['location'] == 'NO Solar':
        capacitylimits.loc['NO','Maximum'] = 460
    elif switch['location'] == 'NL Wind':
        capacitylimits.loc['NL','Maximum'] = 700
    elif switch['location'] == 'NL Solar':
        capacitylimits.loc['NL','Maximum'] = 700
    elif switch['location'] == 'GB Wind':
        capacitylimits.loc['GB','Maximum'] = 2000
    elif switch['location'] == 'GB Solar':
        capacitylimits.loc['GB','Maximum'] = 2000
    
    #if switch['myopic']:
        # some minimum limits
    return capacitylimits


# In[ ]:


def get_country_binary(distance):
    binary_list = []
    for i in range(2**len(distance.keys())):
        binary_list.append(f'{{:0{len(distance.keys())}b}}'.format(i))
    return binary_list

def get_country_switch(distance,binary):
    country_switch = pd.Series(index = distance.keys(), dtype = bool)
    for i,j in zip(distance.keys(),range(len(distance.keys()))):
        country_switch[i] = bool(int(binary[j]))
    return country_switch


# In[ ]:


def build_network():
    network = pypsa.Network()

    network.set_snapshots(t)

    network.add('Bus', 'Electricity')
    network.add('Bus', switch['fuel'])

    network.add('Link',
                switch['fuel']+' Plant',
                bus0 = 'Electricity',
                bus1 = switch['fuel'],
                p_nom_extendable = True,
                p_nom_max = capacitylimits.Maximum[switch['fuel']+' Plant'],
                p_nom_min = capacitylimits.Minimum[switch['fuel']+' Plant'],
                efficiency = eff[switch['fuel']+' Plant'],
                capital_cost = ccost[switch['fuel']+' Plant'],
                marginal_cost = costdata.loc['VOM'][switch['fuel']+' Plant']*eff[switch['fuel']+' Plant'])
    
    network.add('Load',
                'Fuel Demand',
                bus = switch['fuel'],
                p_set = load)

    network.add('StorageUnit',
                'Free Tank',
                bus = switch['fuel'],
                cyclic_state_of_charge = True,
                p_nom_extendable = True)

    if switch['wind']:
        windname = 'Onshore Wind' if plants['onshore'][switch['location']] else 'Offshore Wind'
        network.add('Generator',
                    windname,
                    bus = 'Electricity',
                    p_nom_extendable = True,
                    p_nom_max = capacitylimits.Maximum['Wind'],
                    p_nom_min = capacitylimits.Minimum['Wind'],
                    capital_cost = ccost[windname],
                    marginal_cost = costdata.loc['VOM'][windname],
                    p_max_pu = cf['Wind'])
    
    if switch['solar']:
        network.add('Generator',
                    'Solar PV',
                    bus = 'Electricity',
                    p_nom_extendable = True,
                    p_nom_max = capacitylimits.Maximum['Solar PV'],
                    p_nom_min = capacitylimits.Minimum['Solar PV'],
                    capital_cost = ccost['Solar PV'],
                    marginal_cost = costdata.loc['VOM']['Solar PV'],
                    p_max_pu = cf['PV'])
    
    if switch['battery']:
        network.add('StorageUnit',
                    'Battery',
                    bus = 'Electricity',
                    cyclic_state_of_charge = True,
                    p_nom_extendable = True,
                    p_nom_max = capacitylimits.Maximum['Battery'],
                    p_nom_min = capacitylimits.Minimum['Battery'],
                    capital_cost = ccost['Battery'],
                    marginal_cost = costdata.loc['VOM']['Battery'],
                    efficiency_store = eff['Battery'],
                    efficiency_dispatch = eff['Battery'],
                    max_hours = h_b)
    
    if switch['connection']:
        for country in distance.keys():
            #if country_switch[country]:
            network.add('Bus', country)

            network.add('Generator',
                        'Elec_'+country,
                        bus = country,
                        p_nom_extendable = True,
                        marginal_cost = cprice[country])
            network.add('Load',
                        'Load_'+country, 
                        bus = country, 
                        p_set = cload[country])

            cccost = (annuity(40,0.07)+0.02)*(400*distance[country][0]*dfr+2000*distance[country][1]*dfr+1.5e5)
            cmcost = 0.01
            ceff = 1-(1-eff['HVDC'])*sum(distance[country])*dfr/1000
            
            network.add('Bus', country+' Hub 1') # close to plant
            network.add('Bus', country+' Hub 2') # close to country

            network.add('Link',
                        '%s to %s Hub 1'%(switch['location'],country),
                        bus0 = 'Electricity',
                        bus1 = country+' Hub 1',
                        p_nom_extendable = True)
            network.add('Link',
                        '%s Hub 1 to %s'%(country,switch['location']),
                        bus0 = country+' Hub 1',
                        bus1 = 'Electricity',
                        p_nom_extendable = True,
                        efficiency = ceff,
                        marginal_cost = cmcost)
            network.add('Link',
                        '%s Hub 1 and %s Hub 2'%(country,country), 
                        bus0 = country+' Hub 1',
                        bus1 = country+' Hub 2',
                        p_nom_extendable = True,
                        p_nom_max = capacitylimits.Maximum[country],
                        p_nom_min = capacitylimits.Minimum[country],
                        p_min_pu = -1,
                        capital_cost = cccost)
            network.add('Link',
                        '%s to %s Hub 2'%(country,country),
                        bus0 = country,
                        bus1 = country+' Hub 2',
                        p_nom_extendable = True)
            network.add('Link',
                        '%s Hub 2 to %s'%(country,country),
                        bus0 = country+' Hub 2',
                        bus1 = country,
                        p_nom_extendable = True,
                        efficiency = ceff,
                        marginal_cost = cmcost)
            
    if switch['fuel_transport']: # Fuel transportation by road
        # Fuel transport
        network.add('Bus', 'Truck')
        network.remove("Link","%s Plant"%(switch['fuel']))
        v_t = 60 # Velocity of truck [km/h]
        c_fuel = pd.DataFrame(np.array([[0.98*10**6], [0.7*10**6]]), # CAPEX per truck fuel [EUR/truck]
                              index = ['H2','NH3']) 
        c_dr = 45*8000 # Driver cost [EUR/truck] 
        e_fuel = pd.DataFrame(np.array([[1.5*141.8*1000/3600], [28*23*1000/3600]]), # Energy tonne of fuel per truck [MWh/truck]
                              index = ['H2','NH3']) 
        v_fuel = pd.DataFrame(np.array([[2.6], [0.13]]), # Variable cost [EUR/t*km]
                              index = ['H2','NH3']) 
        factor = pd.DataFrame(np.array([[1.5], [28]]), # Load of fuel [t/truck]
                              index = ['H2','NH3']) 
        t_truck = (2*distance2)/v_t # Transport time of truck (2-way) [h]
        r = e_fuel[0]['%s'%(switch['fuel'])]/t_truck # Relation [MW/truck]
        F_cc = (c_fuel+c_dr)*r**-1 # Fixed cost [EUR/MW]
        V_cc = (v_fuel[0]['%s'%(switch['fuel'])]*distance2)*(factor/e_fuel[0]['%s'%(switch['fuel'])]) # Variable cost [EUR/MWh]
                      
        network.add('Link',
                    '%s Plant'%(switch['fuel']), 
                    bus0 = 'Electricity',
                    bus1 = 'Truck',
                    p_nom_extendable = True,
                    efficiency = eff['%s Plant'%(switch['fuel'])],
                    capital_cost = ccost['%s Plant'%(switch['fuel'])],
                    marginal_cost = costdata.loc['VOM']['%s Plant'%(switch['fuel'])]*eff['%s Plant'%(switch['fuel'])])
        
        network.add('Link',
                    '%s Transport'%(switch['fuel']), 
                    bus0 = 'Truck',
                    bus1 = switch['fuel'],
                    p_nom_extendable = True,
                    capital_cost = F_cc[0]['%s'%(switch['fuel'])],
                    marginal_cost = V_cc[0]['%s'%(switch['fuel'])])
    
    return network


# In[ ]:


eff = get_efficiency()
for switch['year'] in [2030,2040,2050]:
    t = get_time_index(switch['year'])
    load = get_load(switch)
    h_b, costdata, ccost = get_cost_data(switch)
    cprice, cload = get_country_data(switch,t)

    for switch['fuel'] in ['H2','NH3']:          
        for switch['location'] in ['DK Wind','DK Solar','DE Wind','DE Solar','NO Wind','NO Solar','NL Wind','NL Solar','GB Wind','GB Solar']:
            cf = get_capacity_factor(t,plants['location'][switch['location']])
            dfr, distance, distance2 = get_distance(switch)
            capacitylimits = get_capacity_limits(switch)
            network = build_network()
            network.lopf(network.snapshots,
                         pyomo = False,
                         solver_name = 'gurobi')
            network.export_to_netcdf('results/%d_%s_%s.nc'%(switch['year'],switch['fuel'],switch['location']))
                
            
