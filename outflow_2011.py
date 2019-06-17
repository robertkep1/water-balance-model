# -*- coding: utf-8 -*-
"""
Created on Wed May 15 10:15:32 2019

@author: rober
"""


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#import depth_area_regression
import definitions_volume
import time
import stackdata_regression_inf
from datetime import datetime, timedelta

start_time = time.time()


#evaporation input 2011

n_dataset_2011  = 8760
n_dataset_2011_2010 = 17520

evap_input_2011 = r'results\inflow\2011_final\PET_final.xlsx'

evap_data_2011  = pd.read_excel(evap_input_2011)

evap_reach_2011 = evap_data_2011.loc[evap_data_2011['HRU']==26]

list_evap_2011  = np.arange(0,17521,24)

evap_list_2011  = np.zeros(17520)

count_2011      = 0
evaporation_2011= 0

for i in np.arange(n_dataset_2011_2010):
    if i == list_evap_2011[count_2011]:
        evaporation = evap_reach_2011.iloc[count_2011,11]
        count_2011 = count_2011+1
    evap_list_2011[i]=evaporation

###surface inflow input 2011

path_inp_2011 = r'results\inflow\2011_final\2011_final.xlsx'

data_2011 = pd.read_excel(path_inp_2011)

outflow_2011 = data_2011.loc[data_2011['CH']==26]

outflow_2011['cm3_hour'] = outflow_2011.FLOW_OUTcms.multiply(60*60)


###precipitation input 2011


precip_input = r'C:results\inflow\precip.xlsx'
precip_data_2011 = pd.read_excel(precip_input)
precip_2011 = precip_data_2011.iloc[280512:289273]
precip_2011 = precip_2011.reset_index()
precip_2011_list  = np.zeros(n_dataset_2011)
for j in np.arange(n_dataset_2011):
	precip_2011_list[j] = precip_2011.iloc[j,2]
    
seepage                      = stackdata_regression_inf.average_overall[0]/100

water_level_max  = 2.61
discharge_max_hourly         = (definitions_volume.discharge_total_per_culvert(water_level_max)*60*60)


#water balance model 2011

water_volume_2011           = np.zeros(n_dataset_2011_2010)
water_volume_2011[0-1]      = 0
peak_flow_reduction_list    = np.zeros(n_dataset_2011_2010)     
surface_inflow_list         = np.zeros(n_dataset_2011_2010)
gw_inflow_list              = np.zeros(n_dataset_2011_2010)
evaporation_result_list     = np.zeros(n_dataset_2011_2010)
delta_before_dschg_list     = np.zeros(n_dataset_2011_2010)
water_level_before_list     = np.zeros(n_dataset_2011_2010)
area_list                   = np.zeros(n_dataset_2011_2010)
water_vol_before            = np.zeros(n_dataset_2011_2010)
delta_list                  = np.zeros(n_dataset_2011_2010)
actual_outflow_list         = np.zeros(n_dataset_2011_2010)
time_list                   = np.zeros(n_dataset_2011_2010)
residence_time_list         = np.zeros(n_dataset_2011_2010)

for i in np.arange(n_dataset_2011_2010):
    surface_inflow_loop         = outflow_2011.iloc[i,11]
    volume_last_t               = water_volume_2011[i-1]
    volume_before_dschg_area    = volume_last_t + surface_inflow_loop
    area                        = definitions_volume.decision_high_low_water_area(volume_before_dschg_area)
    groundwater_inflow          = area*(seepage/24)
    evaporation                 = ((evap_list_2011[i]/1000)/24)*area
    volume_loop_before_dschg    = surface_inflow_loop + volume_last_t + groundwater_inflow - evaporation 
    outflow_calculated          = surface_inflow_loop + groundwater_inflow - evaporation - (342.67 - volume_last_t)
    
    if volume_loop_before_dschg < 342.7:
        water_volume_2011 [i]      = volume_loop_before_dschg
        actual_outflow_list[i]     = 0
        residence_time_list[i]      = water_volume_2011[i]/evaporation

    
    elif volume_loop_before_dschg > 342.7:
		
        if volume_loop_before_dschg > discharge_max_hourly:
            water_volume_2011 [i]  = volume_loop_before_dschg - discharge_max_hourly
            actual_outflow_list[i] = discharge_max_hourly
		
        elif volume_loop_before_dschg < discharge_max_hourly:
            water_volume_2011[i]   = 342.7
            actual_outflow_list[i] = max(outflow_calculated,0)
            residence_time_list[i] = water_volume_2011[i]/(evaporation+actual_outflow_list[i])

        
    delta_before_dschg_list[i]  = surface_inflow_loop + groundwater_inflow - evaporation
    gw_inflow_list[i]           = groundwater_inflow	
    water_vol_before[i]         = volume_loop_before_dschg
    surface_inflow_list[i]      = surface_inflow_loop
    evaporation_result_list[i]  = evaporation
    area_list[i]                = area
    delta_list[i]               = water_volume_2011[i] - volume_last_t
    time_list[i]                = 1
    print(i)
    
results_2011                    = pd.DataFrame()
results_2011['time']            = time_list
results_2011['inflow']          = surface_inflow_list
results_2011['gw-inflow']       = gw_inflow_list
results_2011['area']            = area_list
results_2011['evap']            = evaporation_result_list
results_2011['delta_before']    = delta_before_dschg_list
results_2011['water_vol_before']= water_vol_before
results_2011['water_lvl_before']= water_level_before_list
results_2011['peak_flow_red']   = peak_flow_reduction_list
results_2011['w-vol-actual']    = water_volume_2011
results_2011['delta']           = delta_list
results_2011['actual_outflow']  = actual_outflow_list
results_2011['residence-time']  = residence_time_list/24
results_2011                    = results_2011.drop(results_2011.index[0:8760])
results_2011['precip']          = precip_2011_list


###add time to dataframe
time = datetime(2011, 1, 1, 00, 00)

time_zero = np.zeros(10)


time_list=pd.concat([pd.DataFrame([i], columns=['times']) for i in range(10)],ignore_index=True)

count_time=0

for k in np.arange(n_dataset_2011):
    time_loop = time + timedelta(hours=float(k))
    results_2011.iloc[k,0]=time_loop
    count_time += 1
    print(time_loop)

results_2011.to_excel (r'results\2011.xlsx')

#print("- %s seconds -" % (time.time() - start_time))