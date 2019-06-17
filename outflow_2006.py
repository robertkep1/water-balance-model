# -*- coding: utf-8 -*-
"""
Created on Wed May 15 10:13:10 2019

@author: rober
"""

#2006

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import definitions_volume
import time
import stackdata_regression_inf
from datetime import datetime, timedelta

start_time = time.time()

#evaporation input 2006

n_dataset_2006  = 8760
n_dataset_2006_2005 = 17520

evap_input_2006 = r'results\inflow\2006_final\PET_final.xlsx'

evap_data_2006  = pd.read_excel(evap_input_2006)

evap_reach_2006 = evap_data_2006.loc[evap_data_2006['HRU']==26]

list_evap_2006  = np.arange(0,17521,24)

evap_list_2006  = np.zeros(17520)

count_2006      = 0
evaporation_2006= 0

for i in np.arange(n_dataset_2006_2005):
    if i == list_evap_2006[count_2006]:
        evaporation = evap_reach_2006.iloc[count_2006,11]
        count_2006 = count_2006+1
    evap_list_2006[i]=evaporation    
    

###surface inflow input 2006

path_inp_2006 = r'results\inflow\2006_final\2006_final.xlsx'

data_2006 = pd.read_excel(path_inp_2006)

outflow_2006 = data_2006.loc[data_2006['CH']==26]

outflow_2006['cm3_hour'] = outflow_2006.FLOW_OUTcms.multiply(60*60)


###precipitation input 2006

precip_input = r'C:results\inflow\precip.xlsx'
precip_data_2006 = pd.read_excel(precip_input)
precip_2006 = precip_data_2006.iloc[236688:245449]
precip_2006 = precip_2006.reset_index()
precip_2006_list  = np.zeros(8760)
for j in np.arange(8760):
	precip_2006_list[j] = precip_2006.iloc[j,2]

seepage                      = stackdata_regression_inf.average_overall[0]/100

water_level_max  = 2.61
discharge_max_hourly         = (definitions_volume.discharge_total_per_culvert(water_level_max)*60*60)


#water balance model 2006

water_volume_2006           = np.zeros(n_dataset_2006_2005)
water_volume_2006[0-1]      = 0
peak_flow_reduction_list    = np.zeros(n_dataset_2006_2005)     
surface_inflow_list         = np.zeros(n_dataset_2006_2005)
gw_inflow_list              = np.zeros(n_dataset_2006_2005)
evaporation_result_list     = np.zeros(n_dataset_2006_2005)
delta_before_dschg_list     = np.zeros(n_dataset_2006_2005)
difference_list             = np.zeros(n_dataset_2006_2005)
water_level_before_list     = np.zeros(n_dataset_2006_2005)
area_list                   = np.zeros(n_dataset_2006_2005)
water_vol_before            = np.zeros(n_dataset_2006_2005)
delta_list                  = np.zeros(n_dataset_2006_2005)
actual_outflow_list         = np.zeros(n_dataset_2006_2005)
time_list                   = np.zeros(n_dataset_2006_2005)
residence_time_list         = np.zeros(n_dataset_2006_2005)

for i in np.arange(n_dataset_2006_2005):
    surface_inflow_loop         = outflow_2006.iloc[i,11]
    volume_last_t               = water_volume_2006[i-1]
    volume_before_dschg_area    = volume_last_t + surface_inflow_loop
    area                        = definitions_volume.decision_high_low_water_area(volume_before_dschg_area)
    groundwater_inflow          = area*(seepage/24)
    evaporation                 = ((evap_list_2006[i]/1000)/24)*area
    volume_loop_before_dschg    = surface_inflow_loop + volume_last_t + groundwater_inflow - evaporation 
    outflow_calculated          = surface_inflow_loop + groundwater_inflow - evaporation - (342.67 - volume_last_t)
   
    if volume_loop_before_dschg < 342.7:
        water_volume_2006 [i]      = volume_loop_before_dschg
        actual_outflow_list[i]     = 0
        residence_time_list[i]      = water_volume_2006[i]/evaporation
        
    elif volume_loop_before_dschg > 342.7:
        
        if volume_loop_before_dschg > discharge_max_hourly:
            water_volume_2006 [i]  = volume_loop_before_dschg - discharge_max_hourly
            actual_outflow_list[i] = discharge_max_hourly
		
        elif volume_loop_before_dschg < discharge_max_hourly:
            water_volume_2006[i]   = 342.7
            actual_outflow_list[i] = max(outflow_calculated,0)
            residence_time_list[i] = water_volume_2006[i]/(evaporation+actual_outflow_list[i])
    
    delta_before_dschg_list[i]  = surface_inflow_loop + groundwater_inflow - evaporation
    gw_inflow_list[i]           = groundwater_inflow	
    water_vol_before[i]         = volume_loop_before_dschg
    surface_inflow_list[i]      = surface_inflow_loop
    evaporation_result_list[i]  = evaporation
    area_list[i]                = area
    delta_list[i]               = water_volume_2006[i] - volume_last_t
    time_list[i]                = 1

    print(i)
    
results_2006                    = pd.DataFrame()
results_2006['time']            = time_list
results_2006['inflow']          = surface_inflow_list
results_2006['gw-inflow']       = gw_inflow_list
results_2006['area']            = area_list
results_2006['evap']            = evaporation_result_list
results_2006['delta_before']    = delta_before_dschg_list
results_2006['water_vol_before']= water_vol_before
results_2006['water_lvl_before']= water_level_before_list
results_2006['peak_flow_red']   = peak_flow_reduction_list
results_2006['w-vol-actual']    = water_volume_2006
results_2006['delta']           = delta_list
results_2006['actual_outflow']  = actual_outflow_list
results_2006['residence-time']  = residence_time_list/24
results_2006                    = results_2006.drop(results_2006.index[0:8760])
results_2006['precip']          = precip_2006_list


###add time to dataframe

time = datetime(2006, 1, 1, 00, 00)

time_zero = np.zeros(10)


time_list=pd.concat([pd.DataFrame([i], columns=['times']) for i in range(10)],ignore_index=True)

count_time=0

for k in np.arange(n_dataset_2006):
    time_loop = time + timedelta(hours=float(k))
    results_2006.iloc[k,0]=time_loop
    count_time += 1
    print(time_loop)

results_2006.to_excel (r'results\2006.xlsx')

#print("- %s seconds -" % (time.time() - start_time))