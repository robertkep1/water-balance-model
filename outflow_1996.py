# -*- coding: utf-8 -*-
"""
Created on Wed May 15 10:10:41 2019

@author: rober
"""

#outflow 1996

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import definitions_volume
import time
import stackdata_regression_inf
from datetime import datetime, timedelta

start_time = time.time()

###evaporation input 1996

n_dataset_1996 = 8784
n_dataset_1996_1995 = 17544

evap_input_1996 = r'results\inflow\1996_final\PET_final.xlsx'

evap_data_1996 = pd.read_excel(evap_input_1996)

evap_reach_1996 = evap_data_1996.loc[evap_data_1996['HRU']==26]

list_evap_1996 = np.arange(0,17545,24)

evap_list_1996=np.zeros(17544)

count_1996=0
evaporation_1996=0
for i in np.arange(n_dataset_1996_1995):
    if i == list_evap_1996[count_1996]:
        evaporation = evap_reach_1996.iloc[count_1996,2]
        count_1996 = count_1996+1
    evap_list_1996[i]=evaporation


###surface inflow input 1996

path_inp_1996 = r'results\inflow\1996_final\1996_final.xlsx'

data_1996 = pd.read_excel(path_inp_1996)

outflow_1996 = data_1996.loc[data_1996['CH']==26]

outflow_1996['cm3_hour'] = outflow_1996.FLOW_OUTcms.multiply(60*60)


###precipitation input

precip_input = r'C:results\inflow\precip.xlsx'
precip_data_1996 = pd.read_excel(precip_input)
precip_1996 = precip_data_1996.iloc[149015:157799]
precip_1996 = precip_1996.reset_index()
precip_1996_list  = np.zeros(8784)
for j in np.arange(8784):
	precip_1996_list[j] = precip_1996.iloc[j,2]

seepage                      = stackdata_regression_inf.average_overall[0]/100

water_level_max  = 2.61
discharge_max_hourly         = (definitions_volume.discharge_total_per_culvert(water_level_max)*60*60)

###water balance model

water_volume_1996           = np.zeros(n_dataset_1996_1995)
water_volume_1996[0-1]      = 0
peak_flow_reduction_list    = np.zeros(n_dataset_1996_1995)     
surface_inflow_list         = np.zeros(n_dataset_1996_1995)
gw_inflow_list              = np.zeros(n_dataset_1996_1995)
evaporation_result_list     = np.zeros(n_dataset_1996_1995)
delta_before_dschg_list     = np.zeros(n_dataset_1996_1995)
water_level_before_list     = np.zeros(n_dataset_1996_1995)
area_list                   = np.zeros(n_dataset_1996_1995)
water_vol_before            = np.zeros(n_dataset_1996_1995)
delta_list                  = np.zeros(n_dataset_1996_1995)
actual_outflow_list         = np.zeros(n_dataset_1996_1995)
time_list                   = np.zeros(n_dataset_1996_1995)
residence_time_list         = np.zeros(n_dataset_1996_1995)

for i in np.arange(n_dataset_1996_1995):
    surface_inflow_loop         = outflow_1996.iloc[i,10]
    volume_last_t               = water_volume_1996[i-1]
    volume_before_dschg_area    = volume_last_t + surface_inflow_loop
    area                        = definitions_volume.decision_high_low_water_area(volume_before_dschg_area)
    groundwater_inflow          = area*(seepage/24)
    evaporation                 = ((evap_list_1996[i]/1000)/24)*area
    volume_loop_before_dschg    = surface_inflow_loop + volume_last_t + groundwater_inflow - evaporation
    outflow_calculated          = surface_inflow_loop + groundwater_inflow - evaporation - (342.67 - volume_last_t)
	
    if volume_loop_before_dschg < 342.7:
        water_volume_1996 [i]      = volume_loop_before_dschg
        actual_outflow_list[i]     = 0
        residence_time_list[i]      = water_volume_1996[i]/evaporation
    
    elif volume_loop_before_dschg > 342.7:
		
        if volume_loop_before_dschg > discharge_max_hourly:
            water_volume_1996 [i]  = volume_loop_before_dschg - discharge_max_hourly
            actual_outflow_list[i] = discharge_max_hourly
		
        elif volume_loop_before_dschg < discharge_max_hourly:
            water_volume_1996[i]   = 342.7
            actual_outflow_list[i] = max(outflow_calculated,0)
            residence_time_list[i] = water_volume_1996[i]/(evaporation+actual_outflow_list[i])
    
    delta_before_dschg_list[i]  = surface_inflow_loop + groundwater_inflow - evaporation
    gw_inflow_list[i]           = groundwater_inflow	
    water_vol_before[i]         = volume_loop_before_dschg
    surface_inflow_list[i]      = surface_inflow_loop
    evaporation_result_list[i]  = evaporation
    area_list[i]                = area
    delta_list[i]               = water_volume_1996[i] - volume_last_t
    time_list[i]                = 1
    print(i)
    
results_1996                    = pd.DataFrame()
results_1996['time']            = time_list
results_1996['inflow']          = surface_inflow_list
results_1996['gw-inflow']       = gw_inflow_list
results_1996['area']            = area_list
results_1996['evap']            = evaporation_result_list
results_1996['delta_before']    = delta_before_dschg_list
results_1996['water_vol_before']= water_vol_before
results_1996['water_lvl_before']= water_level_before_list
results_1996['peak_flow_red']   = peak_flow_reduction_list
results_1996['w-vol-actual']    = water_volume_1996
results_1996['delta']           = delta_list
results_1996['actual_outflow']  = actual_outflow_list
results_1996['residence-time']  = residence_time_list/24
results_1996                    = results_1996.drop(results_1996.index[0:8760])
results_1996['precip']          = precip_1996_list


###add time to dataframe
time = datetime(1996, 1, 1, 00, 00)

time_zero = np.zeros(10)


time_list=pd.concat([pd.DataFrame([i], columns=['times']) for i in range(10)],ignore_index=True)

count_time=0

for k in np.arange(n_dataset_1996):
    time_loop = time + timedelta(hours=float(k))
    results_1996.iloc[k,0]=time_loop
    count_time += 1
    print(time_loop)

results_1996.to_excel (r'results\1996.xlsx')

#print("- %s seconds -" % (time.time() - start_time))