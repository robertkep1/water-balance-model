# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:12:33 2019

@author: rober
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#import depth_area_regression
import definitions_volume
import time
import stackdata_regression_inf

start_time = time.time()

#evaporation input 1996

n_dataset_1996 = 8784

evap_input_1996 = r'C:\Users\rober\OneDrive\Internship\SWAT model\SWAT_final_olyelo\SWAT_final_olyelo\Scenarios\1996\PET.xlsx'

evap_data_1996 = pd.read_excel(evap_input_1996)

evap_reach_1996 = evap_data_1996.loc[evap_data_1996['HRU']==26]

list_evap_1996 = np.arange(0,8785,24)

evap_list_1996=np.zeros(8784)

count_1996=0
evaporation_1996=0
for i in np.arange(n_dataset_1996):
    if i == list_evap_1996[count_1996]:
        evaporation = evap_reach_1996.iloc[count_1996,11]
        count_1996 = count_1996+1
    evap_list_1996[i]=evaporation


#evaporation input 2006

n_dataset_2006  = 8760

evap_input_2006 = r'C:\Users\rober\OneDrive\Internship\SWAT model\SWAT_final_olyelo\SWAT_final_olyelo\Scenarios\2006\PET.xlsx'

evap_data_2006  = pd.read_excel(evap_input_2006)

evap_reach_2006 = evap_data_2006.loc[evap_data_2006['HRU']==26]

list_evap_2006  = np.arange(0,8761,24)

evap_list_2006  = np.zeros(8760)

count_2006      = 0
evaporation_2006= 0
for i in np.arange(n_dataset_2006):
    if i == list_evap_2006[count_2006]:
        evaporation = evap_reach_2006.iloc[count_2006,2]
        count_2006 = count_2006+1
    evap_list_2006[i]=evaporation    


#evaporation input 2011

n_dataset_2011  = 8760

evap_input_2011 = r'C:\Users\rober\OneDrive\Internship\SWAT model\SWAT_final_olyelo\SWAT_final_olyelo\Scenarios\2011\PET.xlsx'

evap_data_2011  = pd.read_excel(evap_input_2011)

evap_reach_2011 = evap_data_2011.loc[evap_data_2011['HRU']==26]

list_evap_2011  = np.arange(0,8761,24)

evap_list_2011  = np.zeros(8760)

count_2011      = 0
evaporation_2011= 0
for i in np.arange(n_dataset_2011):
    if i == list_evap_2011[count_2011]:
        evaporation = evap_reach_2011.iloc[count_2011,11]
        count_2011 = count_2011+1
    evap_list_2011[i]=evaporation

#data input     
path_inp_1996 = r'C:\Users\rober\OneDrive\Internship\SWAT model\SWAT_final_olyelo\SWAT_final_olyelo\Scenarios\1996\1996.xlsx'
path_inp_2006 = r'C:\Users\rober\OneDrive\Internship\SWAT model\SWAT_final_olyelo\SWAT_final_olyelo\Scenarios\2006\2006.xlsx'
path_inp_2011 = r'C:\Users\rober\OneDrive\Internship\SWAT model\SWAT_final_olyelo\SWAT_final_olyelo\Scenarios\2011\2011.xlsx'

data_1996 = pd.read_excel(path_inp_1996)
data_2006 = pd.read_excel(path_inp_2006)
data_2011 = pd.read_excel(path_inp_2011)


outflow_1996 = data_1996.loc[data_1996['CH']==26]
outflow_2006 = data_2006.loc[data_2006['CH']==26]
outflow_2011 = data_2011.loc[data_2011['CH']==26]

outflow_1996['cm3_hour'] = outflow_1996.FLOW_OUTcms.multiply(60*60)
outflow_2006['cm3_hour'] = outflow_2006.FLOW_OUTcms.multiply(60*60)
outflow_2011['cm3_hour'] = outflow_2011.FLOW_OUTcms.multiply(60*60)



seepage                      = stackdata_regression_inf.average_overall[0]/100

#1996

water_volume_1996            = np.zeros(n_dataset_1996)
water_volume_1996[0-1]       = 0
water_level_list_1996        = np.zeros(n_dataset_1996)
surface_outflow_list_1996    = np.zeros(n_dataset_1996)
delta_in_out_1996            = np.zeros(n_dataset_1996)
list_reduction_1996          = np.zeros(n_dataset_1996)
water_volume_before_list_1996= np.zeros(n_dataset_1996)

for i in np.arange(n_dataset_1996):
    surface_inflow_loop         = outflow_1996.iloc[i,8]
    volume_last_t               = water_volume_1996[i-1]
    volume_before_dschg_area    = volume_last_t + surface_inflow_loop
    area                        = definitions_volume.decision_high_low_water_area(volume_before_dschg_area)
    groundwater_inflow          = area*(seepage/24)
    evaporation                 = ((evap_list_1996[i]/1000)/24)*area
    
    volume_loop_before_dschg    = surface_inflow_loop + volume_last_t + groundwater_inflow - evaporation 
    water_level_before_dschg    = definitions_volume.decision_high_low_water_height(volume_loop_before_dschg)
    discharge_max               = (definitions_volume.discharge_total_per_culvert(water_level_before_dschg)*60*60)
    
    difference                  = volume_loop_before_dschg - discharge_max
    
    if discharge_max<(volume_loop_before_dschg-volume_last_t):
        list_reduction_1996[i] = i
    
    if discharge_max == 0:
        water_volume_loop2 = volume_loop_before_dschg
        if water_volume_loop2 < 0:
            water_volume_1996[i] = 0
        else:
            water_volume_1996[i] = water_volume_loop2
    elif discharge_max > 0:
        if difference < 342.67:
            water_volume_1996[i] = 342.67
        else:
            water_volume_1996[i] = difference
    
    water_volume_before_list_1996[i]=volume_loop_before_dschg
    delta_in_out_1996[i]         = surface_inflow_loop+groundwater_inflow-evaporation
    surface_outflow_list_1996[i] = discharge_max
    water_level_list_1996[i]     = water_level_before_dschg
    print(i)


#2006


water_volume_2006            = np.zeros(n_dataset_2006)
water_volume_2006[0-1]       = 0
water_level_list_2006        = np.zeros(n_dataset_2006)
surface_outflow_list_2006    = np.zeros(n_dataset_2006)
delta_in_out_2006            = np.zeros(n_dataset_2006)
list_reduction_2006          = np.zeros(n_dataset_2006)
water_volume_before_list_2006= np.zeros(n_dataset_2006)

for i in np.arange(n_dataset_2006):
    surface_inflow_loop         = outflow_2006.iloc[i,9]
    volume_last_t               = water_volume_2006[i-1]
    volume_before_dschg_area    = volume_last_t + surface_inflow_loop
    area                        = definitions_volume.decision_high_low_water_area(volume_before_dschg_area)
    groundwater_inflow          = area*(seepage/24)
    evaporation                 = ((evap_list_2006[i]/1000)/24)*area
    
    volume_loop_before_dschg    = surface_inflow_loop + volume_last_t + groundwater_inflow - evaporation 
    water_level_before_dschg    = definitions_volume.decision_high_low_water_height(volume_loop_before_dschg)
    discharge_max               = (definitions_volume.discharge_total_per_culvert(water_level_before_dschg)*60*60)
    
    difference                  = volume_loop_before_dschg - discharge_max
    
    if discharge_max<(volume_loop_before_dschg-volume_last_t):
        list_reduction_2006[i] = i
    
    if discharge_max == 0:
        water_volume_loop2 = volume_loop_before_dschg
        if water_volume_loop2 < 0:
            water_volume_2006[i] = 0
        else:
            water_volume_2006[i] = water_volume_loop2
    elif discharge_max > 0:
        if difference < 342.67:
            water_volume_2006[i] = 342.67
        else:
            water_volume_2006[i] = difference
    
    water_volume_before_list_2006[i]=volume_loop_before_dschg
    delta_in_out_2006[i]         = surface_inflow_loop+groundwater_inflow-evaporation
    surface_outflow_list_2006[i] = discharge_max
    water_level_list_2006[i]     = water_level_before_dschg
    print(i)
    
#2011
    
water_volume_2011            = np.zeros(n_dataset_2011)
water_volume_2011[0-1]       = 0
water_level_list_2011        = np.zeros(n_dataset_2011)
surface_outflow_list_2011    = np.zeros(n_dataset_2011)
delta_in_out_2011            = np.zeros(n_dataset_2011)
list_reduction_2011          = np.zeros(n_dataset_2011)
water_volume_before_list_2011= np.zeros(n_dataset_2011)

for i in np.arange(n_dataset_2011):
    surface_inflow_loop         = outflow_2011.iloc[i,9]
    volume_last_t               = water_volume_2011[i-1]
    volume_before_dschg_area    = volume_last_t + surface_inflow_loop
    area                        = definitions_volume.decision_high_low_water_area(volume_before_dschg_area)
    groundwater_inflow          = area*(seepage/24)
    evaporation                 = ((evap_list_2011[i]/1000)/24)*area
    
    volume_loop_before_dschg    = surface_inflow_loop + volume_last_t + groundwater_inflow - evaporation 
    water_level_before_dschg    = definitions_volume.decision_high_low_water_height(volume_loop_before_dschg)
    discharge_max               = (definitions_volume.discharge_total_per_culvert(water_level_before_dschg)*60*60)
    
    difference                  = volume_loop_before_dschg - discharge_max
    
    if discharge_max<(volume_loop_before_dschg-volume_last_t):
        list_reduction_2011[i] = i
    
    if discharge_max == 0:
        water_volume_loop2 = volume_loop_before_dschg
        if water_volume_loop2 < 0:
            water_volume_2011[i] = 0
        else:
            water_volume_2011[i] = water_volume_loop2
    elif discharge_max > 0:
        if difference < 342.67:
            water_volume_2011[i] = 342.67
        else:
            water_volume_2011[i] = difference
    
    water_volume_before_list_2011[i]=volume_loop_before_dschg
    delta_in_out_2011[i]         = surface_inflow_loop+groundwater_inflow-evaporation
    surface_outflow_list_2011[i] = discharge_max
    water_level_list_2011[i]     = water_level_before_dschg
    print(i)
    
    
print("- %s seconds -" % (time.time() - start_time))