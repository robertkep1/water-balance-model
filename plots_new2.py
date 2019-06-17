# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:29:21 2019

@author: rober
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 15 16:18:19 2019

@author: rober
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates
#import depth_area_regression
import definitions_volume
import time
import datetime
from matplotlib import gridspec

#data = pd.read_excel(r'C:\Users\rober\OneDrive\Internship\WaterBalance\Volume calculations\volume_data2.xlsx')


### regression curve plot volume-height for low water (<1016.2)

#h_low_data      = data.iloc[0:24,1].values.astype(float)
#vol_low_data    = data.iloc[0:24,2].values.astype(float)
#
#h_low = np.linspace(0,1.011,100)#1.24
#vol_sim_low_water = definitions_volume.volume_low_water(h_low)



#plt.style.use('ggplot')
#plt.figure('Regression Curve')
#
#plt.plot(h_low,vol_sim_low_water,label=r'$Model$', linewidth=1)
#plt.scatter(h_low_data,vol_low_data,label=r'$Dataset$',s=12)
#plt.annotate('Corr. coef = %.3f' % 0., (0.8,0.2) ,xycoords='axes fraction', ha='center', va='center', size='10')
#plt.xlabel('$h$ [$m$]')
#plt.ylabel('$V$ [$m^2$]')
#
#plt.legend()

#plt.show()


### regression curve plot area-height for low water (<1016.2)

#h_low_data      = data.iloc[0:18,1].values.astype(float)
#area_low_data   = data.iloc[0:18,3].values.astype(float)
#
#h_low=np.linspace(0,1.24,100)
#area_sim_low_water = definitions_volume.area_low_water_prep(h_low)
#
#plt.style.use('ggplot')
#plt.figure('Regression Curve')
#
#plt.plot(h_low,area_sim_low_water, linewidth=1.1)
#plt.scatter(h_low_data,area_low_data,s=17)
#plt.annotate('Corr. coef = %.3f' % 0.999, (0.8,0.2) ,xycoords='axes fraction', ha='center', va='center', size='10')
#plt.annotate('Low Water Depth $(h\leq1.01 m)$', (0.34,0.82) ,xycoords='axes fraction', ha='center', va='center', size='12',fontweight='bold')
#plt.xlabel('$h$ [$m$]')
#plt.ylabel('$A$ [$m^2$]')

#plt.legend()

#plt.show()

###regression curve plot for mid water

#h_mid_data      = data.iloc[0:24,1].values.astype(float)
#vol_mid_data    = data.iloc[0:24,2].values.astype(float)
#
#h_mid = np.linspace(1.011,1.24,100)#1.24
#vol_sim_mid_water = definitions_volume.vol_mid_water(h_mid)



#plt.style.use('ggplot')
#plt.figure('Regression Curve')
#
#plt.plot(h_mid,vol_sim_mid_water,label=r'$Model$', linewidth=1)
#plt.scatter(h_mid_data,vol_mid_data,label=r'$Dataset$',s=12)
#plt.annotate('Corr. coef = %.3f' % 0., (0.8,0.2) ,xycoords='axes fraction', ha='center', va='center', size='10')
#plt.xlabel('$h$ [$m$]')
#plt.ylabel('$V$ [$m^2$]')
#
#plt.legend()

#plt.show()


###regression curve plot for high water (>1016.2)

#h_high_data      = data.iloc[0:24,1].values.astype(float)
#vol_high_data    = data.iloc[0:24,2].values.astype(float)
#
#h_high = np.linspace(1.24,1.814,100)#1.24
#vol_sim_high_water = definitions_volume.volume_high_water(h_high)



#plt.style.use('ggplot')
#plt.figure('Regression Curve')
#
#plt.plot(h_high,vol_sim_high_water,label=r'$Model$', linewidth=1)
#plt.scatter(h_high_data,vol_high_data,label=r'$Dataset$',s=12)
#plt.annotate('Corr. coef = %.3f' % 0., (0.8,0.2) ,xycoords='axes fraction', ha='center', va='center', size='10')
#plt.xlabel('$h$ [$m$]')
#plt.ylabel('$V$ [$m^2$]')
#
#plt.legend()
plt.close()

#fig, axs = plt.subplots(3)
#axs[0].plot(h_low,vol_sim_low_water,label=r'$Model$', linewidth=1.1)
#axs[0].scatter(h_low_data,vol_low_data,label=r'$Dataset$',s=17)
#axs[0].annotate('Corr. coef = %.3f' % 0.999, (0.85,0.1) ,xycoords='axes fraction', ha='center', va='center', size='10')
#axs[0].annotate('Low Water Depth $(h\leq 1.01 m)$', (0.32,0.82) ,xycoords='axes fraction', ha='center', va='center', size='12',fontweight='bold')
#axs[1].plot(h_mid,vol_sim_mid_water,label=r'$Model$', linewidth=1.1)
#axs[1].scatter(h_mid_data,vol_mid_data,label=r'$Dataset$',s=17)
#axs[1].annotate('Corr. coef = %.3f' % 1, (0.85,0.1) ,xycoords='axes fraction', ha='center', va='center', size='10')
#axs[1].annotate('Medium Water Depth $(1.011 m \leq h \leq 1.239 m)$', (0.45,0.82) ,xycoords='axes fraction', ha='center', va='center', size='12',fontweight='bold')
#axs[2].plot(h_high,vol_sim_high_water,label=r'$Model$', linewidth=1.1)
#axs[2].scatter(h_high_data,vol_high_data,label=r'$Dataset$',s=17)
#axs[2].annotate('Corr. coef = %.3f' % 0.999, (0.85,0.1) ,xycoords='axes fraction', ha='center', va='center', size='10')
#axs[2].annotate('High Water Depth $(1.24 m \leq h \leq 1.814 m)$', (0.41,0.82) ,xycoords='axes fraction', ha='center', va='center', size='12',fontweight='bold')
#plt.xlabel('$Time$ [$h$]')
#fig.text(0.01, 0.5, 'Seepage Rate [$cm$]', va='center', rotation='vertical')
#plt.tight_layout



path_1996 = r'results\1996.xlsx'
data_1996 = pd.read_excel(path_1996)

data_1996 = data_1996.reset_index()


#list_data_1996 = np.arange(0,8785,24)

#data_list_1996=np.zeros(366)
#evap_list_1996= np.zeros(366)
#
#count_1996=0


#for i in np.arange(8784):
#    if i == list_data_1996[count_1996]:
#        out = data_list_1996.iloc[count_1996,2]
#        count_1996 = count_1996+1
#    evap_list_1996[i]=count_1996

#for i in np.arange(8784):
#	if i==list_data_1996[count_1996]:
#		lower_bound = list_data_1996[count_1996]-24
#		upper_bound = list_data_1996[count_1996]
#		out = data_1996.iloc[lower_bound:upper_bound,4]
#		data_list_1996[count_1996] = np.sum(out)
#		count_1996 = count_1996+1
#
#data_list_1996[0] = np.sum(data_1996.iloc[0:24,11])
#
#
#
#fig, axs = plt.subplots(3)
#axs[0].plot(delta_list_1996,vol_sim_low_water,label=r'$Model$', linewidth=1.1)


####RAINFALL/DISCHARGE PLOT 1996

#rainfall = data_1996.iloc[2184:7320,15].values.astype(float)
#  
#outflow = data_1996.iloc[2184:7320,13].values.astype(float)
#n_1996 = 5136
#t = np.arange(1,5137)
#data1 = rainfall
#data2 = outflow
#
#dates=pd.DataFrame()
#dates['time']=data_1996.iloc[2184:7320,2]
#dates['time'].dt.strftime('%d-%m')
#dates.index = pd.to_datetime(dates.time, dayfirst=True, format='%Y/%m/%d', exact=True, infer_datetime_format=False)
#
#plt.style.use('ggplot')
#
#fig, ax1 = plt.subplots()
#
#color = 'tab:red'
#ax1.set_xlabel('Time [Year-Month]')
#ax1.set_ylabel('Outflow [$m^3\:h^{-1}$]', color=color)
#ax1.plot(dates.index,data2, color=color)
#ax1.tick_params(axis='y', labelcolor=color)
#ax1.set_xlim(['1996-04-01 00:00:00', '1996-11-01 01:00:00'])
#ax1.set_ylim([0,150000])
##ax1.hlines(y=(definitions_volume.discharge_total_per_culvert(2.61)*60*60),xmin='1996-04-01 00:00:00',xmax='1996-10-31 23:00:00')
#ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
#color = 'tab:blue'
#ax2.set_ylabel('Precipitation [$mm\: h^{-1}$]', color=color)  # we already handled the x-label with ax1
#ax2.bar(dates.index, data1, color=color,width=0.6)
#ax2.tick_params(axis='y', labelcolor=color)
#ax2.set_ylim(0,50)
#ax2.set_xlim(['1996-04-01 00:00:00', '1996-11-01 01:00:00'])
#ax2.grid(b=None)
#ax2.invert_yaxis()
#
#fig.set_figheight(4)
#fig.set_figwidth(12)
#fig.tight_layout()  # otherwise the right y-label is slightly clipped
#
#plt.show()


### WATER STORAGE 1996

#delta_data = data_1996.iloc[0:2664,12]
#evaporation_data = data_1996.iloc[0:2664,6]
#groundwater_inflow_data = data_1996.iloc[0:2664,4]
#water_volume_data = data_1996.iloc[0:2664,11]
#
#dates=pd.DataFrame()
#dates['time']=data_1996.iloc[0:2664,2]
#dates['time'].dt.strftime('%d-%m')
#dates.index = pd.to_datetime(dates.time, dayfirst=True, format='%Y/%m/%d', exact=True, infer_datetime_format=False)
#
#gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2]) 
#
#ax1=plt.subplot(gs[1])
#
#plt.style.use('ggplot')
#plt.plot(dates.index,delta_data,label='$\Delta$ Storage Volume',color='red',lw=2)
#plt.plot(dates.index,evaporation_data*-1,label='Evaporation',ls='--',lw=1.5,color='green')
#plt.plot(dates.index,groundwater_inflow_data,label='Groundwater Inflow',lw=1.5,ls='--',color='blue')
#plt.hlines(0,'1996-01-01 00:00:00','1996-04-20 23:00:00',lw=2)
#plt.ylim([-0.5,2])
#plt.xlim('1996-01-01 00:00:00','1996-04-20 23:00:00')
#plt.xticks(ticks=['1996-01-01 00:00:00','1996-02-01 00:00:00','1996-03-01 00:00:00','1996-04-01 00:00:00','1996-04-20 23:00:00'],labels=['1996-01-01','1996-02-01','1996-03-01','1996-04-01','1996-04-20'])
#ax1.xaxis.set_tick_params(labelsize=12)
#ax1.yaxis.set_tick_params(labelsize=12)
#plt.ylabel('Flow [$m^3 \: h^{-1}$]',fontsize=14)
#plt.xlabel('Time [Year-Month-Day]',fontsize=14)
#plt.legend(fontsize=17)
#plt.tight_layout()
#
#ax2=plt.subplot(gs[0])
#plt.plot(dates.index,water_volume_data,label='Water Storage Volume',color='orange',lw=2)
#plt.xlim('1996-01-01 00:00:00','1996-04-20 23:00:00')
#plt.hlines(342.67,'1996-01-01 03:00:00','1996-04-20 23:00:00',label='Max. Storage Volume',lw=2)
#plt.xticks(ticks=['1996-01-01 00:00:00','1996-02-01 00:00:00','1996-03-01 00:00:00','1996-04-01 00:00:00','1996-04-20 23:00:00'],labels=['1996-01-01','1996-02-01','1996-03-01','1996-04-01','1996-04-20'])
#plt.ylabel('Wetland Storage Volume [$m^3$]',fontsize=13)
#ax2.set_xticklabels([])
#ax3 =ax2.twinx()
#ax3.plot(dates.index, water_volume_data/342.67,lw=0)
#ax3.set_ylabel('Fract. Max. Water Storage Vol. [-]',fontsize=13)
#ax3.grid(b=None)
#ax2.tick_params(labelsize=12)
#ax3.tick_params(labelsize=12)
#ax2.set_xticklabels([])
#ax2.legend(loc='lower left',fontsize=15)
#plt.show()



### PLOTS 2006
#plt.close()
path_2006 = r'results\2006.xlsx'
data_2006 = pd.read_excel(path_2006)

data_2006 = data_2006.reset_index()


### WATER STORAGE 2006

#delta_data = data_2006.iloc[0:2664,12]
#evaporation_data = data_2006.iloc[0:2664,6]
#groundwater_inflow_data = data_2006.iloc[0:2664,4]
#water_volume_data = data_2006.iloc[0:2664,11]
#area_data = data_2006.iloc[0:2664,5]
#
#dates=pd.DataFrame()
#dates['time']=data_2006.iloc[0:2664,2]
#dates['time'].dt.strftime('%d-%m')
#dates.index = pd.to_datetime(dates.time, dayfirst=True, format='%Y/%m/%d', exact=True, infer_datetime_format=False)
#
#gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2]) 
#
#ax1=plt.subplot(gs[1])
#
#plt.style.use('ggplot')
#plt.plot(dates.index,delta_data,label='$\Delta$ Storage Volume',color='red',lw=2)
#plt.plot(dates.index,evaporation_data*-1,label='Evaporation',ls='--',lw=1.5,color='green')
#plt.plot(dates.index,groundwater_inflow_data,label='Groundwater Inflow',lw=1.5,ls='--',color='blue')
#plt.hlines(0,'2006-01-01 00:00:00','2006-04-20 23:00:00',lw=2)
#plt.ylim([-0.5,2])
#plt.xlim('2006-01-01 03:00:00','2006-04-20 23:00:00')
#plt.xticks(ticks=['2006-01-01 03:00:00','2006-02-01 00:00:00','2006-03-01 00:00:00','2006-04-01 00:00:00','2006-04-20 23:00:00'],labels=['2006-01-01','2006-02-01','2006-03-01','2006-04-01','2006-04-20'])
#ax1.xaxis.set_tick_params(labelsize=12)
#ax1.yaxis.set_tick_params(labelsize=12)
#plt.ylabel('Flow [$m^3 \: h^{-1}$]',fontsize=14)
#plt.xlabel('Time [Year-Month-Day]',fontsize=14)
#plt.legend(fontsize=17)
#plt.tight_layout()
#
#ax2=plt.subplot(gs[0])
#plt.plot(dates.index,water_volume_data,label='Water Storage Volume',color='orange',lw=2)
#plt.xlim('2006-01-01 00:00:00','2006-04-20 23:00:00')
#plt.ylim(100,350)
#plt.xticks(ticks=['2006-01-01 00:00:00','2006-02-01 00:00:00','2006-03-01 00:00:00','2006-04-01 00:00:00','2006-04-20 23:00:00'],labels=['2006-01-01','2006-02-01','2006-03-01','2006-04-01','2006-04-20'])
#plt.hlines(342.67,'2006-01-01 03:00:00','2006-04-20 23:00:00',label='$V_{max,lt}$ (=342.67 $m^3$)',lw=2)
#plt.ylabel('Wetland Storage Volume [$m^3$]',fontsize=13)
#ax3 =ax2.twinx()
#ax3.plot(dates.index, water_volume_data/342.67,lw=0)
#ax3.set_ylabel('Fract. Max. Water Storage Vol. [-]',fontsize=13)
#ax3.grid(b=None)
#ax2.tick_params(labelsize=12)
#ax3.tick_params(labelsize=12)
#ax2.set_xticklabels([])
#ax2.legend(loc='lower left',fontsize=15)
#
#plt.show()

###RAINFALL/DISCHARGE PLOT 2006

#rainfall = data_2006.iloc[2184:7320,15].values.astype(float)
#  
#outflow = data_2006.iloc[2184:7320,13].values.astype(float)
#n_1996 = 5136
#t = np.arange(1,5137)
#data1 = rainfall
#data2 = outflow
#
#dates=pd.DataFrame()
#dates['time']=data_2006.iloc[2184:7320,2]
#dates['time'].dt.strftime('%d-%m')
#dates.index = pd.to_datetime(dates.time, dayfirst=True, format='%Y/%m/%d', exact=True, infer_datetime_format=False)
#
#plt.style.use('ggplot')
#
#fig, ax1 = plt.subplots()
#
#color = 'tab:red'
#ax1.set_xlabel('Time [Year-Month]')
#ax1.set_ylabel('Outflow [$m^3\:h^{-1}$]', color=color)
#ax1.plot(dates.index,data2, color=color)
#ax1.tick_params(axis='y', labelcolor=color)
#ax1.set_xlim(['2006-04-01 00:00:00', '2006-11-01 01:00:00'])
#ax1.set_ylim([0,150000])
##ax1.hlines(y=(definitions_volume.discharge_total_per_culvert(2.61)*60*60),xmin='1996-04-01 00:00:00',xmax='1996-10-31 23:00:00')
#ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
#color = 'tab:blue'
#ax2.set_ylabel('Precipitation [$mm\: h^{-1}$]', color=color)  # we already handled the x-label with ax1
#ax2.bar(dates.index, data1, color=color,width=0.6)
#ax2.tick_params(axis='y', labelcolor=color)
#ax2.set_ylim(0,50)
#ax2.set_xlim(['2006-04-01 00:00:00', '2006-11-01 01:00:00'])
#ax2.grid(b=None)
#ax2.invert_yaxis()
#
#fig.set_figheight(4)
#fig.set_figwidth(12)
#fig.tight_layout()  # otherwise the right y-label is slightly clipped
#
#plt.show()



### PLOTS 2011
#plt.close()
path_2011 = r'results\2011.xlsx'
data_2011 = pd.read_excel(path_2011)

data_2011 = data_2011.reset_index()


### WATER STORAGE 2011

#delta_data = data_2011.iloc[0:2664,12]
#evaporation_data = data_2011.iloc[0:2664,6]
#groundwater_inflow_data = data_2011.iloc[0:2664,4]
#water_volume_data = data_2011.iloc[0:2664,11]
#area_data = data_2011.iloc[0:2664,5]
#
#dates=pd.DataFrame()
#dates['time']=data_2011.iloc[0:2664,2]
#dates['time'].dt.strftime('%d-%m')
#dates.index = pd.to_datetime(dates.time, dayfirst=True, format='%Y/%m/%d', exact=True, infer_datetime_format=False)
#
#gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2]) 
#
#ax1=plt.subplot(gs[1])
#
#plt.style.use('ggplot')
#plt.plot(dates.index,delta_data,label='$\Delta$ Storage Volume',color='red',lw=2)
#plt.plot(dates.index,evaporation_data*-1,label='Evaporation',ls='--',lw=1.5,color='green')
#plt.plot(dates.index,groundwater_inflow_data,label='Groundwater Inflow',lw=1.5,ls='--',color='blue')
#plt.hlines(0,'2011-01-01 00:00:00','2011-04-20 23:00:00',lw=2)
#plt.ylim([-0.5,2])
#plt.xlim('2011-01-01 03:00:00','2011-04-20 23:00:00')
#plt.xticks(ticks=['2011-01-01 03:00:00','2011-02-01 00:00:00','2011-03-01 00:00:00','2011-04-01 00:00:00','2011-04-20 23:00:00'],labels=['2011-01-01','2011-02-01','2011-03-01','2011-04-01','2011-04-20'])
#ax1.xaxis.set_tick_params(labelsize=12)
#ax1.yaxis.set_tick_params(labelsize=12)
#plt.ylabel('Flow [$m^3 \: h^{-1}$]',fontsize=14)
#plt.xlabel('Time [Year-Month-Day]',fontsize=14)
#plt.legend(fontsize=17)
#plt.tight_layout()
#
#ax2=plt.subplot(gs[0])
#plt.plot(dates.index,water_volume_data,label='Water Storage Volume',color='orange',lw=2)
#plt.xlim('2011-01-01 00:00:00','2011-04-20 23:00:00')
#plt.ylim(100,350)
#plt.xticks(ticks=['2011-01-01 00:00:00','2011-02-01 00:00:00','2011-03-01 00:00:00','2011-04-01 00:00:00','2011-04-20 23:00:00'],labels=['2011-01-01','2011-02-01','2011-03-01','2011-04-01','2011-04-20'])
#plt.hlines(342.67,'2011-01-01 03:00:00','2011-04-20 23:00:00',label='$V_{max,lt}$ (=342.67 $m^3$)',lw=2)
#plt.ylabel('Wetland Storage Volume [$m^3$]',fontsize=13)
#ax3 =ax2.twinx()
#ax3.plot(dates.index, water_volume_data/342.67,lw=0)
#ax3.set_ylabel('Fract. Max. Water Storage Vol. [-]',fontsize=13)
#ax3.grid(b=None)
#ax2.tick_params(labelsize=12)
#ax3.tick_params(labelsize=12)
#ax2.set_xticklabels([])
#ax2.legend(loc='lower left',fontsize=15)
#
#plt.show()

##RAINFALL/DISCHARGE PLOT 2011

#rainfall = data_2011.iloc[2160:7296,15].values.astype(float)
#  
#outflow = data_2011.iloc[2160:7296,13].values.astype(float)
#n_1996 = 5136
#t = np.arange(1,5137)
#data1 = rainfall
#data2 = outflow
#
#dates=pd.DataFrame()
#dates['time']=data_2011.iloc[2160:7296,2]
#dates['time'].dt.strftime('%d-%m')
#dates.index = pd.to_datetime(dates.time, dayfirst=True, format='%Y/%m/%d', exact=True, infer_datetime_format=False)
#
#plt.style.use('ggplot')
#
#fig, ax1 = plt.subplots()
#
#color = 'tab:red'
#ax1.set_xlabel('Time [Year-Month]')
#ax1.set_ylabel('Outflow [$m^3\:h^{-1}$]', color=color)
#ax1.plot(dates.index,data2, color=color)
#ax1.tick_params(axis='y', labelcolor=color)
#ax1.set_xlim(['2011-04-01 00:00:00', '2011-11-01 01:00:00'])
#ax1.set_ylim([0,150000])
##ax1.hlines(y=(definitions_volume.discharge_total_per_culvert(2.61)*60*60),xmin='1996-04-01 00:00:00',xmax='1996-10-31 23:00:00')
#ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
#color = 'tab:blue'
#ax2.set_ylabel('Precipitation [$mm\: h^{-1}$]', color=color)  # we already handled the x-label with ax1
#ax2.bar(dates.index, data1, color=color,width=0.6)
#ax2.tick_params(axis='y', labelcolor=color)
#ax2.set_ylim(0,50)
#ax2.set_xlim(['2011-04-01 00:00:00', '2011-11-01 01:00:00'])
#ax2.grid(b=None)
#ax2.invert_yaxis()
#
#fig.set_figheight(4)
#fig.set_figwidth(12)
#fig.tight_layout()  # otherwise the right y-label is slightly clipped
#
#plt.show()

###1996
wet_season_1996 = data_1996.iloc[2880:7296]
dry_season_1996 = data_1996.iloc[0:2904]


surface_inflow_dry_season_1996 = dry_season_1996['inflow'].sum()
surface_inflow_wet_season_1996 = wet_season_1996['inflow'].sum()

gw_inflow_dry_season_1996 = dry_season_1996['gw-inflow'].sum()
gw_inflow_wet_season_1996 = wet_season_1996['gw-inflow'].sum()

evaporation_dry_season_1996 = dry_season_1996['evap'].sum()
evaporation_wet_season_1996 = wet_season_1996['evap'].sum()

surface_outflow_dry_season_1996 = dry_season_1996['actual_outflow'].sum()
surface_outflow_wet_season_1996 = wet_season_1996['actual_outflow'].sum()

res_time_dry_season_1996 = dry_season_1996['residence-time'].mean()
res_time_wet_season_1996 = wet_season_1996['residence-time'].mean()

mean_volume_dry_season_1996 = dry_season_1996['w-vol-actual'].mean()
mean_volume_wet_season_1996 = wet_season_1996['w-vol-actual'].mean()

min_volume_dry_season_1996 = dry_season_1996['w-vol-actual'].min()
min_volume_wet_season_1996 = wet_season_1996['w-vol-actual'].min()


###2006
wet_season_2006 = data_2006.iloc[2880:7296]
dry_season_2006 = data_2006.iloc[0:2904]


surface_inflow_dry_season_2006 = dry_season_2006['inflow'].sum()
surface_inflow_wet_season_2006 = wet_season_2006['inflow'].sum()

gw_inflow_dry_season_2006 = dry_season_2006['gw-inflow'].sum()
gw_inflow_wet_season_2006 = wet_season_2006['gw-inflow'].sum()

evaporation_dry_season_2006 = dry_season_2006['evap'].sum()
evaporation_wet_season_2006 = wet_season_2006['evap'].sum()

surface_outflow_dry_season_2006 = dry_season_2006['actual_outflow'].sum()
surface_outflow_wet_season_2006 = wet_season_2006['actual_outflow'].sum()

res_time_dry_season_2006 = dry_season_2006['residence-time'].mean()
res_time_wet_season_2006 = wet_season_2006['residence-time'].mean()

mean_volume_dry_season_2006 = dry_season_2006['w-vol-actual'].mean()
mean_volume_wet_season_2006 = wet_season_2006['w-vol-actual'].mean()

min_volume_dry_season_2006 = dry_season_2006['w-vol-actual'].min()
min_volume_wet_season_2006 = wet_season_2006['w-vol-actual'].min()

###2011
wet_season_2011 = data_2011.iloc[2880:7296]
dry_season_2011 = data_2011.iloc[0:2904]


surface_inflow_dry_season_2011 = dry_season_2011['inflow'].sum()
surface_inflow_wet_season_2011 = wet_season_2011['inflow'].sum()

gw_inflow_dry_season_2011 = dry_season_2011['gw-inflow'].sum()
gw_inflow_wet_season_2011 = wet_season_2011['gw-inflow'].sum()

evaporation_dry_season_2011 = dry_season_2011['evap'].sum()
evaporation_wet_season_2011 = wet_season_2011['evap'].sum()

surface_outflow_dry_season_2011 = dry_season_2011['actual_outflow'].sum()
surface_outflow_wet_season_2011 = wet_season_2011['actual_outflow'].sum()

res_time_dry_season_2011 = dry_season_2011['residence-time'].mean()
res_time_wet_season_2011 = wet_season_2011['residence-time'].mean()

mean_volume_dry_season_2011 = dry_season_2011['w-vol-actual'].mean()
mean_volume_wet_season_2011 = wet_season_2011['w-vol-actual'].mean()

min_volume_dry_season_2011 = dry_season_2011['w-vol-actual'].min()
min_volume_wet_season_2011 = wet_season_2011['w-vol-actual'].min()