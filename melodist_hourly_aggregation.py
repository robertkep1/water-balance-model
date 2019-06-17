# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:04:31 2019

@author: rober
"""

import matplotlib
import matplotlib.pyplot as plt
import melodist
import numpy as np
import pandas as pd
import scipy.stats

def plot(obs, sim):
    plt.figure()
    ax = plt.gca()
    obs.loc[plot_period].plot(ax=ax, color='black', label='obs', lw=2)
    sim.loc[plot_period].plot(ax=ax)
    plt.legend()
    plt.show()

def calc_stats(obs, sim):
    df = pd.DataFrame(columns=['mean', 'std', 'r', 'rmse', 'nse'])

    obs = obs.loc[validation_period]
    sim = sim.loc[validation_period]
    
    df.loc['obs'] = obs.mean(), obs.std(), 1, 0, 1
    
    for c in sim.columns:
        osdf = pd.DataFrame(data=dict(obs=obs, sim=sim[c])).dropna(how='any')
        o = osdf.obs
        s = osdf.sim
    
        r = scipy.stats.pearsonr(o, s)[0]
        rmse = np.mean((o - s)**2)
        nse = 1 - np.sum((o - s)**2) / np.sum((o - o.mean())**2)
        df.loc[c] = s.mean(), s.std(), r, rmse, nse
    
    return df

def print_stats(obs, sim):
    df = calc_stats(obs, sim)
    html = df.round(2).style
    return html

path_inp = #input for the hourly data
path_inp_daily = #input for the daily data that will be disaggregated to hourly data
longitude = 8.86
latitude = 51.00
timezone = 1

daily_data=pd.read_csv(path_inp_daily, index_col=0, parse_dates=True)

calibration_period = slice('2003-01-01', '2017-12-31')
validation_period = slice('2018-01-01', '2018-12-31')
plot_period = slice('2016-07-03', '2016-08-14')

data_obs_hourly = pd.read_csv(path_inp, index_col=0, parse_dates=True)
data_obs_hourly = melodist.util.drop_incomplete_days(data_obs_hourly)


station = melodist.Station(lon=longitude, lat=latitude,
                          timezone=timezone,
                          data_daily=daily_data)

station.statistics = melodist.StationStatistics(data_obs_hourly.loc[calibration_period])

stats = station.statistics
stats.calc_precipitation_stats()

precipdf = pd.DataFrame()
for method in ('equal', 'cascade'):
    station.disaggregate_precipitation(method=method)
    precipdf[method] = station.data_disagg.precip
    
precipdf['cascade'].to_csv (r'C:\Users\rober\OneDrive\Internship\SWAT model\inputdataNOTDELETE\climatedata\robert.keppler7jAESi\output_cascade_hourly.csv', index=False)

