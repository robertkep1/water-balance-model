# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 16:10:05 2019

@author: robert
"""

#!/usr/bin/env python3


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
import time
from scipy import linalg as LA
from scipy.stats import norm
import scipy.stats as stats
import math

# ----- SCRIPT EXECUTION TIMER ----
start_time = time.time()

# ---- DATA ----

xls   	= pd.ExcelFile(r'results\inflow\hydraulic_conductivity.xlsx')
data 	= pd.read_excel (xls, 'Sheet2')


t1 		= data.iloc[47:57,2].values.astype(float)
y1 		= data.iloc[47:57,4].values.astype(float)
l1 		= data.iloc[44,7]
h01 	= y1[0]

t2 		= data.iloc[61:71,2].values.astype(float)
y2 		= data.iloc[61:71,4].values.astype(float)
l2 		= data.iloc[58,7]
h02 	= y2[0]

t3		= data.iloc[77:87,2].values.astype(float)
y3 		= data.iloc[77:87,4].values.astype(float)
l3 		= data.iloc[74,7]
h03 	= y3[0]

t4 		= data.iloc[92:102,2].values.astype(float)
y4 		= data.iloc[92:102,4].values.astype(float)
l4 		= data.iloc[89,7]
h04 	= y4[0]

t5 		= data.iloc[107:117,2].values.astype(float)
y5 		= data.iloc[107:117,4].values.astype(float)
l5 		= data.iloc[104,7]
h05 	= y5[0]

t6 		= data.iloc[120:130,2].values.astype(float)
y6 		= data.iloc[120:130,4].values.astype(float)
l6 		= data.iloc[119,7]
h06 	= y6[0]

t7	 	= data.iloc[135:145,2].values.astype(float)
y7 		= data.iloc[135:145,4].values.astype(float)
l7 		= data.iloc[132,7]
h07 	= y7[0]

t8	 	= data.iloc[150:160,2].values.astype(float)
y8 		= data.iloc[150:160,4].values.astype(float)
l8 		= data.iloc[147,7]
h08 	= y8[0]

t9	 	= data.iloc[165:175,2].values.astype(float)
y9 		= data.iloc[165:175,4].values.astype(float)
l9 		= data.iloc[162,7]
h09 	= y9[0]

t10	 	= data.iloc[180:191,2].values.astype(float)
y10 	= data.iloc[180:191,4].values.astype(float)
l10 	= data.iloc[177,7]
h010 	= y10[0]

t11	 	= data.iloc[196:207,2].values.astype(float)
y11 	= data.iloc[196:207,4].values.astype(float)
l11 	= data.iloc[193,7]
h011 	= y11[0]

t12	 	= data.iloc[212:223,2].values.astype(float)
y12 	= data.iloc[212:223,4].values.astype(float)
l12 	= data.iloc[209,7]
h012 	= y12[0]

t13	 	= data.iloc[228:239,2].values.astype(float)
y13 	= data.iloc[228:239,4].values.astype(float)
l13 	= data.iloc[225,7]
h013 	= y13[0]

t14	 	= data.iloc[244:255,2].values.astype(float)
y14 	= data.iloc[244:255,4].values.astype(float)
l14 	= data.iloc[241,7]
h014 	= y14[0]

t15	 	= data.iloc[260:271,2].values.astype(float)
y15 	= data.iloc[260:271,4].values.astype(float)
l15 	= data.iloc[257,7]
h015 	= y15[0]

t16	 	= data.iloc[276:287,2].values.astype(float)
y16 	= data.iloc[276:287,4].values.astype(float)
l16 	= data.iloc[273,7]
h016 	= y16[0]

t17	 	= data.iloc[292:303,2].values.astype(float)
y17 	= data.iloc[292:303,4].values.astype(float)
l17 	= data.iloc[289,7]
h017 	= y17[0]

t18	 	= data.iloc[308:319,2].values.astype(float)
y18 	= data.iloc[308:319,4].values.astype(float)
l18 	= data.iloc[305,7]
h018 	= y18[0]

t19	 	= data.iloc[324:331,2].values.astype(float)
y19 	= data.iloc[324:331,4].values.astype(float)
l19 	= data.iloc[321,7]
h019 	= y19[0]

t20	 	= data.iloc[336:343,2].values.astype(float)
y20 	= data.iloc[336:343,4].values.astype(float)
l20 	= data.iloc[333,7]
h020 	= y20[0]

t21	 	= data.iloc[348:355,2].values.astype(float)
y21 	= data.iloc[348:355,4].values.astype(float)
l21 	= data.iloc[345,7]
h021 	= y21[0]

t22	 	= data.iloc[360:367,2].values.astype(float)
y22 	= data.iloc[360:367,4].values.astype(float)
l22 	= data.iloc[357,7]
h022 	= y22[0]

t23	 	= data.iloc[372:379,2].values.astype(float)
y23 	= data.iloc[372:379,4].values.astype(float)
l23 	= data.iloc[369,7]
h023 	= y23[0]

t24	 	= data.iloc[384:391,2].values.astype(float)
y24 	= data.iloc[384:391,4].values.astype(float)
l24 	= data.iloc[381,7]
h024 	= y24[0]

t25	 	= data.iloc[396:403,2].values.astype(float)
y25 	= data.iloc[396:403,4].values.astype(float)
l25 	= data.iloc[393,7]
h025 	= y25[0]

t26	 	= data.iloc[408:415,2].values.astype(float)
y26 	= data.iloc[408:415,4].values.astype(float)
l26 	= data.iloc[405,7]
h026 	= y26[0]

t27	 	= data.iloc[420:426,2].values.astype(float)
y27 	= data.iloc[420:426,4].values.astype(float)
l27 	= data.iloc[417,7]
h027 	= y27[0]

t28	 	= data.iloc[431:437,2].values.astype(float)
y28 	= data.iloc[431:437,4].values.astype(float)
l28 	= data.iloc[428,7]
h028 	= y28[0]

t29	 	= data.iloc[442:448,2].values.astype(float)
y29 	= data.iloc[442:448,4].values.astype(float)
l29 	= data.iloc[439,7]
h029 	= y29[0]

t30	 	= data.iloc[453:459,2].values.astype(float)
y30 	= data.iloc[453:459,4].values.astype(float)
l30 	= data.iloc[450,7]
h030 	= y30[0]

t31	 	= data.iloc[463:469,2].values.astype(float)
y31 	= data.iloc[463:469,4].values.astype(float)
l31 	= data.iloc[460,7]
h031 	= y31[0]

t32	 	= data.iloc[474:480,2].values.astype(float)
y32 	= data.iloc[474:480,4].values.astype(float)
l32 	= data.iloc[471,7]
h032 	= y32[0]

t_data_list = (t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20,t21,t22,t23,t24,t25,t26,t27,t28,t29,t30,t31,t32)
y_data_list = (y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,y15,y16,y17,y18,y19,y20,y21,y22,y23,y24,y25,y26,y27,y28,y29,y30,y31,y32)
h0_list 	= (h01,h02,h03,h04,h05,h06,h07,h08,h09,h010,h011,h012,h013,h014,h015,h016,h017,h018,h019,h020,h021,h022,h023,h024,h025,h026,h027,h028,h029,h030,h031,h032)
l_list  	= (l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13,l14,l15,l16,l17,l18,l19,l20,l21,l22,l23,l24,l25,l26,l27,l28,l29,l30,l31,l32)

t 			= np.linspace(0,70,100)
a 			= data.iloc[0,13]#.values.astype(float)

### WORKING AREA
def function (p1, p2):
    y = p1 + (h0 - p1) * np.exp(-(p2*t)/l)
    return y


def correlation(result):

#    # The function returns jac, the sensitivity matrix X
    # (where (jac jac.T) is a Gauss-Newton approximation of
    # the Hessian of residuals)
    X = result['jac']
#    print(np.dot(X.T, X))
   
    # Residuals are returned as fun
    res = result['fun']
    
	# Calculated variance of residuals
    N , p = X.shape[0], X.shape[1]
    varres = 1/(N-p) * np.dot(res.T,res)

	# Covariance matrix of parameters
    covp = varres * LA.inv(np.dot(X.T, X))
#    print('Covariance matrix of parameters \n {}, \n'.format(covp))    
    
	# Standard deviations of the parameters
    sd = np.sqrt(np.diag(covp))
#    print('Standard deviations of parameters \n {0} \n'.format(sd))
    
#	 Correlation coefficients
    cc = np.empty_like(covp)
    for i,sdi in enumerate(sd):
        for j,sdj in enumerate(sd):
            cc[i,j] = covp[i,j]/(sdi*sdj)
    np.set_printoptions(precision=4)
#    print('Correlation coefficients of parameters \n {} \n'.format(cc))
    
	# Mean squared error
    mse = 2*result['cost']/X.shape[0]
    rmse = np.sqrt(mse)
#    print('Root mean square error \n {}'.format(rmse))
    
    return X,covp,sd,cc,mse,rmse


# ---- CALIBRATION ----

### Calibration function
def fsim(p0):
    p1,p2 = p0
    y = p1 + (h0 - p1) * np.exp(-(p2*t)/l)
    return y

### Initial guess
p0 = [0.1,0.2]

### Bounds
bounds_param = ([.05,0],[0.5,1])
   
### Residuals
def residuals(p0,fsim,tsim,t_data,y_data):
    ysim 		= np.c_[fsim(p0)] # Use column vector
    f_interp 	= interp1d(tsim,ysim,axis=0)
    y 			= f_interp(t_data)     # Interpolate simulation results over t_data
    y_data 		= np.c_[y_data]
    residuals 	= np.sum(np.abs(y_data-y),axis=1)
    return residuals

### Calibration routine
n_datasets 				= len(t_data_list)
cov_matrix_list 		= np.array(np.zeros(len(t_data_list)), dtype=object)
p1 						= np.zeros(len(t_data_list))
p2 						= np.zeros(len(t_data_list))
root_mean_sq_err_list_p1= np.zeros(len(t_data_list))
root_mean_sq_err_list_p2= np.zeros(len(t_data_list))
std_dev_list_p1			= np.array(np.zeros(len(t_data_list)), dtype=object)
std_dev_list_p2			= np.array(np.zeros(len(t_data_list)), dtype=object)

for i in np.arange(n_datasets):
	h0 							= h0_list[i]
	l 							= l_list[i]
	t_data 						= t_data_list[i]
	y_data 						= y_data_list[i]
	estimation_res 				= least_squares(residuals,p0,args=(fsim,t,t_data,y_data),bounds=bounds_param)
	cov_matrix_list[i] 			= correlation(estimation_res)[1]
	root_mean_sq_err_list_p1[i] = correlation(estimation_res)[4]
	root_mean_sq_err_list_p2[i] = correlation(estimation_res)[5]
	std_dev_list_p1[i] 			= correlation(estimation_res)[2][0]
	std_dev_list_p2[i] 			= correlation(estimation_res)[2][1]
	p_est 						= estimation_res['x']
	p1[i],p2[i] 				= p_est
	

y_sim1 = p1[0] + (h01 - p1[0]) * np.exp(-(p2[0]*t)/l1)-h01
y_sim2 = p1[1] + (h02 - p1[1]) * np.exp(-(p2[1]*t)/l2)-h02
y_sim3 = p1[2] + (h03 - p1[2]) * np.exp(-(p2[2]*t)/l3)-h03
y_sim4 = p1[3] + (h04 - p1[3]) * np.exp(-(p2[3]*t)/l4)-h04
y_sim5 = p1[4] + (h05 - p1[4]) * np.exp(-(p2[4]*t)/l5)-h05
y_sim6 = p1[5] + (h06 - p1[5]) * np.exp(-(p2[5]*t)/l6)-h06
y_sim7 = p1[6] + (h07 - p1[6]) * np.exp(-(p2[6]*t)/l7)-h07
y_sim8 = p1[7] + (h08 - p1[7]) * np.exp(-(p2[7]*t)/l8)-h08
y_sim9 = p1[8] + (h09 - p1[8]) * np.exp(-(p2[8]*t)/l9)-h09
y_sim10 = p1[9] + (h010 - p1[9]) * np.exp(-(p2[9]*t)/l10)-h010
y_sim11 = p1[10] + (h011 - p1[10]) * np.exp(-(p2[10]*t)/l11)-h011
y_sim12 = p1[11] + (h012 - p1[11]) * np.exp(-(p2[11]*t)/l12)-h012
y_sim13 = p1[12] + (h013 - p1[12]) * np.exp(-(p2[12]*t)/l13)-h013
y_sim14 = p1[13] + (h014 - p1[13]) * np.exp(-(p2[13]*t)/l14)-h014
y_sim15 = p1[14] + (h015 - p1[14]) * np.exp(-(p2[14]*t)/l15)-h015
y_sim16 = p1[15] + (h016 - p1[15]) * np.exp(-(p2[15]*t)/l16)-h016
y_sim17 = p1[16] + (h017 - p1[16]) * np.exp(-(p2[16]*t)/l17)-h017
y_sim18 = p1[17] + (h018 - p1[17]) * np.exp(-(p2[17]*t)/l18)-h018
y_sim19 = p1[18] + (h019 - p1[18]) * np.exp(-(p2[18]*t)/l19)-h019
y_sim20 = p1[19] + (h020 - p1[19]) * np.exp(-(p2[19]*t)/l20)-h020
y_sim21 = p1[20] + (h021 - p1[20]) * np.exp(-(p2[20]*t)/l21)-h021
y_sim22 = p1[21] + (h022 - p1[21]) * np.exp(-(p2[21]*t)/l22)-h022
y_sim23 = p1[22] + (h023 - p1[22]) * np.exp(-(p2[22]*t)/l23)-h023
y_sim24 = p1[23] + (h024 - p1[23]) * np.exp(-(p2[23]*t)/l24)-h024
y_sim25 = p1[24] + (h025 - p1[24]) * np.exp(-(p2[24]*t)/l25)-h025
y_sim26 = p1[25] + (h026 - p1[25]) * np.exp(-(p2[25]*t)/l26)-h026
y_sim27 = p1[26] + (h027 - p1[26]) * np.exp(-(p2[26]*t)/l27)-h027
y_sim28 = p1[27] + (h028 - p1[27]) * np.exp(-(p2[27]*t)/l28)-h028
y_sim29 = p1[28] + (h029 - p1[28]) * np.exp(-(p2[28]*t)/l29)-h029
y_sim30 = p1[29] + (h030 - p1[29]) * np.exp(-(p2[29]*t)/l30)-h030
y_sim31 = p1[30] + (h031 - p1[30]) * np.exp(-(p2[30]*t)/l31)-h031
y_sim32 = p1[31] + (h032 - p1[31]) * np.exp(-(p2[31]*t)/l32)-h032


###Set first value as zero
y_sim1 [0]= 0
y_sim2 [0]= 0
y_sim3 [0]= 0
y_sim4 [0]= 0
y_sim5 [0]= 0
y_sim6 [0]= 0
y_sim7 [0]= 0
y_sim8 [0]= 0
y_sim9 [0]= 0
y_sim10[0] = 0
y_sim11[0] = 0
y_sim12[0] = 0
y_sim13[0] = 0
y_sim14[0] = 0
y_sim15[0] = 0
y_sim16[0] = 0
y_sim17[0] = 0
y_sim18[0] = 0
y_sim19[0] = 0
y_sim20[0] = 0
y_sim21[0] = 0
y_sim22[0] = 0
y_sim23[0] = 0
y_sim24[0] = 0
y_sim25[0] = 0
y_sim26[0] = 0
y_sim27[0] = 0
y_sim28[0] = 0
y_sim29[0] = 0
y_sim30[0] = 0
y_sim31[0] = 0
y_sim32[0] = 0


### infiltration rate formula and loop with parameters hB(p1) and k(p2)

def actual_infiltration (k,hB,h0,l):
    I = k * ((h0 - hB)/l)
    return I

inf_rate_list 	= np.zeros(len(t_data_list))

for i in np.arange(n_datasets):
	hB 	= p1[i]
	k 	= p2[i]
	l 	= l_list[i]
	h0 	= h0_list[i]
	inf_rate_list[i] = actual_infiltration(k,hB,h0,l)
	
inf_rate_list_cmd = inf_rate_list*100*24 #infiltration rate list in cm/d


### R-squared calculations

r_squared_list 		= np.zeros(n_datasets)
mse_list 			= np.zeros(n_datasets) #mean standard error 
y_fit_list 			= np.array(np.zeros(n_datasets), dtype=object)

for i in np.arange(n_datasets):
	h0_temp 			= h0_list[i]
	l_temp 				= l_list[i]
	n_data 				= len(y_data_list[i])
#	print (n_data)
#	print(i)
	y_data_temp 		= y_data_list[i]
	p1_temp 			= p1[i]
	p2_temp 			= p2[i]
	t_data_temp 		= t_data_list[i]
	y_mean 				= np.mean(y_data_temp)
	zaehler_list 		= np.zeros(n_data)
	nenner_list			= np.zeros(n_data)
	mse_temp_list 		= np.zeros(n_data)
	y_fit_list_temp 	= np.zeros(n_data)
	
	for j in np.arange(n_data):	
		y_r2 	 			= y_data_temp[j]
		t_r2 				= t_data_temp[j]
		y_fit 				= p1_temp + (h0_temp - p1_temp) * np.exp(-(p2_temp*t_r2)/l_temp)
		zaehler 			= np.square(y_r2-y_fit)
		nenner 				= np.square(y_r2-y_mean)
		zaehler_list[j]		= zaehler
		nenner_list[j]		= nenner
		y_fit_list_temp[j] 	= y_fit

	y_fit_list[i] 		= y_fit_list_temp
	summe_zaehler 		= np.sum(zaehler_list)
	summe_nenner 		= np.sum(nenner_list)
	mse_list[i]			= (1/n_data)*summe_zaehler
	r_squared_list[i] 	= 1-(summe_zaehler/summe_nenner)
	
### Monte-Carlo-simulation
plt.close()
	
loop_rounds_mc 	= 10000 #number of loops 
loop_list_all 	= np.array(np.zeros(n_datasets), dtype=object)
loop_list_mean 	= np.zeros(n_datasets) #mean value 
loop_list_var 	= np.zeros(n_datasets) #variance list
loop_list_std   = np.zeros(n_datasets) #standard deviation list
loop_list_max 	= np.zeros(n_datasets) #maximum values of every list
loop_list_min 	= np.zeros(n_datasets) #minimum values of every list
loop_list_median= np.zeros(n_datasets) 

for j in np.arange(n_datasets):
    h0_mc 			= h0_list[j]
    l_mc 			= l_list[j]
    p1_mc 			= p1[j]
    p2_mc 			= p2[j]
    rmse_p1_mc 	 	= root_mean_sq_err_list_p1[j] #root mean square error from p1 to be used as standard deviation in random number generator
    rmse_p2_mc 	 	= root_mean_sq_err_list_p2[j]
    std_dev_list_p1_mc = std_dev_list_p1[j]
    std_dev_list_p2_mc = std_dev_list_p2[j]
    loop_list		= np.zeros(loop_rounds_mc)
    
    for i in np.arange(loop_rounds_mc):
        hB_mc 				= np.random.normal(loc=p1_mc, scale=std_dev_list_p1_mc)
        k_mc 				= np.random.normal(loc=p2_mc, scale=std_dev_list_p2_mc)
		
        infiltration 		= k_mc * ((h0_mc - hB_mc)/l_mc)
        loop_list[i] 		= infiltration
        
    loop_list_cmd 			= loop_list*24*100*-1 #seepage rate (defined as inverse infiltration rate) in cm/day
    loop_list_all[j] 		= loop_list_cmd
    average					= np.mean(loop_list_cmd)
    variance 				= np.var(loop_list_cmd)
    median                  = np.median(loop_list_cmd)
    loop_list_max[j] 		= max(loop_list_cmd)
    loop_list_min[j] 		= min(loop_list_cmd)
    loop_list_mean[j]		= average
    loop_list_var[j] 		= variance
    loop_list_std[j] 		= np.sqrt(variance)
    loop_list_median[j]     = median


###Overall average value for infiltration as derived from all 32 infiltration measurement points

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))

   
weights = 1/(loop_list_var)    

average_overall = weighted_avg_and_std(values=loop_list_mean, weights=weights)


###normal distribution curve plot for resulting final average value
#
#mu = average_overall[0]
#sigma = average_overall[1]
#x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
#
#x_pdf_average = norm.pdf(x,average_overall[0],average_overall[1])
##samples=loop_list_mean
#x_pdf_max_average=max(stats.norm.pdf(x, mu, sigma))
#annotation_string=r"""$\bar x$ = {0:.4f}
#    $\sigma$={1:.2f}""".format(average_overall[0],sigma)
#
#bin_means, bin_edges, binnumber = stats.binned_statistic(x, x_pdf_average, statistic='mean', bins=25)
#bin_width = (bin_edges[1] - bin_edges[0])
#bin_centers = bin_edges[1:] - bin_width/2
#
#
#
#
#plt.figure()
##plt.hist(samples, bins=70, normed=True, histtype='stepfilled', alpha=0.2, label='histogram of data')
#plt.plot(x, stats.norm.pdf(x, mu, sigma))
#plt.vlines(x=average_overall[0], ymin=0, ymax = x_pdf_max_average, linewidth=1, color='k')
#plt.legend(fontsize=10)
#plt.annotate(annotation_string, (0.72,0.88) ,xycoords='axes fraction', ha='center', va='center', size='10') #'$\overline {x}$ = %.3f' % loop_list_mean[i]
#plt.ylim(bottom=0)
#plt.xlabel('Seepage rate [cm/d]')
#plt.ylabel('Seepage rate [cm/d]')
#plt.show()



###normal distribution curve plot Monte-Carlo-simulation for all 32 infiltration points

#symbol='x'
#for i in np.arange(n_datasets):
#	plt.subplot(4,8,i+1)
#	mu = loop_list_mean[i]
#	variance = loop_list_var[i]
#	sigma = math.sqrt(variance)
#	x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
#	
#	x_pdf = norm.pdf(x,loop_list_mean[i],loop_list_var[i]) # for loop_list_std loop_list_var replaced
#	samples=loop_list_all[i]
#	x_pdf_max=max(stats.norm.pdf(x, mu, sigma))
#	annotation_string=r"""$\bar x$ = {0:.2f}
#    $\sigma$={1:.2f}""".format(loop_list_mean[i],sigma)
#	bin_means, bin_edges, binnumber = stats.binned_statistic(x, x_pdf, statistic='mean', bins=25)
#	bin_width = (bin_edges[1] - bin_edges[0])
#	bin_centers = bin_edges[1:] - bin_width/2
#	
#	plt.hist(samples, bins=70, normed=True, histtype='stepfilled', alpha=0.2, label='histogram of data',color='r')
#	plt.plot(x, stats.norm.pdf(x, mu, sigma))
#	plt.vlines(x=loop_list_mean[i], ymin=0, ymax = x_pdf_max, linewidth=1, color='k')
#	plt.annotate(annotation_string, (0.72,0.88) ,xycoords='axes fraction', ha='center', va='center', size='10') #'$\overline {x}$ = %.3f' % loop_list_mean[i]
#
#plt.subplots_adjust(wspace=0.4, hspace=0.2, top=0.98, bottom=0.06, left=0.04, right=0.99)
#
#plt.show()



#plt.close()


### regression curve plot
#
#
#plt.style.use('ggplot')
#plt.figure('Regression Curve')
#list_number=np.arange(1,33)
#
#strings = ('INF1','INF2','INF3','INF4','INF5','INF6','INF7','INF8','INF9','INF10','INF11','INF12','INF13','INF14','INF15','INF16','INF17','INF18','INF19','INF20','INF21','INF22','INF23','INF24','INF25','INF26','INF27','INF28','INF29','INF30','INF31','INF32')
#
#y_sim_list = (y_sim1,y_sim2,y_sim3,y_sim4,y_sim5,y_sim6,y_sim7,y_sim8,y_sim9,y_sim10,y_sim11,y_sim12,y_sim13,y_sim14,y_sim15,y_sim16,y_sim17,y_sim18,y_sim19,y_sim20,y_sim21,y_sim22,y_sim23,y_sim24,y_sim25,y_sim26,y_sim27,y_sim28,y_sim29,y_sim30,y_sim31,y_sim32)
#
#for i in np.arange(n_datasets):
#	plt.subplot(4,8,i+1)
#	plt.plot(t,y_sim_list[i]*100)#label=r'$Model$')
#	plt.scatter(t_data_list[i],(y_data_list[i]-h0_list[i])*100)#label=r'$Dataset$'
#	plt.annotate(strings[i], (0.2,0.9), xycoords='axes fraction', ha='center', va='center', size='12', fontweight='bold')
#	plt.annotate('Corr. coef = %.3f' % r_squared_list[i], (0.6,0.11) ,xycoords='axes fraction', ha='center', va='center', size='10')
#plt.subplots_adjust(wspace=0.4, hspace=0.2, top=0.99, bottom=0.065, left=0.06, right=0.995)
#plt.legend()



###BOXPLOT RESULTS

### Boxplot results SD values smaller than 5

#plt.style.use('ggplot')
#values_var_lower_2 = ('INF1','INF2','INF3','INF4','INF5','INF7','INF8','INF9','INF10','INF11','INF12','INF13','INF14','INF15','INF16','INF17','INF18','INF25','INF27','INF32')
#
#data_small_to_plot = [loop_list_all[0], loop_list_all[1], loop_list_all[2], loop_list_all[3], loop_list_all[4], loop_list_all[6], loop_list_all[7], loop_list_all[8], loop_list_all[9], loop_list_all[10], loop_list_all[11], loop_list_all[12], loop_list_all[13], loop_list_all[14], loop_list_all[15], loop_list_all[16], loop_list_all[17], loop_list_all[24],loop_list_all[26], loop_list_all[31]]
#  
#fig = plt.figure(1, figsize=(9, 6))
#ax = fig.add_subplot(111)
#bp_small = ax.boxplot(data_small_to_plot, patch_artist=True,showfliers=False)
#
#for box in bp_small['boxes']:
#    # change outline color
#    box.set( color='#7570b3',linewidth=1.7)
#    # change fill color
#    box.set( facecolor = '#1b9e77' )
#
#for whisker in bp_small['whiskers']:
#    whisker.set(color='#7570b3',linewidth=1.3)
#
#for median in bp_small['medians']:
#    median.set(color='r', linewidth=1.5)
#
#ax.set_xticklabels(labels=values_var_lower_2,size='12')
##plt.set_yticklabels(size='13')
#plt.ylabel('Seepage Rate [cm/d]', size='15')
#plt.xlabel('Name Infiltration Point', size='15')
#plt.tight_layout()
###fig.savefig(r'C:\Users\rober\OneDrive\Internship\Own writings\Latex\thesis structure\images\infiltration_small.png',dpi=700,bbox_inches='tight')

###Boxplot results SD between 5 and 500

#plt.style.use('ggplot')
#values_var_middle = ('INF6','INF20','INF21','INF22','INF23','INF24','INF26')
#
#data_middle_to_plot = [loop_list_all[5], loop_list_all[19], loop_list_all[20], loop_list_all[21], loop_list_all[22], loop_list_all[23], loop_list_all[25]]
#  
#fig = plt.figure(1, figsize=(9, 6))
#ax = fig.add_subplot(111)
#bp_middle = ax.boxplot(data_middle_to_plot, patch_artist=True,showfliers=False)
#
#for box in bp_middle['boxes']:
#    # change outline color
#    box.set( color='#7570b3',linewidth=1.7)
#    # change fill color
#    box.set( facecolor = '#1b9e77' )
#
#for whisker in bp_middle['whiskers']:
#    whisker.set(color='#7570b3',linewidth=1.3)
#
#for median in bp_middle['medians']:
#    median.set(color='r', linewidth=1.5)
#
#ax.set_xticklabels(labels=values_var_middle,size=13)
#plt.tight_layout()
#plt.ylabel('Seepage Rate [cm/d]', size='15')
#plt.xlabel('Name Infiltration Point', size='15')
###fig.savefig(r'C:\Users\rober\OneDrive\Internship\Own writings\Latex\thesis structure\images\infiltration_middle.png',dpi=700,bbox_inches='tight')
#fig.show()

###Boxplot results SD greater than 500

#plt.style.use('ggplot')
#values_var_large = ('INF19','INF28','INF29','INF30','INF31')
#
#data_large_to_plot = [loop_list_all[18], loop_list_all[27], loop_list_all[28], loop_list_all[29], loop_list_all[30]]
#  
#fig = plt.figure(1, figsize=(9, 6))
#ax = fig.add_subplot(111)
#bp_large = ax.boxplot(data_large_to_plot, patch_artist=True,showfliers=False)
#
#for box in bp_large['boxes']:
#    # change outline color
#    box.set( color='#7570b3',linewidth=1.7)
#    # change fill color
#    box.set( facecolor = '#1b9e77' )
#
#for whisker in bp_large['whiskers']:
#    whisker.set(color='#7570b3',linewidth=1.3)
#
#for median in bp_large['medians']:
#    median.set(color='r', linewidth=1.5)
#
#ax.set_xticklabels(labels=values_var_large,size=13)
#plt.tight_layout()
#plt.ylabel('Seepage Rate [cm/d]', size='15')
#plt.xlabel('Name Infiltration Point', size='15')
###fig.savefig(r'C:\Users\rober\OneDrive\Internship\Own writings\Latex\thesis structure\images\infiltration_large.png',dpi=700,bbox_inches='tight')
#fig.show()

print("- %s seconds -" % (time.time() - start_time))