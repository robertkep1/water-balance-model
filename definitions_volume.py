# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:06:12 2019

@author: rober
"""

import numpy as np
import pandas as pd
import scipy.optimize as optimize
import math
from operator import add

### --- calculation of discharge through culverts as defined by: Bodhaine, George Lawrence. Measurement of peak discharge at culverts by indirect methods. US Government Printing Office, 1968. ---

### specific pipe coefficients

pipe_coefficients       = pd.read_excel(r'Culvert discharge\coefficients_pipe_c.xlsx')
gravitation             = float(9.81)
manning_coeff           = float(0.015)
pipe_diameter           = float(0.9)
ratio_inl_outl          = 4/5


### type 3 flow as defined by Bodhaine  (1968)

def type_3_flow(water_level_entrance,water_level_outlet,pipe_length,elevation,discharge_guess):

    def ratio_headwater_pipe ():
        ratio = (water_level_entrance-elevation)/pipe_diameter
        return float(ratio)
    
    ratio_headwater=ratio_headwater_pipe()
#    print('ratio_headwater',ratio_headwater)
    def discharge_coefficient():   #computes discharge coefficient C
        s=ratio_headwater
        c = 0.886584 + 0.2150039246039883*s - 0.29569966842302525*s**2 + 0.08032610044946863*s**3
        return float(c)
#    print('discharge-coeff',discharge_coefficient())
    def energy_equation (x):
#        print('energy_eq',- (x**2/(2*gravitation*discharge_coefficient())) + water_level_entrance - elevation)        
        return float(- (x**2/(2*gravitation*discharge_coefficient())) + water_level_entrance - elevation)

    
    def d_2():
#        print('d2',optimize.fixed_point(energy_equation,0)) 
        d_2=optimize.fixed_point(energy_equation,0)
        if d_2>pipe_diameter:
            d_2=pipe_diameter
        return float(d_2)
               
    def frict_loss_inl_outl():
        ratio_2                 = round(d_2()/pipe_diameter, 2) #computes d_2/D      
        if ratio_2==0:
            ratio_2=0.01
        coefficient_row_2       = pipe_coefficients.loc[pipe_coefficients['d/D']==ratio_2]    #selects df row for Ck according to ratio of d/D
        coefficient_2           = coefficient_row_2['Ck'].values.astype(float)     #selects value for Ck from previously selected df row
        coefficient_2           = coefficient_2.item()
        conveyance_coeff_2      = float(((pipe_diameter**(8/3))/manning_coeff)*coefficient_2)    #computes conveyance coefficient K2
        
        ratio_3                 = round (water_level_outlet/pipe_diameter, 2) #computes d3 (=h4) / D
        if ratio_3==0:
            ratio_3=0.01
        coefficient_row_3       = pipe_coefficients.loc[pipe_coefficients['d/D']==ratio_3]
        coefficient_3           = coefficient_row_3['Ck'].values.astype(float)
        coefficient_3           = coefficient_3.item()
        conveyance_coeff_3      = float(((pipe_diameter**(8/3))/manning_coeff)*coefficient_3)  #computes conveyance coefficient K3
#        print('coeff2',coefficient_2)
#        print('ratio2',ratio_2)
#        print('type',(type(coefficient_2)))
        return float((discharge_guess**2*pipe_length)/(conveyance_coeff_2*conveyance_coeff_3)) #computes friction loss between inlet and outlet

    def area_parameter():
        ratio_3                 = round (water_level_outlet/pipe_diameter, 2) #computes d3 (=h4) / D
        if ratio_3==0:
            ratio_3=0.01
        coefficient_row_3       = pipe_coefficients.loc[pipe_coefficients['d/D']==ratio_3]
        coefficient_3           = coefficient_row_3['Co'].values.astype(float)
        return float(coefficient_3 * pipe_diameter**2) #returns area parameter A

    def velocity_head_2(): #computes v_2^2/(2g*C^2)
        v_2 = float(np.sqrt(2)*discharge_coefficient() * np.sqrt(gravitation*(water_level_entrance-elevation-d_2())))
        return float(v_2**2 / (2*gravitation*(discharge_coefficient()**2)))

    def velocity_head_1(): #computes v_1^2/(2g*C^2)
        return float(d_2() + velocity_head_2() - water_level_entrance + elevation)
        
    square_root=water_level_entrance + velocity_head_1() - water_level_outlet - frict_loss_inl_outl()
    if square_root <= 0:
        square_root = 0.000001
    else:
        square_root=water_level_entrance + velocity_head_1() - water_level_outlet - frict_loss_inl_outl()

    return float(discharge_coefficient()*area_parameter()*np.sqrt(2*gravitation*(square_root)))


def initial_guess(water_level_outlet, gravitation, water_level_entrance):
    
    def area_parameter():
        ratio_3                 = round (water_level_outlet/pipe_diameter, 2) #computes d3 (=h4) / D
        if ratio_3==0:
            ratio_3=0.01
        coefficient_row_3       = pipe_coefficients.loc[pipe_coefficients['d/D']==ratio_3]
        coefficient_3           = coefficient_row_3['d/D'].values[0]
     
        return float(coefficient_3 * pipe_diameter**2) #returns area parameter A
    
    return float(0.95*area_parameter()*np.sqrt(2*gravitation*(water_level_entrance-water_level_outlet)))


###type 4 flow as defined by Bodhaine (1968)

def type_4_flow(water_level_entrance,water_level_outlet,pipe_length,elevation):
    
    pipe_area = float(pipe_diameter*math.pi) #A0
    
    def ratio_headwater_pipe ():
        ratio = (water_level_entrance-elevation)/pipe_diameter
        return float(ratio)

    ratio_headwater=ratio_headwater_pipe()

    def discharge_coefficient():   #computes discharge coefficient C
        s=ratio_headwater
        c = 0.886584 + 0.2150039246039883*s - 0.29569966842302525*s**2 + 0.08032610044946863*s**3
        return float(c)
    
    hydr_radius = float(0.25*pipe_diameter) #R0


    return float(pipe_area*discharge_coefficient()*np.sqrt((2*gravitation*(water_level_entrance-water_level_outlet))/
                                           (1+((29*discharge_coefficient()**2*manning_coeff**2*pipe_length)/(hydr_radius**(4/3))))))
    
def ratio_tailwater_pipe(water_level_outlet):
    return water_level_outlet/pipe_diameter


###Type 3 flow iteration 

def type_3_flow_final(water_level_outlet,water_level_entrance,pipe_length,elevation):
    
    discharge_guess = initial_guess(water_level_outlet,gravitation,water_level_entrance)   #discharge guessed

    b = round(type_3_flow(water_level_entrance,water_level_outlet, pipe_length, elevation, discharge_guess), 5)

    while (discharge_guess < b): #loop if guess is smaller than calculated
    
        discharge_guess = round(discharge_guess + 0.0001,5)#0.001,4
        b = round(type_3_flow(water_level_entrance,water_level_outlet, pipe_length, elevation, discharge_guess=discharge_guess), 5)

        if (discharge_guess == b):
            break
    
    while (discharge_guess > b): #loop if guess is higher than calculated
    
        discharge_guess = round(discharge_guess - 0.0001,5)#0.001,4
        b = round(type_3_flow(water_level_entrance,water_level_outlet, pipe_length, elevation, discharge_guess), 5)

        if (discharge_guess == b):
            break
    return b    

###type 5 flow as defined by Bodhaine (1968)

def type_5_flow (water_level_outlet,water_level_entrance,pipe_length,elevation):
        
    pipe_area = float(pipe_diameter*math.pi) #A0
    
    def ratio_headwater_pipe ():
        ratio = (water_level_entrance-elevation)/pipe_diameter
        return float(ratio)

    ratio_headwater=ratio_headwater_pipe()

    def discharge_coefficient():   #computes discharge coefficient C
        s=ratio_headwater
        c = 0.886584 + 0.2150039246039883*s - 0.29569966842302525*s**2 + 0.08032610044946863*s**3
        return float(c)
    
    return pipe_area*discharge_coefficient()*np.sqrt(2*gravitation*(water_level_entrance-elevation))


###decision which flow type to choose

def decision_flow_type(water_level_entrance,pipe_length,elevation):
    water_level_outlet_test=(ratio_inl_outl)*water_level_entrance
    if ratio_tailwater_pipe(water_level_outlet=water_level_outlet_test)<=1:
        outflow=type_3_flow_final(water_level_outlet=water_level_outlet_test,water_level_entrance=water_level_entrance,elevation=elevation,pipe_length=pipe_length)
        print('type3',outflow)
    else:
        outflow=type_4_flow(water_level_entrance=water_level_entrance,water_level_outlet=water_level_outlet_test,pipe_length=pipe_length,elevation=elevation)
        print('type4',outflow)
    return outflow


###discharge over all 8 culverts for the specific wetland at Olyelo Wii Dyel, Kotomor Subcounty, Uganda

def discharge_total_per_culvert(h_in):
    outlet_height           = np.float64([0.951, 0.863, 0.776, 0.771, 0.906, 1, 0.932, 0.849])
    elevation               = np.float64([0.24, 0.24, 0.24, 0.24, 0.286, 0.146, 0.308, 0.323])
    pipe_length             = np.float64([7.812501, 7.663736, 7.797803, 7.753839, 7.73535, 7.637441, 7.341153, 7.2699])
    discharge_total         = np.zeros(8)
    if h_in>1.814:
        h=2.91
    else:
        h=h_in
    
    for i in np.arange(8):
        outlet_height_loop      = outlet_height[i]
        elevation_loop          = elevation[i]
        inlet_height            = outlet_height_loop+elevation_loop
        water_level_loop        = h-elevation_loop-outlet_height_loop
#        print(water_level_loop)
        pipe_length_loop        = pipe_length[i]
        discharge_validation    = h/inlet_height
        if discharge_validation <= 1:
            discharge_total[i]  = 0
        else:
            discharge_total[i]  = decision_flow_type(water_level_entrance=water_level_loop,pipe_length=pipe_length_loop,elevation=0)
    return sum(discharge_total)



### --- area-height-relations and volume-height-relations fot the specific wetland at Olyelo Wii Dyel, Kotomor Subcounty, Uganda

####defintions if h below 1016.2 m / 

def area_low_water_prep (h):
    """
    function to transfer water depth -> area
    returns water surface area a
    R2=0.9994
    """    
    return 1340.6*h**3 - 874.31*h**2 + 631.99*h - 30.793

def volume_low_water (h):
	'''
	function for volume
	'''
	return 395.64*h**3 - 127.4*h**2 + 62.993*h - 2.3112


def vol_height_low_water (v):
    '''
    function to transfer vol -> water depth
    for water depth < 1015.897 m absolute / 1.011 m relative
    returns water depth h
    '''
    y=float(v)
    x = (7**(2/3) *10**(1/3)* (4239 *np.sqrt(6) *np.sqrt(2695368150000000* y**2 - 18715163398940000* y + 153557582801738701) + 539073630000 *y - 1871516339894)**(1/3) - (209060327 *7**(1/3)* 10**(2/3))/(4239* np.sqrt(6)* np.sqrt(2695368150000000* y**2 - 18715163398940000* y + 153557582801738701) + 539073630000* y - 1871516339894)**(1/3) + 63700)/593460
    return x

def area_low_water (volume):
    height_temp=vol_height_low_water (volume) 
    return area_low_water_prep (height_temp)


###definitions if h between 1.011 and 1.24 / 1015.897 and 1016.126

def vol_mid_water (h):
    return 2376.3*h**2 -3784.5*h +1740.5


def vol_height_mid_water (v):
    '''
    function to transfer vol -> water depth
    for water depth < 1016.2 m absolute / 1.314 m relative
    returns water depth h
    '''
    y=v
    x = (20247* np.sqrt(6)* np.sqrt(6149115135000* y**2 - 1457601393116300* y + 90996045001417111) + 122982302700* y - 14576013931163)**(1/3)/(40494* 5**(1/3)) - (76875113* 5**(1/3))/(40494* (20247* np.sqrt(6)* np.sqrt(6149115135000* y**2 - 1457601393116300* y + 90996045001417111) + 122982302700* y - 14576013931163)**(1/3)) + 22103/40494
    return x


#definitions if h between 1016.126 and 1016.7 / 1.24 and 1.814

def area_high_water (vol):
    '''function to transfer water depth -> area
    for water depth >= 1016.2 m absolute / 1.314 m relative
    '''
    return 5912.7 * h -5417.4

def volume_high_water (h):
    '''function for volume for high water
    R2=1
    '''
    return 3012*h**2 - 5625.4*h + 3068.8

def vol_height_high_water (v):
    '''
    function to transfer vol -> water depth
    for water depth < 1016.2 m absolute / 1.314 m relative
    returns water depth h
    '''
    y=v
    x = (3**(1/3)* (6059* np.sqrt(3)* np.sqrt(396483994800* y**2 + 70824741253920* y + 126798428070458867) + 6608066580* y + 590206177116)**(1/3) - (165570035* 3**(2/3))/(6059* np.sqrt(3)* np.sqrt(396483994800* y**2 + 70824741253920* y + 126798428070458867) + 6608066580* y + 590206177116)**(1/3) + 27621)/36354
    return x


####decision which definition to choose depending on water depth
    
def decision_high_low_water_area(volume):
    '''
    defines which formula to use and returns area
    '''
    if volume<342.67:
        area_calc=area_low_water(volume)
    else:
        area_calc=1096.8
    return area_calc

def decision_high_low_water_height(volume):
    '''
    defines which formula to use and returns height
    '''
    if volume<=342.67:
        height_calc = vol_height_low_water(volume)
#        print('low')
    elif 342.67< volume <= 701.86:
        height_calc = vol_height_mid_water(volume)
#        print('mid')
    elif volume>701.86:
        height_calc = vol_height_high_water(volume)
#        print('high')
    return height_calc
 