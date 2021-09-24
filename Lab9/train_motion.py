# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 14:03:58 2019
Lab09
@author: Ryan Dalby
"""

import numpy as np

def train_motion(t,y,params):
    """
    Given current time t, a vector y which contains [y, v],
    and params containing [gravity, air density, train mass, frontal area, drag coefficent, rolling coefficent of friction, propulsion force]
    Returns a vector [dy/dt, dv/dt] ([velocity, accleration])
    """
    
    #Assign params to variables
    g, rho, m, A, Cd, Crr, Fp = params[:7]
    
    #extract velocity from y
    velocity = y[1]
    
    #Calculate Fd and Frr
    Fd = (rho * Cd * A * velocity**2)/2
    Frr = m * g * Crr
    
    #Determine derivative values of dydt and dvdt
    dydt = velocity
    dvdt = (Fp - Fd - Frr) / m
    
    return dydt, dvdt
    