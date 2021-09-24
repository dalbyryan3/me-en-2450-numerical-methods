# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:43:25 2019
 Lab09
@author: Ryan Dalby
"""

import numpy as np
from train_motion import train_motion
from euler import euler
from rk4 import rk4
import matplotlib.pyplot as plt

g = 9.81 #m/s^2
rho = 1.0 #kg/m^3
trainMass = 10.0 #kg
A = 0.05 #m^2
Fp = 1.0 #N Constant Propulsive Force
Cd = 0.4 #Coefficient of Drag
Crr = 0.002 #Coefficient of Rolling Resistance

#Initial Conditions
x0 = 0 #m
v0 = 0 #m/s

#What t values we want to analyze
#tVals = np.linspace(0,10,101)
tVals1 = np.linspace(0,10,11) #Step size of 1s

func = lambda t, x: train_motion(t, x, [g,rho,trainMass, A, Cd, Crr, Fp])

#RK4_xv_output = rk4(func, tVals, [x0, v0])[1] #Index [y,v] from [t, [y,v]]
#xVals = RK4_xv_output[:,0]
#vVals = RK4_xv_output[:,1]

#Step size of 1s
RK4_xv_output1 = rk4(func, tVals1, [x0, v0])[1] #Index [y,v] from [t, [y,v]]
xVals1 = RK4_xv_output1[:,0]
vVals1 = RK4_xv_output1[:,1]

#euler_xv_output = euler(func, tVals, [x0, v0])[1] #Index [y,v] from [t, [y,v]]
#xVals = euler_xv_output[:,0]
#vVals = euler_xv_output[:,1]

#plot vals
#fig, (ax1, ax2) = plt.subplots(ncols=2, figsize = (10,5))
#ax1.plot(tVals, xVals)
#ax1.set_xlabel("Time")
#ax1.set_ylabel("X position")
#ax1.set_title("Train x postion vs time")
#ax2.plot(tVals, vVals)
#ax2.set_xlabel("Time")
#ax2.set_ylabel("Velocity")
#ax2.set_title("Train x velocity vs time")

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize = (10,5))
ax1.plot(tVals1, xVals1)
ax1.set_xlabel("Time")
ax1.set_ylabel("X position")
ax1.set_title("Train x postion vs time using RK4\n with time steps of 1s")
ax2.plot(tVals1, vVals1)
ax2.set_xlabel("Time")
ax2.set_ylabel("Velocity")
ax2.set_title("Train x velocity vs time using RK4\n with time steps of 1s")
