# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 14:59:00 2019
Lab09
@author: Ryan Dalby
"""

import numpy as np
from ball_motion import ball_motion
from euler import euler
from rk4 import rk4
import matplotlib.pyplot as plt

g = 9.81 #m/s^2
rho = 1.0 #kg/m^3
ballMass = 0.01 #kg
ballRadius = 0.03 #m
Cd = 0.4 #Coefficent of Drag

#Initial Conditions
z0 = -2.0 #m
v0 = 0.0 #m/s

#What t values we want to analyze
#tVals = np.linspace(0,10,101)
tVals1 = np.linspace(0,10,6) #Step size of 2s
tVals2 = np.linspace(0,10,11) #Step size of 1s
tVals3 = np.linspace(0,10,51) #Step size of 0.2s
tVals4 = np.linspace(0,10,101) #Step size of 0.1s


func = lambda t, y: ball_motion(t, y, [g,ballMass,rho,ballRadius, Cd])

#RK4_yv_output = rk4(func, tVals, [z0, v0])[1] #Index [y,v] from [t, [y,v]]
#yVals = RK4_yv_output[:,0]
#vVals = RK4_yv_output[:,1]

#Step size of 2s
RK4_yv_output1 = rk4(func, tVals1, [z0, v0])[1] #Index [y,v] from [t, [y,v]]
yVals1 = RK4_yv_output1[:,0]
#Step size of 1s
RK4_yv_output2 = rk4(func, tVals2, [z0, v0])[1] #Index [y,v] from [t, [y,v]]
yVals2 = RK4_yv_output2[:,0]
#Step size of 0.2s
RK4_yv_output3 = rk4(func, tVals3, [z0, v0])[1] #Index [y,v] from [t, [y,v]]
yVals3 = RK4_yv_output3[:,0]
#Step size of 0.1s
RK4_yv_output4 = rk4(func, tVals4, [z0, v0])[1] #Index [y,v] from [t, [y,v]]
yVals4 = RK4_yv_output4[:,0]

#euler_yv_output = euler(func, tVals, [z0, v0])[1] #Index [y,v] from [t, [y,v]]
#yVals = euler_yv_output[:,0]
#vVals = euler_yv_output[:,1]

#plot vals
#fig, (ax1, ax2) = plt.subplots(ncols=2, figsize = (10,5))
#ax1.plot(tVals, yVals)
#ax1.set_xlabel("Time")
#ax1.set_ylabel("Y position")
#ax1.set_title("Ball y postion vs time")
#ax2.plot(tVals, vVals)
#ax2.set_xlabel("Time")
#ax2.set_ylabel("Velocity")
#ax2.set_title("Ball y velocity vs time")

fig, (ax1) = plt.subplots(ncols=1, figsize = (10,10))
ax1.plot(tVals1, yVals1)
ax1.plot(tVals2, yVals2)
ax1.plot(tVals3, yVals3)
ax1.plot(tVals4, yVals4)
ax1.legend(['2s step size', '1s step size', '0.2s step size', '0.1s step size'], prop={'size': 15})
ax1.set_xlabel("Time")
ax1.set_ylabel("Y position")
ax1.set_title("Ball y postion vs time")

