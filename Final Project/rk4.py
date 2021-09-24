# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:05:06 2019
Lab09
@author: Ryan Dalby
"""
import numpy as np

def rk4(odefun, tspan, y0):
    """Uses RK4 Method to calculate the solution to an ODE.

    Parameters
    ----------
    odefun : callable
        A callable function to the derivative function defining the system.
    tspan : array_like
        An array of times (or other independent variable) at which to evaluate
        RK4 Method.  The nTimes values in this array determine the size of
        the outputs.
    y0 : array_like
        Array containing the initial conditions to be used in evaluating the
        odefun.  Must be the same size as that expected for the
        second input of odefun.

    Returns
    -------
    t : ndarray
        t[i] is the ith time (or other independent variable) at which
        RK4 was evaluated.
    y : ndarray
        y[i] is the ith dependent variable at which RK4 Method was
        evaluated.

    Notes
    -----
    Uses RK4 Method to calculate the solution to the ODE defined by the
    system of equations odefun, over the time steps defined by tspan, with
    the initial conditions y0. The time output is stored in the array t,
    while the dependent variables are stored in the array y. RK4
    Method uses the equation:

                       
       Y(i+1) = Y(i) + (slope) * delta_t 
                       

    where delta_t is the time step, Y(i) is the values of the dependent
    variables at the current time, Y(i+1) is the values of the dependent
    variables at the next time, and slope is (1.0/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    where k1, k2, k3, and k4 are calculated using the typical Runge Kutta 4 method 
    of calculating 4 slope estimates, each in slightly different ways.
    """
    
    tspan = np.asarray(tspan)
    y0Val = np.asarray(y0) 
    # Determine the number of items in outputs
    num_times = tspan.shape[0] #length of t and y
    num_yVals = y0Val.shape[0] #width of y 
    # Initialize outputs
    t = np.zeros(num_times)
    y = np.empty((num_times,num_yVals)) #Y will be num_times in length(corresponding to t) but each slot can have an array value of y at a corresponding t

    # Assign first row of outputs
    t[0] = tspan[0]
    y[0] = y0Val 

    # Assign other rows of output
    for idx in range(num_times-1):
        
        # Calculate time step
        dt = tspan[idx+1] - tspan[idx]

        # Calculate the slope at current time
        k1 = np.asarray(odefun(t[idx], y[idx,:]))
        k2 = np.asarray(odefun((t[idx] + .5*dt), (y[idx] + .5*k1*dt)))
        k3 = np.asarray(odefun((t[idx] + .5*dt), (y[idx] + .5*k2*dt)))
        k4 = np.asarray(odefun((t[idx] + dt), (y[idx] + k3*dt)))
        
        dydt = (1.0/6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Calculate the next state
        t[idx+1] = tspan[idx+1]
        y[idx+1] = y[idx,:] + dydt * dt

    return t, y

