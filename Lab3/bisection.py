# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 23:09:38 2019
Lab3 Bisection
@author: Ryan Dalby
"""

import numpy as np
import matplotlib.pyplot as plt

def bisection(func, a, b, tol=1e-6, maxiter=10, plotOutput=False):
    """
    Given a function, bounds(a and b) for which a root exists between, a number of max iterations and a tolerence,
    will return the value of the root(either after max iterations or after tolerence has been reached)
    """
    if(func(a)*func(b) >= 0): #Means that either a or b is a root or there is 
        if(func(a) == 0): 
            return a
        elif(func(b) == 0):
            return b
        else:
            raise ValueError("a and b do not bracket a root")

    currentA = a #the current a postion
    currentB = b #the current b postion
    storeA = a #holds value of a before it possibly gets moved
    storeB = b #holds value of b before it possibly gets moved
    lastC = 0 #last iteration's root esitmate
    i = 0 # loop counter
    while(i < maxiter):
        c = (currentA + currentB) / 2
        #here we move either a or b to c
        if(func(currentA)*func(c) > 0): #if the function at a and c are the same sign then move a to c
            storeA = currentA
            currentA = c
        elif(func(currentA)*func(c) < 0): #if the function at a and c are different signs then move b to c
            storeB = currentB
            currentB = c 
        else: #else the function at c is 0 and c is the root 
            return c
        Ea = ((c - lastC) / c) * 100 #this is the current approximate relative error in percent  
        lastC = c #set lastC for next loop
        i += 1 #set counter value for next loop
        
        if(plotOutput):#plots output for each iteration
           plotIter(func, a, b, storeA, storeB, lastC, Ea, i)
        
        if(abs(Ea) < tol):#before enetering loop again will check if we have reached the tolerance, if so we return c
            return lastC
        
    raise RuntimeError("Root was not found within approximate relative error with the max iterations given")
    
def plotIter(f, xMin, xMax, curA, curB, c, ea, i):
    """
    Helper method for plotting output of bisection method
    """
    xVals = np.linspace(xMin,xMax)
    acb = np.array([curA,c,curB])
    plt.figure()
    plt.suptitle("Step {}\nZero Estimate = {}\nApproximate Relative Error = {}%".format(i, c, ea) )
    plt.axhline(color = "black", linestyle = "dashed")
    plt.plot(xVals, f(xVals))
    plt.scatter(acb, f(acb), color = "r" ,marker="o", s=200, linewidths = 3, facecolors = "none")
    plt.show()
    


