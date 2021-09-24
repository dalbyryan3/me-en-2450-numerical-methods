# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 18:34:16 2019
HW2 Newton-Raphson
@author: Ryan Dalby
"""

import scipy.optimize as sciOp

def newtonRaphson(f, dfdx, initialGuess, tolerence):
    """
    Given a function and its derivative, an initial guess for a root value, and a termination tolerence in approximate relative error(in percent)
    will attempt to find a root of the original function and return it along with the iterations it took to terminate (will return (root,i)).
    """
    UPPER_LIMIT_ON_ITERATIONS = 100000
    x = initialGuess #will hold current x value
    lastX = initialGuess #will hold last x value
    Ea = 100 #approximate relative error, in percent, start arbitrarily at 100 to get while loop to run the first time
    i = 0 #iteration counter, will incrment at bottom of loop and when it exits will be how many loops completed
    while(abs(Ea) > tolerence):
        if(dfdx(x) == 0): #method cannot be computed if dervivative at point we are looking at is 0 
            raise Exception("The derivative at a point was 0")
        x = x - (f(x) / dfdx(x)) #netwton-raphson
        Ea = ((x - lastX) / x) * 100 #calculate current approximate relative error
        lastX = x
        i += 1
        if(i == UPPER_LIMIT_ON_ITERATIONS): #sets limit on number of iterations to find root
            raise Exception("The root could not be found under the limit of iterations")
            
    return (x,i)

E = 29 * 10**4 #psi
w0 = 250 #3000 lbs/ft in lbs/in
I = 723 #in^4
L = 180 #15 ft in inches

y = lambda x: (w0/(120*E*I*L)) * (-x**5 + 2*L**2*x**3 - L**4*x)
f = lambda x: (w0/(120*E*I*L)) * (-5*x**4 + 6*L**2*x**2 - L**4) #dy/dx
dfdx = lambda x: (w0/(120*E*I*L)) * (-20*x**3 + 12*L**2*x)

tol = .001 #tolerence for newton raphson in percent
intialGuess = 90 #inches
newtonRaphsonInfo = newtonRaphson(f, dfdx, intialGuess, tol) #0 index is root, 1 is iterations to find root
scipyRoot = sciOp.fsolve(f, 90)
print("Newton Raphson, root = {}inches iterations = {} tolerence = {}% initial guess = {}inches".format(newtonRaphsonInfo[0], newtonRaphsonInfo[1], tol, intialGuess))
print("Scipy fsolve, root = {}inches".format(scipyRoot[0]))
maxDisplacement = y(newtonRaphsonInfo[0])
print("Max displacement = {}inches".format(maxDisplacement))

