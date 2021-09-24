# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:31:37 2019
Final Project ME EN 2450
Driver Script
@author: Ryan Dalby
"""
import numpy as np
import pandas as pd
from rk4 import rk4
from findmin import findmin
from plot_results import plot_results
import time as time

#We are assuming our design is a tank with a piston on the back

def train_motion(t,y,designParams,constParams):
    """
    Given current time t, a vector y which contains [x, v]
    and designParams which contains [Lt, rO, rhoT, P0gage, rg, Lr, rp],
    and constParams containing [rhoA, Patm, Cd, Cr, muS, rw, mw, g]
    Returns a vector [dx/dt, dv/dt] ([velocity, accleration])
    """
    P0Eq = lambda P0gage: P0gage + Patm #Pa initial tank absolute pressure
    mTotalEq = lambda Lt, rO, rhoT, Lr, rp: mw + (rhoT*np.pi*Lt*(rO**2 - riEq(rO)**2)) + MpEq(rp, Lr)  #kg total mass of train = mass of wheels + mass of tank + mass of piston
    AtEq = lambda rO: np.pi * rO**2 #m^2 train frontal area
    V0Eq = lambda rO, Lt: np.pi * riEq(rO)**2 * Lt #m^3 tank volume
    LpEq = lambda Lr: 1.5 * Lr #m total length of pistion
    MpEq = lambda rp, Lr: 1250.0 * (np.pi * rp**2 *LpEq(Lr)) #kg mass of piston
    ApEq = lambda rp: np.pi * rp**2 #m^2 area of the piston
    riEq = lambda rO: rO/1.15 #m inside radius of tank
    
    #Assign params to variables
    Lt, rO, rhoT, P0gage, rg, Lr, rp = designParams[:7]
    rhoA, Patm, Cd, Cr, muS, rw, mw, g = constParams[:8]
    
    #extract position and velocity from y
    position = y[0] 
    velocity = y[1]
    
    Ap = ApEq(rp)
    P0 = P0Eq(P0gage)
    V0 = V0Eq(rO, Lt)
    At = AtEq(rO)
    mTotal = mTotalEq(Lt, rO, rhoT, Lr, rp)

    Fd = 0.5 * Cd * rhoA * At * velocity**2 #part of drag term
    Fr = Cr * mTotal * g #part of rolling resistance term
    
    #Determine derivative values of dydt and dvdt
    if(position <= (Lr * (rw/rg))): #accelerating
        Ft = Ap * (rg/rw) * (((P0*V0) / (V0 + Ap*(rg/rw)*position)) - Patm)  #part of thrust term 
        dydt = velocity
        dvdt = (1.0 / (mTotal + mw)) * (Ft - Fd - Fr)
    else: #decelerating
        dydt = velocity
        dvdt = (1.0 / mTotal) * (-Fd - Fr)
    return dydt, dvdt

def driver(stepSize, maxIters, y0, designParams, constParams):
    """
    Given stepSize, maxIters, y0 (x0, v0), designParams, constParams
    will give tVals, xVals, yVals as described by train_motion until
    maxIters at a step size is achieved or train is at rest (velocity = 0)
    """
    # Uses RK4 to integrate Newton's law
    movingTrainEq = lambda t, y: train_motion(t, y, designParams, constParams)
    t0 = 0.0
    t, y = [t0], [y0]
    tspan = np.array([t0, t0+stepSize])
    iters = 1
    while iters <= maxIters:
        tp, yp = rk4(movingTrainEq, tspan, y[-1])
        if yp[-1,1] < np.finfo(float).eps: # Velocity is essentially less than 0
            break
        
        t.append(tp[-1]) #append current time value
        y.append(yp[-1]) #append current y(x,v) value
        
        tspan[:] = [tspan[1], tspan[1] + stepSize] #next t span
        iters = iters + 1
    y = np.asarray(y)
    return np.asarray(t), y[:,0], y[:,1]


def objfun(x, objParams, stepSize, maxIters, y0, constParams):
    """
    Given a vector x of design parameters (to optimize), objParams([xMin, xMax]), stepSize, maxIters, y0 ([x0, v0]), constParams
    Will return the time to go an amount between objParams.  Will restrict possible design parameter space by velocity being equal to 0 
    between objParams and time being less that the max time as indicated by stepSize and maxIters.  Will also check for the following design constraints:
    height of train, width of train, train length, pinion gear radius, wheel slippage, if these contraints are violated will send objfun to a maximum t value
    """
    Lt, rO, rhoT, P0gage, rg, Lr, rp = x[:]
    xMin = objParams[0]
    xMax = objParams[1]
    tMax = stepSize * maxIters
    
    tVals, xVals, vVals = driver(stepSize, maxIters, y0, [Lt, rO, rhoT, P0gage, rg, Lr, rp], constParams)
    
    if (xVals.max() > xMax or xVals.max() < xMin ):
    #Means we are off the track or have not made it to destination,
    #discard this information for optimization (send return to tmax)
#        print("0", xVals.max())
        return tMax
 
    #make sure train is capable of passing through tunnel(final check after constraining design variables with ranges)
    #height cannot exceed 0.23m
    HtEq = lambda rO: 2*rO + 2*rw #m height of train
    if (HtEq(rO) >= 0.23):
#        print("1", HtEq(rO))
        return tMax
    
    #width cannot exceed 0.2m
    WtEq = lambda rO: 2*rO #m width of the train
    if(WtEq(rO) >= 0.2):
#        print("2", WtEq(rO))
        return tMax
    
    #make sure train length is less than 1.5m so it will fit on starting line
    LpEq = lambda Lr: 1.5 * Lr #m total length of pistion
    LTotalEq = lambda Lt, Lr: Lt + LpEq(Lr) #m total length of the train
    if(LTotalEq(Lt,Lr) >= 1.5):
#        print("3", LTotalEq(Lt,Lr))
        return tMax
    
    #make sure the radius of the pinion gear is less than the radius of the train wheel
    if(rg > rw):
#        print("4")
        return tMax 
 
    #make sure wheels do not slip
    riEq = lambda rO: rO/1.15 #m inside radius of tank
    MpEq = lambda rp, Lr: 1250.0 * (np.pi * rp**2 *LpEq(Lr)) #kg mass of piston
    mTotalEq = lambda Lt, rO, rhoT, Lr, rp: mw + (rhoT*np.pi*Lt*(rO**2 - riEq(rO)**2)) + MpEq(rp, Lr)  #kg total mass of train = mass of wheels + mass of tank + mass of piston
    ApEq = lambda rp: np.pi * rp**2 #m^2 area of the piston

    Ft = ApEq(rp) * (rg/rw) * (P0gage) #traction force initially, we will see if it slips  
    if(Ft > (muS * mTotalEq(Lt, rO, rhoT, Lr, rp) * 0.5 *g)):
#        print("5", Ft, (muS * mTotalEq(Lt, rO, rhoT, Lr, rp) * 0.5 *g))
        return tMax
#    print('SUCCESS', x)
    return np.max(tVals) #Return biggest t value, which is our stopping time that meets all the constrains


def testTrain(f, ranges, method = 'both', materialName = 'material'):
    """
    Given an objective function and ranges for train optimization along with the optimization to use(default is both brute and monte carlos) will display optimal results using
    brute force and monte carlos optimization, and if optimal material name is already known will include in results
    Will also return optimal results
    """
    
    riEq = lambda rO: rO/1.15 #m inside radius of tank
    HtEq = lambda rO: 2*rO + 2*rw #m height of train
    MpEq = lambda rp, Lr: 1250.0 * (np.pi * rp**2 *LpEq(Lr)) #kg mass of piston
    mTotalEq = lambda Lt, rO, rhoT, Lr, rp: mw + (rhoT*np.pi*Lt*(rO**2 - riEq(rO)**2)) + MpEq(rp, Lr)  #kg total mass of train = mass of wheels + mass of tank + mass of piston
    AtEq = lambda rO: np.pi * rO**2 #m^2 train frontal area
    V0Eq = lambda rO, Lt: np.pi * riEq(rO)**2 * Lt #m^3 tank volume
    LpEq = lambda Lr: 1.5 * Lr #m total length of pistion
    
    doBrute = True
    doMonteCarlos = True
    bruteResults = []
    mcResults = []
    if(method == 'brute'):
        doMonteCarlos = False
    if(method == 'monte carlos'):
        doBrute = False
        
    if(doBrute):
        t1 = time.time()
        res = findmin(f, ranges, method = 'brute')
        t2 = time.time()
#        print(res)
        #Keep in mind, adjust ranges if brute force returns the orginal left end values of the ranges
        #This is because for the ranges given the brute force never made a combination of design parameters
        #that met all of the constraints.  Thus all objective function were returns were sent to the same value, and when minimized default was returned
        count = 0
        for rang in ranges: #check that we actually found a solution that meet constraints
            if(rang[0] != res[count]):
                break
            if(count == 6):
                raise Exception("Given ranges did not find a solution that met constraints using brute since the left index of ranges was returned from optimizer")
            count += 1

        tValsOpt, xValsOpt, vValsOpt = driver(stepSize, maxIters, [0,0], res, constParams)
        tFinish = np.interp(objParams[0], xValsOpt, tValsOpt)
        xFinish = xValsOpt.max()
        plot_results(tValsOpt, xValsOpt, vValsOpt, objParams[0], objParams[1])
        Lt, rO, rhoT, P0gage, rg, Lr, rp = res[:]
        exTime = t2 - t1
        
        bruteResults = ['brute', tFinish, xFinish, Lt, rO*1000, HtEq(rO), '{} (density = {:.2f})'.format(materialName, rhoT), mTotalEq(Lt,rO,rhoT,Lr,rp), AtEq(rO), P0gage, V0Eq(rO,Lt), rg*1000, Lr, LpEq(Lr), rp*1000, MpEq(rp, Lr),exTime]

    
    if(doMonteCarlos):
        t1 = time.time()
        res = findmin(f, ranges, method = 'monte carlos')
        t2 = time.time()
#        print(res)
        tValsOpt, xValsOpt, vValsOpt = driver(stepSize, maxIters, [0,0], res, constParams)
        tFinish = np.interp(objParams[0], xValsOpt, tValsOpt)
        xFinish = xValsOpt.max()
        plot_results(tValsOpt, xValsOpt, vValsOpt, objParams[0], objParams[1])
        Lt, rO, rhoT, P0gage, rg, Lr, rp = res[:]
        exTime = t2 - t1
        mcResults = ['monte carlos', tFinish, xFinish, Lt, rO*1000, HtEq(rO), '{} (density = {:.2f}kg/m^3)'.format(materialName, rhoT), mTotalEq(Lt,rO,rhoT,Lr,rp), AtEq(rO), P0gage, V0Eq(rO,Lt), rg*1000, Lr, LpEq(Lr), rp*1000, MpEq(rp, Lr), exTime]

    df = pd.DataFrame([bruteResults, mcResults], columns = ['method name', 'time to finish(s)', 'x position at finish(m)', 'length of train(m)', 'outer radius of train(mm)', 'height of train(m)', 'material of train', 'total mass of train(kg)', 'train frontal area(m^2)', 'initial tank gauge pressure(Pa)', 'tank volume(m^3)', 'pinion gear radius(mm)', 'length of piston stroke(m)', 'total length of piston(m)', 'radius of piston(mm)', 'mass of piston(kg)', 'execution time(s)']).transpose()
    print(df.to_string())
    print("Max iterations used in driver/RK4: {}\nStep size used in driver/RK4: {}".format(maxIters,stepSize))
    return df   




#Fixed Physical Parameters 
rhoA = 1.0 #kg/m^3 air density 
Patm = 101325.0 #Pa atmospheric pressure 
Cd = 0.80 #coefficent of drag
Cr = 0.03 #rolling friction coefficent 
muS = 0.7 #coefficient of static friction
rw = 0.020 #m = 20mm wheel radius 
mw = 0.1 #kg = 100g mass of wheels and axles 
g = 9.81 #m/s^2

#Design Physical Parameter Ranges (Chosen with some broad realistic values in mind, by looking up practical values for parts online) 

#length of train, relates to total length of the train which is constrained to be < 1.5m 
LtRange = [(0.03, 1.25), (.277699, .277699)]#m 
#outer radius of train- constrained to fit in tunnel, width must be < 0.2m , height which is 2rO + 2rw must be < than height of 0.23m
rORange = [(.003, .095), (0.0841375, 0.0841375)]#m 

#density of train material- constrained to reasonable piping/tubing material densities,--> (acrylic density, approx. copper density)
rhoTRange = [(1200, 9000), (1400, 1400)] #kg/m^3 

#initial tank gauge pressure- constrained to < 30psig which is equal to 206843 Pa 
P0gageRange = [(20684.3, 206843.0), (87668.5, 87668.5)] #Pa 
        
#radius of pinion gear radius- constrained to be < radius of wheel
rgRange = [(.001, .020), (0.00635, 0.00635)] #m 

#length of piston stroke, relates to length of piston and subsequently to the total length of the train which is constrained to be < 1.5m
LrRange = [(.01, .70), (0.3048,0.3048)] #m 

#radius of piston-  constrained to fit in tunnel, width must be < 0.2m , height which is 2rO + 2rw must be < than height of 0.23m
rpRange = [(.003, .095), (0.009525, 0.009525)] #m 


ranges = [LtRange[0], rORange[0], rhoTRange[0], P0gageRange[0], rgRange[0], LrRange[0], rpRange[0]]
rangesRealistic = [LtRange[1], rORange[1], rhoTRange[1], P0gageRange[1], rgRange[1], LrRange[1], rpRange[1]]


x0 = 0.0 #m
v0 = 0.0 #m
y0 = np.array([x0, v0])
stepSize = 0.1
maxIters = 1000
constParams = np.array([rhoA, Patm, Cd, Cr, muS, rw, mw, g])
objParams = np.array([10.0,12.5])

f = lambda x: objfun(x, objParams, stepSize, maxIters, y0, constParams)
results = testTrain(f, ranges, method='monte carlos')
resultsActual = testTrain(f, rangesRealistic, method='monte carlos')