# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 17:45:50 2019
AID01
@author: Ryan Dalby
"""
import numpy as np

diameters = np.array([2.0, 2.0, 2.0, 8.0, 8.0, 8.0, 8.0])#inches
radii = diameters/2 #inches
lengths = np.array([10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 24.0]) # inches
startingPressures = np.array([10.0, 100.0, 150.0, 10.0, 100.0, 150.0, 150.0])# psi
P0 = 88113.833 #Pa (26.02 inHg)

def energyEq(r, h, P0, P1):
    """Takes in an ndarray of radii(of tank), heights/lengths(of tank), surrounding atmospheric pressure
    and pressure inside the tank and gives the max energy we could get by bringing the gas inside the tank 
    to the same state as the environment"""
    V = np.pi * r**2 * h
    ans = (P1 * V * (np.log(P1/P0) + P0/P1 - 1))
    return ans

def inToMeters(inch):
    """Converts inches to meters"""
    return inch * (.0254)

def psiToPa(psi):
    """Converts psi to pascals"""
    return psi * (6894.76)

def averageForce(energy, distanceForceApplied):
    """Converts from total energy(J) and the distance a force was applied(m) to average force(N)"""
    return energy/distanceForceApplied



    
energyVals = energyEq(inToMeters(radii), inToMeters(lengths), P0, psiToPa(startingPressures))
forceVals = averageForce(energyVals, 10.0) #force is over 10m

table = np.array((diameters,lengths,startingPressures, energyVals, forceVals)).transpose()

np.set_printoptions(suppress = True)
print(table)
