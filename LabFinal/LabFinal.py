# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 14:50:52 2019
ME EN 2450 Lab Final
@author: Ryan Dalby
"""
import numpy as np
import matplotlib.pyplot as plt

print("Fundamental Programming Concepts")

#1
b = np.zeros((3,1), dtype = 'float')
for i in range(3):
    b[i] = 20 + i*20

#2
A = np.zeros((3,3), dtype = 'float')
counter = 1
for i in range(3):
    for j in range(3):
        A[i][j] = counter
        counter +=1
     
#3
#This matrix multiplication just works for matricies/vectors like A and b
product = np.empty((3,1))
for i in range(A.shape[0]):
    rowSum = 0
    for j in range(A.shape[1]):
        rowSum += A[i][j] * b[j]
    product[i] = rowSum
    
#4
#Not using an inner for loop
product2 = np.empty((3,1))
for i in range(A.shape[0]):
    rowSum = 0
    rowSum = np.dot(A[i][:], b[:])
    product2[i] = rowSum
    
print("1. b = \n{}\n\n2. A = \n{}\n\n3. Product is: \n{}\n \n4. Product2 is: \n{}\n".format(b,A,product,product2))
     
   

print()
print("Root Finding Methods in Engineering")
epsilon = 1.43e-6 #m
rho = 1.38 #kg/m^3
mu = 1.91e-5 #Ns/m^2
D = 0.0061 #m
V = 54.0 #m/s

Re = (rho * V * D)/ mu #Calculate Reynolds Number
gFunc = lambda f: (1.0/np.sqrt(f)) + 2.0 * np.log10((epsilon/(3.7*D))+(2.51/(Re * np.sqrt(f))))
dgdfFunc = lambda f: (-0.5) * f**(-1.5) * (1.0 + (2.18261/Re)* (((epsilon/D)/3.7 + 2.51 / (Re * np.sqrt(f)))**-1.0))

#1
fVals = np.linspace(0.01, 0.05)
plt.plot(fVals, gFunc(fVals), label = 'g(f)')
plt.xlabel("f")
plt.ylabel("g")
plt.legend()
plt.show()
#We see root is appox f = 0.025

#2
def newtonRaphson(func, derivFunc, initialGuess, errorTol):
    """
    Given a function and it's derivative wrt 1 variable, an initial root guess,
    and a relative approx. error tolerence (in percent)
    will solve for and return the root to the func along with iterations taken to get there
    """
    eA = 100.0 #Arbitrary eA to begin for loop
    x = initialGuess
    lastxVal = x
    iterCount = 0
    while eA > errorTol:
        x = x - func(x)/derivFunc(x)
        eA = np.abs((x - lastxVal)/ x) * 100
        lastxVal = x
        iterCount+=1
    return x, iterCount

#3
root, requiredIter = newtonRaphson(gFunc, dgdfFunc, 0.025, 0.1)
print("1. From the plot we guess the root of the function is approximately f = 0.025.\n\n2/3. Using Newton Raphson with a 0.1% approximate relative error tolerance we get the root to be f = {:.5f} which took {} iterations\n".format(root, requiredIter))



print()
print("Solving ODEs")

k = 0.048
y0 = 2.8 #m

def dydtBase(y):
    if(y < 0):
        return 0 #Physically means tank is empty
    else:
        return -k * np.sqrt(y)

dydtFunc = lambda y : dydtBase(y)

#1
def euler(derivFunc, initialValue, stepSize, numSteps):
    """
    Given a function of a derivative in the form of dy(y)/dt (indicates dy/dt is a function of y), an initial value say y(0) = initialValue, a step size, and number of steps 
    Will return (tVals, yVals) that correspond with eachother
    """
    dt = stepSize
    y = initialValue
    t = 0
    yVals = [initialValue]
    tVals = [0]
    for _ in range(numSteps):
        y = y + derivFunc(y)*dt
        t = t + dt
        yVals.append(y)
        tVals.append(t)
    
    return (np.array(tVals), np.array(yVals))
      
#2  
tVals1, yVals1 = euler(dydtFunc, y0, 5, 20)
tVals2, yVals2 = euler(dydtFunc, y0, 0.5, 200)
tVals3, yVals3 = euler(dydtFunc, y0, 0.1, 1000)

print("1/2. ")
plt.plot(tVals1, yVals1, label = 'dt = 5 min')
plt.plot(tVals2, yVals2, label = 'dt = 0.5 min')
plt.plot(tVals3, yVals3, label = 'dt = 0.1 min')
plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.show()

#3
#Use t and y Vals3 since it is the most accurate version we calculated
for i in range(yVals3.shape[0]):
    if(yVals3[i] < 1e-3): #We have the index of the effectively empty tank
        tEmpty = tVals3[i]
        print("3. The tank is effectively empty at t = {:.2f} minutes".format(tEmpty))
        break
    
