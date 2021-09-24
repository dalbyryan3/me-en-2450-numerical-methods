# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 17:45:15 2019
HW1b Exercise 4 
@author: Ryan Dalby
"""
import math

def form1(x, numTerm):
    '''Enter x value to evaluate at and number of terms to approximate to
    using formula 1, gives a matrix that for each term gives:
    [current term number, resulting value of approximation that iteration, true relative error, approximate relative error]'''
    actualValue = (math.e)**(-1 * x) #actual value that formula approximates
    currentResult = 0 #current result of formula corresponds to current result of sum in formula
    lastResult = 0 #previous result(previous iteration) from formula
    matrix = [] #matrix will hold information about each approximation(with a given amount of sum terms)
    for n in range(numTerm):
        currentResult += ((-1)**n)* (x**n)/(math.factorial(n)) #formula
        Et = getEt(actualValue, currentResult)
        Ea = ((currentResult - lastResult)/currentResult) * 100
        lastResult = currentResult
        matrix.append([(n+1), currentResult, Et, Ea]) #add information to matrix about sum term we are on
        
    return matrix


def form2(x, numTerm):
    '''Enter x value to evaluate at and number of terms to approximate to
    using formula 2, gives a matrix that for each term gives:
    [current term number, resulting value of approximation that iteration, true relative error, approximate relative error'''
    actualValue = (math.e)**(-1 * x) #actual value that formula approximates
    sumResult = 0 #current result of sum in formula
    lastResult = 0 #previous result(previous iteration) from formula
    matrix = [] #matrix will hold information about each approximation(with a given amount of sum terms)
    for n in range(numTerm):
        sumResult += (x**n)/(math.factorial(n)) #sum formula
        currentResult = 1 / sumResult #current reuslt of formula is 1/sumResult
        Et = getEt(actualValue, currentResult)
        Ea = ((currentResult - lastResult)/currentResult) * 100
        lastResult = currentResult
        matrix.append([(n+1), currentResult, Et, Ea]) #add information to matrix about sum term we are on
        
    return matrix

def getEt(T, A):
    '''Given T true value and A approximate value gives true relative error'''
    return ((T - A) / T) * 100 

def printByRow(m):
    '''Prints by row for a matrix'''
    for i in m:
        print(i)

m1 = form1(5, 20)
m2 = form2(5, 20)
printByRow(m1)
printByRow(m2)