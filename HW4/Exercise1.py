# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 16:13:40 2019
HW4 Exercise 1 
@author: Ryan Dalby
"""
import numpy as np

def LUDecomposition(A):
    """
    Given an A matrix will return an L and U matrix 
    """
    n = np.shape(A)[0] #assuming nxn matrix 
    U = np.array(A) #make copy of A to store upper triangular form which is our U matrix
    L = np.eye(n)#makes empty diagonal array with ones in the diagonal to store factors from gauss elimination which is our L matrix
    
    #Get L and U matricies using naive gauss elimination steps of forward elimination and extracting steps
    for k in range(n - 1):
        for i in range(k + 1, n):
            s = U[i,k] / U[k,k]
            L[i,k] = s
            for j in range(k, n):
                U[i,j] = U[i,j] - s * U[k,j]
    return [L,U]

def LUSolve(L, U, b):
    """
    Given an L and U decomposition matricies of a matrix A and a corresponding b, will solve for x in Ax = b
    """
    
    n = np.shape(L)[0] #assuming nxn matrix 
    #Perform forward substitution (solve Ld = b for d)
    d = np.zeros(shape = (n,1), dtype='float') #create b vector of correct size
    d[0] = b[0]/L[0,0] #index first element out of b vector and set it to the solutin value for bn(we know L is lower triangular matrix)
    for i in range(0, n):
        s = 0.0
        for j in range(0,i):
            s = s + L[i,j] * d[j]
        d[i] = (b[i] - s) / L[i,i]
    
    
    #Perform back substitution (solve Ux = d for x)
    xSol = np.zeros(shape = (n,1), dtype='float') #create X vector of correct size
    xSol[n-1] = b[n-1] / U[n-1,n-1] #index last element of xSol and set it to the solution value for xn(we know U is an upper triangular matrix)
    for i in range(n-1, -1, -1):
        s = 0.0
        for j in range(i+1, n):
            s = s + U[i,j] * xSol[j]
        xSol[i] = (d[i] - s) / U[i,i]
    return xSol

def LUDecomposeAndSolve(A, b):
    """
    Given A and b will solve Ax = b for x by decomposing A into L and U and solving for x using L and U
    """
    LU = LUDecomposition(A) #get L an U matricies that correspond with A
    xAns = LUSolve(LU[0], LU[1], b) 
    return xAns


#Do exercise 1b
A = np.array([[8,4,-1],[-2,5,1],[2,-1,6]], dtype='float')
b = np.array([[11],[4],[7]], dtype='float')

print(LUDecomposeAndSolve(A, b))


