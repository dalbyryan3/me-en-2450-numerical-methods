import numpy as np

# IMPORTANT!!! IF YOU CHOOSE TO WRITE BISECTION AND COLEBROOK EQUATION IN ONE
# FILE, YOU WILL NEED TO CHANGE THE FOLLOWING IMPORT STATEMENTS:
from colebrook_equation import colebrook_equation
from bisection import bisection



"""Calculates the major losses due to frictional effects in a pipe.  This
script can handle either laminar or turbulent flow.

Parameters
----------
e : float
D : float
    Pipe diameter
rho : float
    The density
mu : float
L : float
    Pipe length
V : float
    Fluid velocity
a, b : float
    Initial upper and lower bounds of the friction factor
tol : float {1e-6}
    Bisection method error tolerance
maxiter : int {10}
    Maximum number of bisects
plot_output : bool {False}
    Plot incremental results during bisection procedure
verbose : bool {False}
    Verbose output

Returns
-------
head_loss : float
    The calculated losses due to frictional effects in the pipe

Notes
-----
The bisection function needs to have been created in order for this
script to work.  It is expected that the function should use the Bisection
algorithm, although other bracketing methods will also work.

The colebrook_equation function also needs to be present in order for this
script to work.  This function should return the difference between the left
and right sides of the Colebrook Equation, with the inputs in the order:
(friction factor, roughness, pipe diameter, Reynolds Number)

The pipe properties are declared in the first section, while the numerical
method properties are declared in the second section.  The Reynolds number
is used to determine whether the flow is laminar or turbulent, which is used
to determine whether or not to use the colebrook_equation function to
calculate the friction factor.  The head loss due to frictional effects is
then calculated and reported.
"""
    
# System Properties.
e = 0.0002
D = 0.25 # m
rho = 1000 # kg/m^3
mu = 6e-4 # Pa/s
L = 110 # m
V = 10 # m/s

# Bisection Properties
a = 0.0001
b = 1
tol = 0.0001
maxiter = 50
plot_output = True
verbose = True

# Reynolds Number
Re = rho * V * D / mu

# Friction factor
if Re < 2300:
    # Report that the flow is laminar and calculate the friction factor
    # using the simple formula
    if verbose:
        print('\nThe flow is laminar.')
    f = 64. / Re

else:
    # Report that the flow is turbulent, declare the function handle for
    # the Colebrook Equation and run bisection to find the friction
    # factor.
    if verbose:
        print('\nThe flow is turbulent.')
    fun = lambda x: colebrook_equation(x, e, D, Re)
    f = bisection(fun, a, b, tol, maxiter,plot_output)

if verbose:
    print('The friction factor is {0:10.8f}.'.format(f))

# Calculate and Report Head Loss
head_loss = f * L * V ** 2 / (D * 2)

print('The head loss due to frictional effects is '
      '{0:5.2f} (m/s)^2.'.format(head_loss))
