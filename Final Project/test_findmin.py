import numpy as np
from findmin import findmin

def test_brute():
    def fun(x):
        """Function is x1^2+x2^2-8=0"""
        return abs(8 - np.sum(x**2))
    x0, fval, _, _ = findmin(fun, [(1, 3), (1, 5)], Ns=50, method='brute',
                             full_output=True)
    assert fval <= .001
    # Perfectly bracket the roots, so fval should be 0.0 exactly
    x0, fval, _, _ = findmin(fun, [(1, 3), (1, 3)], Ns=3, method='brute',
                             full_output=True)
    assert abs(x0[0]-2) < np.finfo(float).eps
    assert abs(x0[1]-2) < np.finfo(float).eps
    assert fval == 0.0


def test_monte_carlos():
    def fun(x):
        """Function is x1^2+x2^2-8=0"""
        return abs(8 - np.sum(x**2))
    x0, fval, _, _ = findmin(fun, [(1, 3), (1, 3)], Ns=1000, method='monte carlos',
                             full_output=True)
    assert fval <= .05


def test_finish():
    # The optimizer has the option to use a built in optimizer to "finish" the
    # minimization. This test makes sure that the plumbing is set up correctly.
    def fun(x):
        """Function is x1^2+x2^2-8=0"""
        return abs(8 - np.sum(x**2))
    x0, fval, _, _ = findmin(fun, [(1, 3), (1, 3)], Ns=1000, method='monte carlos',
                             full_output=True, finish=True)
    assert fval <= .05


def test_projectile_motion():
    """Find the angle of launch that achieves a specified maximum height for a
    projectile modeled by drag free Newton's law.

    The problem is formulated as a minimization problem. An optimizer calls a
    function at different angles theta. The function calculates the height
    associated with this angle and returns to the optimizer the difference
    between this calculated height and the desired height. The optimizer
    keeps calling the function with different values of theta until this
    difference is minimized. We call this function the "objective function".

    Many physics problems can be set up with a few layers between the
    optimizer, objective function, and the actual physics routine. The
    division of work makes for shorter, easier to understand functions. This
    test demonstrates how to set up such a problem. It solves the
    minimization problem (i.e., finding the minimum difference between
    h(theta) and the desired height) with the following three stages:

    o Physics (projectile_motion): the routine that does the actual physics.
      This routine solves the physics at each time step - computing v(t) and
      a(t) for the projectile. It is called by the driver.
    o Driver (driver): drives the problem forward in time by integrating the
      equations of motion. We use RK4 driver, though other integrators
      could be used. This routine is called by the objective function.
    o Objective function (objfun): This routine computes the thing that we
      want to minimize (h(theta)-h in this case). This routine is called by
      the optimizer. The objective function calls the driver with parameters
      chosen by the optimizer and computes the quantity we want minimized.
    o Optimizer (findmin): We don't usually write our own optimzer, but use
      optimizers available. The optimizer chooses parameters and sends them
      to the objective function. The objective function computes the value to
      be minimized. The optimizer calls the objective function with different
      parameters until the parameter set that gives a minimum is found. If
      the minimum is not found, the optimizer usually returns some kind of
      failure code.

    Optimizers take the objective function as input. They also take starting
    values for each variable to be optimized, or ranges for each variable to
    be optimized. The optimizer used below, takes a range in the form
    (low, high) for each variable to be optimized. It uses the ranges to
    construct a n-dimensional grid on which it searches for the minimum. For
    this problem, we are only optimizing 1 variable (theta), but the
    optimizer can optimize more.

    """
    from rk4 import rk4
    def projectile_motion(t, y, p):
        # Equation of motion in y direction for projectile
        # p: parameters
        # p = [v0, g, theta]
            # note that this is dx/dt with the integrated eqn for dv/dt 
            # plugged in for v. Normally this is kept as v, but the problem is
            # the initial condition v0 depends on theta. This would mean
            # needing to modify the driver or objective function to calculate
            # v0 special.
            # wierd, this form means it doesn't depend on the current position
            # or current velocity
        x, v = y
        v0, g, theta = p
        dydt = np.zeros(2)
        dydt[0] = v0 * np.sin(theta) - g * t
        dydt[1] = -g
        return dydt

    def driver(driver_params, y0, odefun_params):
        # pull out driver parameters
        step_size, max_iters = driver_params
        # Uses RK4 to integrate Newton's law
        odefun = lambda t, y: projectile_motion(t, y, odefun_params)
        t0 = 0.
        t, y = [t0], [y0]
        tspan = np.array([t0, t0+step_size])
        iters = 1
        while iters <= max_iters:
            tp, yp = rk4(odefun, tspan, y[-1])
            if yp[-1,1] < np.finfo(float).eps:
                # Velocity less than 0
                break
            t.append(tp[-1])
            y.append(yp[-1])
            # Increment the initial state for the next solve
            tspan[:] = [tspan[1], tspan[1] + step_size]
            iters = iters + 1
        y = np.asarray(y)
        return np.asarray(t), y[:,0], y[:,1]

    def objfun(x, obj_params, driver_params, y0, odefun_params):
        # Objective function, returns difference between desired and computed
        # heights
        
        # first pull out values needed for optimization conditions
        height = obj_params[0]
        
        # now make a copy of parameters and replace the spot with nans with the
        # current values found by findmin that are placed in x. The order of x
        # depends on the order of the ranges fed to findmin
        p = odefun_params.copy()
        p[2] = x[0]
        
        # now run the rk4 on your odefun via the driver, which runs it one step
        # at a time and stops if velocity goes negative
        t, y, v = driver(driver_params, y0, p)
        
        ### now take the results and use the optimization conditions to
        ### set fval for findmin to work with to determine if this is the
        ### optimal value, or if it is another
    
        # take the resulting values and find the highest height reached
        h = np.amax(y)
        # make sure height didn't come out as negative
        assert h >= 0.
        # optimization condition is smallest difference between current height
        # and desired height for this problem
        return abs(height-h) / height

    # Now do the actual test. odefun_params below is the parameter array that 
    # is sent to the driver and, eventually, the projectile motion function.
    #     look at time a params is separated out to know what order and what
    #     values are contained
    # The angle of attack is the variable we actually want to optimize, so we
    # set its value to NaN when constructing params. The optimizer will choose
    # values for theta and those values will be passed to the driver for
    # projectile motion
    
    # set values needed for other sections
    g = 9.81    # gravity, m/s^2
    x0 = 0.0    # initial position, m
    v0 = 10.0   # initial velocity, m/s
    
    # set values for optimizing conditions
    desired_theta = np.pi/4  # This is the value we are trying to find!
    desired_height = (v0**2 * np.sin(desired_theta)**2) / (2*g)     # we have the right theta when we get this height!
    obj_params = np.zeros([1,1])
    obj_params[0] = desired_height   # m, the height we are trying to find via optimization
    
    # setup an array of constant parameters
    odefun_params = np.zeros([3,1])
    odefun_params[0] = v0       # initial velocity, m/s
    odefun_params[1] = g        # gravity, m/s^2
    odefun_params[2] = np.nan      # fill this with theta during optimize, radians

    # Set the search domain for theta. It has to be somewhere between 0 and 90
    # degrees. The optimizer will use the range to create the search grid to
    # optimize.
    ranges = [
        (0., np.pi/2)     # theta, radians
    ]
    
    # set driver function parameters
    step_size = 0.01
    max_iters = 100
    driver_params = np.array([step_size,max_iters])
    y0 = np.array([x0,v0])    # vector of initial conditions

    # Objective functions are usually functions of many variables, not just the
    # variable being optimized. But, optimizers often only call the objective
    # function with the parameters being optimized. Thus, we create a lambda
    # function that is a function of just the optimized variables and send it
    # to the optimizer. The lambda function is just a wrapper around the actual
    # objective function and allows us to pass additional information needed by
    # the actual objective function.
    f = lambda x: objfun(x, obj_params, driver_params, y0, odefun_params)

    # Now call the optmizer.
    res = findmin(f, ranges, method='brute', Ns=20)
    theta_opt = res[0]
    err = abs(desired_theta-theta_opt)/desired_theta
    # Large error because we evaluated on a coarse grid
    assert err < .1

    # Try again, using monte carlos, and more points
    res = findmin(f, ranges, method='monte carlos', Ns=500)
    theta_opt = res[0]
    err = abs(desired_theta-theta_opt)/desired_theta
    # Slightly smaller error because we evaluated with more points, but they
    # are random
    assert err < .01

    # Try again, let the optimizer finish with built in optimizer. Error should
    # be smaller.
    res = findmin(f, ranges, method='monte carlos', Ns=10, finish=True)
    theta_opt = res[0]
    err = abs(desired_theta-theta_opt)/desired_theta
    assert err < 1e-4

test_projectile_motion()
