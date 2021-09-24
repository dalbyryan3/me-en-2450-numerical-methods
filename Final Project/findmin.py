import numpy as np

def brute(f, ranges, args=(), Ns=3, full_output=False):
    """Minimize a function over a given range by brute force.

    Uses the "brute force" method, i.e. computes the function's value
    at each point of a multidimensional grid of points, to find the global
    minimum of the function.

    See `optimize` for a description of inputs and ouputs

    """
    # Generate the parameter space
    lrange = list(ranges)
    N = len(ranges)
    for k in range(N):
        low, high = lrange[k]
        lrange[k] = np.linspace(low, high, Ns)
    xs = np.array(np.meshgrid(*lrange)).T.reshape(-1, N)
    return find_fmin_on_grid(f, xs, args, full_output)


def monte_carlos(f, ranges, args=(), Ns=1000, full_output=False):
    """Minimize a function over a given range by Monte Carlos sampling

    Uses Monte Carlos method, i.e. computes the function's value
    at each point of a multidimensional grid of random points, to find the
    minimum of the function.

    See `optimize` for a description of inputs and ouputs

    """
    # Generate the parameter space
    np.random.seed(17)
    lrange = list(ranges)
    N = len(ranges)
    xs = np.zeros((Ns, N))
    for k in range(N):
        low, high = lrange[k]
        xs[:, k] = np.random.uniform(low, high, Ns)
    return find_fmin_on_grid(f, xs, args, full_output)


def find_fmin_on_grid(f, xs, args, full_output):
    """Loop over sets of parameters in parameter race and call the objective
    function for each parameter. After collecting the output, find the
    realization with the minimum.
    """
    Nx = len(xs)
    Jout = np.zeros(Nx)
    for k in range(Nx):
        Jout[k] = f(xs[k], *args)
    idx = np.nanargmin(Jout)
    if not full_output:
        return xs[idx], Jout[idx]
    return xs[idx], Jout[idx], xs, Jout


def findmin(f, ranges, args=(), Ns=None, full_output=False, method='brute',
            finish=False):
    """Wrapper around optimization methods

    Parameters
    ----------
    f : callable
        The objective function to be minimized. Must be in the
        form ``f(x, *args)``, where ``x`` is the argument in
        the form of a 1-D array and ``args`` is a tuple of any
        additional fixed parameters needed to completely specify
        the function.
    ranges : tuple
        Each component of the `ranges` tuple must be a range tuple of the
        form ``(low, high)``. The program uses these to create the grid of
        points on which the objective function will be computed.
    args : tuple
        Any additional fixed parameters needed to completely specify
        the function.
    Ns : int, optional
        Number of grid points along the axes, if not otherwise
        specified.
    full_output : bool, optional
        If True, return the evaluation grid and the objective function's
        values on it.
    method : string, optional
        The optimization method
    finish : bool, optional
        If True, finish the optimization using scipy.optimize.fmin using the
        results of the "method" minimization as the initial guess.

    Returns
    -------
    x0 : ndarray
        A 1-D array containing the coordinates of a point at which the
        objective function had its minimum value.
    fval : float
        Function value at the point `x0`. (Returned when `full_output` is
        True.)
    grid : ndarray
        Representation of the evaluation grid.  It has the same
        length as `x0`. (Returned when `full_output` is True.)
    Jout : ndarray
        Function values at each point of the evaluation grid. (Returned
        when `full_output` is True.)

    """
    if method == 'brute':
        Ns = Ns or 3
        x0, J0, xs, Jout = brute(f, ranges, args=args, Ns=Ns, full_output=True)
    elif method == 'monte carlos':
        Ns = Ns or 1000
        x0, J0, xs, Jout = monte_carlos(f, ranges, args=args, Ns=Ns, full_output=True)
    else:
        valid_methods = ('brute', 'monte carlos')
        raise ValueError('optimization method must be one of {0!r}'.format(
            ', '.join(valid_methods)))

    # Mask any values that are not finite
    mask = np.isfinite(Jout)
    xs = xs[mask]
    Jout = Jout[mask]
    if not len(xs):
        raise RuntimeError('Failed to find optimized parameters')

    if finish:
        import scipy.optimize
        res = scipy.optimize.fmin(f, x0, args=args, full_output=True)
        x0, J0 = res[0:2]

    if not full_output:
        return x0
    return x0, J0, xs, Jout
