import numpy as np

def cooling_fin(x, y, d, h, k, Too):
    """Represents the one-dimensional second-order heat equation

        T'' = hP/kA (T - Too)

    as a system of first-order differential equations:

        T' = z
        z' = hP/kA (T - Too)

    Parameters
    ----------
    x : float
        Position along the cooling fin
    y : ndarray
        y[0:2] are the temperature and its gradient at x, respectively
    d, h, k, Too : float
        Cooling fin diameter, heat transfer coefficient, thermal
        conductivity, and ambient temperature, respectively

    Returns
    -------
    fp : ndarray
        Derivative dy/dx.  That is, fp[j]=dy[j]/dx

    """
    # Check inputs
    if d <= 0:
        raise ValueError('Cooling fin diameter must be > 0')
    if h <= 0:
        raise ValueError('Heat transfer coefficient must be > 0')
    if k <= 0:
        raise ValueError('Thermal conductivity must be > 0')
    y = np.asarray(y)
    if len(y) != 2:
        raise ValueError('y must have length = 2')

    P = np.pi * d
    A = np.pi * (d / 2) ** 2
    alpha = (h * P) / (k * A)

    return np.array([y[1], alpha * (y[0] - Too)])


def test_fast():
    h = 20
    k = 200
    d = 0.1
    Too = 300
    x = 1
    y = np.array([100, .5])
    try:
        fp = cooling_fin(x, y, -d, h, k, Too)
        assert False, 'Negative d not detected'
    except ValueError:
        pass
    try:
        fp = cooling_fin(x, y, d, -h, k, Too)
        assert False, 'Negative h not detected'
    except ValueError:
        pass
    try:
        fp = cooling_fin(x, y, d, h, -k, Too)
        assert False, 'Negative k not detected'
    except ValueError:
        pass
    try:
        fp = cooling_fin(x, y[1:], d, h, k, Too)
        assert False, 'len(y) != 2 not detected'
    except ValueError:
        pass

    fp = cooling_fin(x, y, d, h, k, Too)
    P = np.pi * d
    A = np.pi * (d / 2) ** 2
    alpha = (h * P) / (k * A)
    d2Tdx2 = alpha * (y[0] - Too)
    assert abs(y[1] - fp[0]) < np.finfo(float).eps
    assert abs(d2Tdx2 - fp[1]) < np.finfo(float).eps