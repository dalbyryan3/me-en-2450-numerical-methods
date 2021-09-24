import numpy as np


def ball_motion(t, y, params):
    """Ordinary differential equation function that models the dynamics of
    a falling ball.

    Parameters
    ----------
    t : float
        Current time value.
    y : array_like
        [Current y, Current velocity]
    params : array_like
        Physical system parameters
        params[0]: Gravity
        params[1]: Ball Mass
        params[2]: Air density
        params[3]: Ball radius
        params[4]: Drag coef􏰁icient

    Returns
    -------
    [dydt, dvdt] or d([y,v])/dt: float
        [Current estimate of the derivative of position(velocity), Current estimate of the derivative of the velocity.]

    Notes
    -----
    Calculates the value of the derivative of the [position(y), velocity] at the current
    time step (derivatives), given the inputs t (the current time) and y (the
    current [y, velocity]).

    """
    # Declare system constants
    gravity, ball_mass, air_density, ball_radius, cf_drag = params[:5]

    # Calculate intermediate system properties
    area = np.pi * ball_radius ** 2

    # Assign inputs to variables
    velocity = y[1]

    # Calculate forces
    weight = ball_mass * gravity
    drag_force = 0.5 * air_density * area * cf_drag * velocity ** 2

    # Calculate derivative estimates
    acceleration_estimate = (weight - drag_force) / ball_mass
    

    # Assemble output
    dydt = velocity
    dvdt = acceleration_estimate
    

    return dydt, dvdt
