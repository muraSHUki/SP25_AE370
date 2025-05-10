import numpy as np

def fdtd_step(u_prev, u_curr, c, dt, dx, dy, mask=None, source_func=None, source_loc=None, t=0.0):
    """
    Perform one FDTD time step for the 2D wave equation.

    Parameters
    ----------
    u_prev : 2D ndarray
        Wave field at time step n-1.
    u_curr : 2D ndarray
        Wave field at time step n.
    c : float
        Wave speed.
    dt : float
        Time step size.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    mask : 2D ndarray or None
        Boolean array indicating interior (True) vs exterior (False) grid points.
    source_func : function or None
        Time-dependent source function f(t).
    source_loc : tuple of ints or None
        Grid location (i, j) where source is applied.
    t : float
        Current simulation time.

    Returns
    -------
    u_next : 2D ndarray
        Wave field at time step n+1.
    """
    nx, ny = u_curr.shape
    u_next = np.zeros_like(u_curr)

    # Interior update using central differences
    u_next[1:-1, 1:-1] = (
        2 * u_curr[1:-1, 1:-1] - u_prev[1:-1, 1:-1]
        + (c * dt) ** 2 * (
            (u_curr[2:, 1:-1] - 2 * u_curr[1:-1, 1:-1] + u_curr[:-2, 1:-1]) / dx**2 +
            (u_curr[1:-1, 2:] - 2 * u_curr[1:-1, 1:-1] + u_curr[1:-1, :-2]) / dy**2
        )
    )

    # Apply source term if defined
    if source_func is not None and source_loc is not None:
        i, j = source_loc
        u_next[i, j] += (dt ** 2) * source_func(t)

    # Apply boundary or masking
    if mask is not None:
        u_next[~mask] = 0.0

    return u_next
