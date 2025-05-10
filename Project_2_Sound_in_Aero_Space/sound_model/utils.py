import numpy as np

###### NORM CALCULATIONS ###########################################################################

def grid_L2_norm(u, dx, dy):
    """
    Compute the grid-function 2-norm of a field u on a uniform grid.

    Parameters
    ----------
    u : 2D ndarray
        Field values at grid points.
    dx, dy : float
        Grid spacing in x and y.

    Returns
    -------
    float
        Grid-based L2 norm: sqrt(sum u_ij^2 * dx * dy)
    """
    return np.sqrt(np.sum(u**2) * dx * dy)

###### VALUE NORMALIZATION ##########################################################################

def normalize_field(u):
    """
    Normalize a 2D field to the range [-1, 1].

    Parameters
    ----------
    u : 2D ndarray
        Field to normalize.

    Returns
    -------
    2D ndarray
        Normalized field.
    """
    max_val = np.max(np.abs(u))
    return u / max_val if max_val != 0 else u

###### MASK HANDLING ###############################################################################

def apply_mask(u, mask):
    """
    Zero out values of u outside the mask.

    Parameters
    ----------
    u : 2D ndarray
        Field to mask.
    mask : 2D boolean ndarray
        True = inside room, False = outside.

    Returns
    -------
    2D ndarray
        Masked field.
    """
    u_masked = u.copy()
    u_masked[~mask] = 0.0
    return u_masked
