import numpy as np

def fdtd_update(p_nm1, p_n, dx, dt, c, domain_mask=None):
    """
    Performs a single FDTD update step for the 2D scalar wave equation.

    Discretized update:
        p_np1[i,j] = 2*p_n[i,j] - p_nm1[i,j] + (c*dt/dx)^2 * Laplacian[p_n]

    Parameters:
        p_nm1 : ndarray, pressure field at time step n-1
        p_n   : ndarray, pressure field at time step n
        dx    : spatial grid size (assumed square grid)
        dt    : time step size
        c     : wave speed
        domain_mask : optional boolean array (True = valid region)

    Returns:
        p_np1 : ndarray, pressure field at next time step (n+1)
    """
    Nx, Ny = p_n.shape
    p_np1 = np.zeros_like(p_n)

    # Interior FDTD update (second-order central difference)
    p_np1[1:-1, 1:-1] = (
        2 * p_n[1:-1, 1:-1] - p_nm1[1:-1, 1:-1] +
        (c * dt / dx)**2 * (
            p_n[2:, 1:-1] + p_n[:-2, 1:-1] +
            p_n[1:-1, 2:] + p_n[1:-1, :-2] -
            4 * p_n[1:-1, 1:-1]
        )
    )

    # Boundary conditions: fixed (Dirichlet)
    p_np1[0, :] = p_np1[-1, :] = 0
    p_np1[:, 0] = p_np1[:, -1] = 0

    # Apply mask to suppress values outside the domain
    if domain_mask is not None:
        p_np1[~domain_mask] = 0

    return p_np1
