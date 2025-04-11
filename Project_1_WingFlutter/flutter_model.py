import numpy as np

def wing_flutter_rhs(t, y, params):
    """
    Compute the right-hand side of the 2-DOF pitch-plunge aeroelastic system.

    Parameters:
        t (float): Current time (unused, for generality).
        y (ndarray): State vector [h, h_dot, theta, theta_dot].
        params (dict): Dictionary containing physical and aerodynamic parameters.

    Returns:
        dydt (ndarray): Derivative vector [h_dot, h_ddot, theta_dot, theta_ddot].
    """

    h, h_dot, theta, theta_dot = y
    rho = params['rho']
    U = params['U']
    b = params['b']
    m = params['m']
    I_alpha = params['I_alpha']
    S_alpha = params['S_alpha']
    k_h = params['k_h']
    k_theta = params['k_theta']
    CL_alpha = params['CL_alpha']
    CM_alpha = params['CM_alpha']

    alpha_eff = theta + h_dot / U
    L = rho * U**2 * b * CL_alpha * alpha_eff
    M = rho * U**2 * b**2 * CM_alpha * alpha_eff

    h_ddot = (L - S_alpha * theta_dot - k_h * h) / m
    theta_ddot = (M - S_alpha * h_ddot - k_theta * theta) / I_alpha

    return np.array([h_dot, h_ddot, theta_dot, theta_ddot])

def rk45_step_scaled(fun, t, y, h, params, rtol, atol):
    """
    Perform a single adaptive RK45 (Dormand-Prince) integration step.

    Parameters:
        fun (function): RHS function of the ODE system.
        t (float): Current time.
        y (ndarray): Current state.
        h (float): Current time step.
        params (dict): Parameters passed to the RHS function.
        rtol (float): Relative error tolerance.
        atol (float): Absolute error tolerance.

    Returns:
        y5 (ndarray): Fifth-order accurate estimate of the next state.
        err (float): RMS scaled error between 5th- and 4th-order estimates.
    """

    a = [[],
         [1/5],
         [3/40, 9/40],
         [44/45, -56/15, 32/9],
         [19372/6561, -25360/2187, 64448/6561, -212/729],
         [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]]
    
    c = [0, 1/5, 3/10, 4/5, 8/9, 1]
    b5 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
    b4 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100]

    k = []
    for i in range(6):
        yi = y.copy()
        for j in range(i):
            yi += h * a[i][j] * k[j]
        k.append(fun(t + c[i] * h, yi, params))

    y5 = y + h * sum(bi * ki for bi, ki in zip(b5, k))
    y4 = y + h * sum(bi * ki for bi, ki in zip(b4, k))

    scale = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
    err = np.sqrt(np.mean(((y5 - y4) / scale) ** 2))
    return y5, err

def rk45_solver(fun, t_span, y0, params, h_init=1e-2, rtol=1e-4, atol=1e-6, max_steps=100000, h_min=1e-6, h_max=0.2):
    """
    Integrate an ODE system using an adaptive RK45 method (Dormand-Prince).

    Parameters:
        fun (function): RHS function of the ODE system.
        t_span (tuple): Tuple (t0, tf) defining the integration time interval.
        y0 (ndarray): Initial state vector.
        params (dict): Parameters for the RHS function.
        h_init (float): Initial time step size.
        rtol (float): Relative tolerance for adaptive stepping.
        atol (float): Absolute tolerance for adaptive stepping.
        max_steps (int): Maximum number of steps before stopping.
        h_min (float): Minimum allowable step size.
        h_max (float): Maximum allowable step size.

    Returns:
        t_values (ndarray): Array of time values.
        y_values (ndarray): Array of state vectors corresponding to t_values.
    """
    t0, tf = t_span
    t = t0
    y = np.array(y0, dtype=float)
    h = h_init

    t_values = [t]
    y_values = [y.copy()]

    for _ in range(max_steps):
        if t >= tf:
            break

        if t + h > tf:
            h = tf - t

        y_next, err = rk45_step_scaled(fun, t, y, h, params, rtol, atol)

        if err <= 1.0:
            t += h
            y = y_next
            t_values.append(t)
            y_values.append(y.copy())

        # Step size adaptation
        safety = 0.9
        if err == 0:
            h_new = h * 2
        else:
            h_new = h * min(max(safety * err**(-0.2), 0.2), 5.0)
        h = max(min(h_new, h_max), h_min)

    return np.array(t_values), np.array(y_values)

def rk4_solver(fun, t_span, y0, h, params):
    """
    Classic fixed-step Runge-Kutta 4th order integrator.

    Parameters:
        fun     -- RHS function: dy/dt = f(t, y, params)
        t_span  -- Tuple (t0, tf): time range to integrate over
        y0      -- Initial state vector
        h       -- Fixed time step size
        params  -- Dictionary of system parameters

    Returns:
        t_values, y_values: Arrays of time points and state vectors
    """
    t0, tf = t_span
    t_vals = [t0]
    y_vals = [np.array(y0, dtype=float)]
    t = t0
    y = np.array(y0, dtype=float)

    while t < tf:
        if t + h > tf:
            h = tf - t
        k1 = fun(t, y, params)
        k2 = fun(t + h/2, y + h/2 * k1, params)
        k3 = fun(t + h/2, y + h/2 * k2, params)
        k4 = fun(t + h, y + h * k3, params)
        y += (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        t_vals.append(t)
        y_vals.append(y.copy())

    return np.array(t_vals), np.array(y_vals)