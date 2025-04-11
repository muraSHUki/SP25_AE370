import sys                                                                                         #
import os                                                                                          #
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))                 #
                                                                                                   #
from flutter_model import wing_flutter_rhs, rk4_solver                                             #
import numpy as np                                                                                 #
import matplotlib.pyplot as plt                                                                    #
####################################################################################################



###### FLUTTER ONSET ###############################################################################
def detect_flutter(solver):
    """
    Sweeps a range of freestream velocities to detect flutter onset using the fixed-step RK4 solver.

    Parameters:
    - solver: reference to rk4_solver

    For each velocity U:
    - Integrates the wing flutter system over time
    - Computes max pitch (θ) and plunge (h) amplitudes
    - Calculates system energy to identify unbounded growth (flutter)

    Returns:
    - max_theta_values: list of max |θ(t)| for each U
    - max_h_values: list of max |h(t)| for each U
    - energy_growth_flags: list of booleans indicating flutter (True if energy grew significantly)
    """
    max_theta_values = []
    max_h_values = []
    energy_growth_flags = []

    for U in U_values:
        params = base_params.copy()
        params['U'] = U

        t_vals, y_vals = solver(wing_flutter_rhs, (0, 5), y0, h=0.01, params=params)

        max_theta = np.max(np.abs(y_vals[:, 2]))
        max_h = np.max(np.abs(y_vals[:, 0]))

        m, I, k_h, k_t = params['m'], params['I_alpha'], params['k_h'], params['k_theta']
        h, h_dot, th, th_dot = y_vals[:, 0], y_vals[:, 1], y_vals[:, 2], y_vals[:, 3]
        energy = 0.5 * m * h_dot**2 + 0.5 * k_h * h**2 + 0.5 * I * th_dot**2 + 0.5 * k_t * th**2
        is_growing = np.mean(energy[-10:]) > 2 * np.mean(energy[:10])

        max_theta_values.append(max_theta)
        max_h_values.append(max_h)
        energy_growth_flags.append(is_growing)

    return max_theta_values, max_h_values, energy_growth_flags



###### SIMULATION PARAMETERS #######################################################################
# --- Shared parameters
base_params = {
    'rho': 1.225,
    'b': 0.5,
    'm': 100.0,
    'I_alpha': 20.0,
    'S_alpha': 30.0,
    'k_h': 5e4,
    'k_theta': 2000.0,
    'CL_alpha': 2 * np.pi,
    'CM_alpha': -0.1 * 2 * np.pi
}

# --- Initial Conditions
y0 = [0.01, 0.0, 0.05, 0.0]

# --- Velocity Values
U_values = np.linspace(0.0001, 100, 100)



###### RUN SIMULATION ##############################################################################
# --- Run RK4 solver across all U values
rk4_theta, rk4_h, rk4_growth = detect_flutter(rk4_solver)

# --- Plot
plt.figure(figsize=(10, 6))
plt.plot(U_values, rk4_theta, '--', label='Max θ(t)', alpha=0.7)
plt.plot(U_values, rk4_h, '-', label='Max h(t)', alpha=0.7)

flutter_U_rk4 = next((U for U, grow in zip(U_values, rk4_growth) if grow), None)
if flutter_U_rk4:
    plt.axvline(x=flutter_U_rk4, linestyle='--', color='red', label=f'Flutter Onset ≈ {flutter_U_rk4:.1f} m/s')

plt.xlabel('Freestream Velocity U (m/s)')
plt.ylabel('Max Displacement')
plt.title('Flutter Detection Using RK4')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig("../results/velocity_sim.png", dpi=300)