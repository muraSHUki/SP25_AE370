import sys                                                                                         #
import os                                                                                          #
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))                 #
                                                                                                   #
from flutter_model import wing_flutter_rhs, rk45_solver                                            #
import numpy as np                                                                                 #
import matplotlib.pyplot as plt                                                                    #
####################################################################################################

###### SIMULATION PARAMETERS #######################################################################
# --- World Parameters
params = {
    'rho': 1.225,
    'U': 60.0,
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

# --- Time range
t_span = (0, 10)

###### RUN SIMULATION ##############################################################################
# --- Run
t_vals, y_vals = rk45_solver(wing_flutter_rhs, t_span, y0, params)

# --- Plot
plt.figure(figsize=(10, 6))
plt.plot(t_vals, y_vals[:, 0], label='Plunge h(t)')
plt.plot(t_vals, y_vals[:, 2], label='Pitch θ(t)')
plt.xlabel('Time (s)')
plt.ylabel('Displacement')
plt.title(f'Custom RK45 Wing Flutter Simulation (U = {params['U']} m/s)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig("../results/flutter_sim.png", dpi=300)