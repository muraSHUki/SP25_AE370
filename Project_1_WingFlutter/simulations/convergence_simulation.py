import sys                                                                                         #
import os                                                                                          #
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))                 #
                                                                                                   #
from flutter_model import wing_flutter_rhs, rk4_solver                                             #
import numpy as np                                                                                 #
import matplotlib.pyplot as plt                                                                    #
####################################################################################################



###### SIMULATION PARAMETERS #######################################################################
# --- World Parameters
params = {
    'rho': 1.225, 
    'U': 30.0, 
    'b': 0.5, 
    'm': 100.0,
    'I_alpha': 20.0, 
    'S_alpha': 0.0,
    'k_h': 5e4, 
    'k_theta': 2000.0,
    'CL_alpha': 2*np.pi, 
    'CM_alpha': -0.1 * 2*np.pi
}

# --- Initial Conditions
y0 = [0.01, 0.0, 0.01, 0.0]

# --- Time Range
t_span = (0, 2.0)



# ###### RUN SIMULATION ##############################################################################
# --- Reference Solution
_, y_ref = rk4_solver(wing_flutter_rhs, t_span, y0, 1e-5, params)
y_ref_final = y_ref[-1]

# --- Run for different h
hs = [0.2, 0.1, 0.05, 0.025, 0.0125]
errors = []

for h in hs:
    _, y_vals = rk4_solver(wing_flutter_rhs, t_span, y0, h, params)
    y_final = y_vals[-1]
    err = np.linalg.norm(y_final - y_ref_final) / np.linalg.norm(y_ref_final)
    errors.append(err)

# --- Plot error vs h
plt.figure(figsize=(10, 6))
plt.loglog(hs, errors, 'o-', label='RK4 error')
plt.loglog(hs, [e * (hs[0]/h)**4 for h, e in zip(hs, errors)], 's--', label='$\mathcal{O}(h^4)$')
plt.xlabel('Time Step Size h', fontsize=14)
plt.ylabel('Relative Error', fontsize=14)
plt.title('RK4 Convergence for Wing Flutter Model')
plt.grid(True, which='both')
plt.xticks(hs, [h for h in hs])
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()
plt.savefig("../results/convergence.png", dpi=300)