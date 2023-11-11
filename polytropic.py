import numpy as np
from scipy.integrate import solve_ivp

def lane_emden_root(t, y, n):
    return y[0]

def lane_emden_integrand(t, y, n):

    """
    Lane-Emden: d/dx (x^2*dtheta/dx) = -x^2*theta^n
    define z = x^2*dtheta/dx gives
    dz/dx = -x^2*theta^n
    dtheta/dx = z/x^2
    define y = [theta, z]
    """

    return [y[1]/t**2, -t**2*y[0]**n]

class PolytropicSolver:

    def __init__(self, n):
        self._n = n

        lane_emden_root.terminal = True
        lane_emden_root.direction = -1

        delta_x = 1E-4
        theta_0 = 1 - delta_x**2/6 
        z_0 = -delta_x**3/3

        self._sol = solve_ivp(lane_emden_integrand, [delta_x, 50], [theta_0, z_0], events = lane_emden_root, args = (self._n,), dense_output = True)

    def __call__(self, rho_c, M):

        x_array = np.linspace(0, self._sol.t[-1], num = 1000)

        rho_array = rho_c*(self._sol.sol(x_array)[0])**self._n

        M_hat = 4*np.pi*np.trapz(x_array**2 * rho_array, x_array)

        alpha = np.cbrt(M/M_hat)
        r_array = alpha*x_array

        return r_array, rho_array
    
