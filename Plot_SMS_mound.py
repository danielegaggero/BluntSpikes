#%%
import matplotlib.pyplot as plt
import numpy as np

from copy import copy
import os
#os.chdir('/Users/daniele/Dropbox/Fisica/2023/EvolvingDensities-main/')

from scipy.interpolate import interp1d, UnivariateSpline
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from astropy.cosmology import Planck18
from evolving_densities import Density, smooth_Bspline
from polytropic import PolytropicSolver
from colourmap_maker import lch_colour_map, luv_colour_map

rgb_palette = np.array([[255, 97, 229], [241, 0, 123], 
                        [151, 0, 71], [54, 3, 27], 
                        [179, 126, 12], [255, 190, 11], 
                        [250, 237, 185], [86, 235, 215],
                        [0, 128, 117], [0, 59, 59]], dtype = np.float32)/255

rgb_palette_dict = {'purple pizzazz': rgb_palette[0], 'flickr pink': rgb_palette[1],
                    'jazzberry jam': rgb_palette[2], 'dark sienna': rgb_palette[3],
                    'dark goldenrod': rgb_palette[4], 'amber': rgb_palette[5],
                    'blond': rgb_palette[6], 'turquiose': rgb_palette[7],
                    'pine green': rgb_palette[8], 'rich black': rgb_palette[9]}

#%%
import sys
from importlib import reload
reload(sys.modules['evolving_densities'])
from evolving_densities import Density

plt.rcParams.update({
    "text.usetex": True
})

#%%

G_N = 4.302e-3 #Units of (pc/solar mass) (km/s)^2
c_light = 2.9979e05 #km/s

G_N_Mpc = 1e-6*4.302e-3 #(Mpc/solar mass) (km/s)^2

h = 0.678
Omega_DM = 0.1186/(h**2)
H0 = 100.0*h #(km/s) Mpc^-1
H0_peryr = 67.8*(3.24e-20)*(60*60*24*365)
ageUniverse = 13.799e9 #y
Omega_L = 0.692
Omega_m = 0.308
Omega_r = 9.3e-5

z_eq = 3375.0
rho_eq = 1512.0 #Solar masses per pc^3
sigma_eq = 0.005 #Variance of DM density perturbations at equality
lambda_max = 3.0 #Maximum value of lambda = 3.0*z_dec/z_eq (i.e. binaries decouple all the way up to z_dec = z_eq)

def Hubble(z):
    return H0_peryr*np.sqrt(Omega_L + Omega_m*(1+z)**3 + Omega_r*(1+z)**4)

def Hubble2(z):
    return H0*np.sqrt(Omega_L + Omega_m*(1+z)**3 + Omega_r*(1+z)**4)

rho_critical_today_Mpc = 3.0*H0**2/(8.0*np.pi*G_N_Mpc) #Solar masses per Mpc^3
rho_critical_today = rho_critical_today_Mpc / 1.e18

def rho_critical_Mpc(z):
    return 3.0*Hubble2(z)**2/(8*np.pi*G_N_Mpc) #Solar masses per pc^3

def rho_critical(z):
    return 3.0*Hubble2(z)**2/(8*np.pi*G_N_Mpc*1.e18) #Solar masses per pc^3

print("Critical density today [MSun/pc**3] ", rho_critical(0))

r_array = np.logspace(-10, 3, num = 1301)

M_halo = 1.E8
z = 15
## Concentration
c = 3. #https://arxiv.org/pdf/1502.00391.pdf Fig. 7

#R_vir = 1E3 #0.784*(Planck18.Om0/Planck18.Om(15))**(-1/3) * (M_halo*1E-8*Planck18.h)**(1/3) * (10/(z + 1)) / Planck18.h *1E3
## Virial radius in pc
R_vir = (3.*M_halo/(4.*np.pi*200.*rho_critical(z)))**(1./3)  #pc

print("Critical density at z = 15", rho_critical(z), " [Msun/pc**3]")
print("Virial radius at z = 15", R_vir, " [pc]")

"""
R_s = R_vir/c
a_NFW = np.log(1 + c) - c/(1 + c)
rho_s = M_halo/(4. * np.pi * R_s**3. * a_NFW)
def rho_NFW(r):
    return rho_s * (R_s/r) / (1 + r/R_s)**2 * np.exp(-r/R_vir)
"""

R_s = R_vir/c
a_NFW = np.log(1 + c) - c/(1 + c)
rho_halo = M_halo/(4/3 * np.pi * R_vir**3)
rho_0_prime = rho_halo*c**3/(3*a_NFW)
def rho_NFW(r):
    return rho_0_prime * (R_s/r) / (1 + r/R_s)**2 * np.exp(-r/R_vir)

#input("Press Enter to continue...")

rho_0_gas_i = 1E-23 * (3.0857E18)**3 / (1.9885E33) # M_sun/pc^3
r_c_i = 1E1
def rho_gas_initial(r):
    return rho_0_gas_i / (1 + r/r_c_i)**2

rho_0_gas_f = 1E-15 * (3.0857E18)**3 / (1.9885E33) # M_sun/pc^3
r_c_f = 1E-3
def rho_gas_final(r):
    return rho_0_gas_f / (1 + r/r_c_f)**2

plt.figure()
plt.loglog(r_array, rho_NFW(r_array), c = rgb_palette_dict['rich black'])
plt.loglog(r_array, rho_gas_initial(r_array), c = rgb_palette_dict['pine green'])
plt.loglog(r_array, rho_gas_final(r_array), c = rgb_palette_dict['turquiose'])
plt.ylim(bottom = 1E-4)
#plt.show()

#%%

#rho_M_data = np.genfromtxt('/Users/daniele/Dropbox/Fisica/2023/EvolvingDensities-main/rho_M.csv', delimiter = ',')
rho_M_data = np.genfromtxt('rho_M.csv', delimiter = ',')

logrho_M_interp = interp1d(rho_M_data[:, 0], np.log10(rho_M_data[:, 1]), kind = 'quadratic', bounds_error = False,
                           fill_value = (np.log10(rho_M_data[0, 1]), 0))
rho_M_interp = lambda x: 10**logrho_M_interp(x) * (3.0857E18)**3 / (1.9885E33)

x_test = np.linspace(0, np.max(rho_M_data[:, 0]), num = 1000)
x_test = np.append(x_test, np.max(rho_M_data[:, 0]) - np.logspace(-8, -1, num = 1000))
x_test = np.unique(x_test)

r_reconstructed = [0]
M_tot = 1E5 # M_sun

for i in range(len(x_test) - 1):

    dM = (x_test[i + 1] - x_test[i]) * M_tot # M_sun

    rho = (rho_M_interp(x_test[i]) + rho_M_interp(x_test[i + 1]))/2 

    r = r_reconstructed[-1]

    dr = np.cbrt(3*dM/(4*np.pi*rho) + r**3) - r

    r_reconstructed.append(r + dr)

    #if i%100 == 0: print(r_reconstructed[-1], dr)

r_reconstructed = np.array(r_reconstructed)

rho_SMS_bloated_interp = interp1d(np.array(r_reconstructed), rho_M_interp(x_test), fill_value = (rho_M_interp(x_test[0]), 0), bounds_error = False)

m_proto = 1.E5
n = 3
rho_c = 0.1 * (3.0857E18)**3 / (1.9885E33) # M_sun/pc^3
poly_solver = PolytropicSolver(n)
r_array_poly, rho_array_poly = poly_solver(rho_c, m_proto)
rho_proto_poly_interp = interp1d(r_array_poly, rho_array_poly, fill_value = (rho_array_poly[0], 0), bounds_error = False)

r_core = 5E-7

#%%

plt.figure()
plt.loglog(r_array, rho_NFW(r_array), c = rgb_palette_dict['dark sienna'], label = 'initial NFW')
#plt.loglog(r_array, rho_gas_initial(r_array), c = rgb_palette_dict['pine green'], label = 'initial gas cloud')
#plt.loglog(r_array, rho_gas_final(r_array), c = rgb_palette_dict['turquiose'], label = 'collapsed gas cloud')
plt.loglog(r_array, rho_SMS_bloated_interp(r_array), c = rgb_palette_dict['flickr pink'], ls="--", label = r'$10^5 M_\odot$ SMS, Hosokawa solution')
plt.loglog(r_array_poly, rho_proto_poly_interp(r_array_poly), c = rgb_palette_dict['pine green'], label = r'$10^5 M_\odot$ SMS, Lane-Emden solution')
# plt.loglog(r_array, rho_proto_poly_interp(r_array), c = rgb_palette_dict['flickr pink'])
plt.xlim(1E-9, 1E3)
plt.ylim(bottom = 1E-4)
plt.xlabel(r'$r$ (pc)')
plt.ylabel(r'$\rho$ ($M_\odot$ pc$^{-3}$)')
plt.legend()
plt.savefig('./figures/density_NFW_gas_star_densities.pdf')
plt.show()
#%%

print("> Creating Density objects...")

N_part = int(1e2) #
density_NFW = Density('density_NFW_SMS', rho_NFW, r_array, N_particles = N_part)
density_NFW.setup_potentials()
psi_NFW = density_NFW.get_psi()
M_tot_NFW = density_NFW.M_enclosed(r_array[-1])
rho_initial = density_NFW(r_array)

density_gas_initial = Density('density_gas_initial', rho_gas_initial, r_array, N_particles = int(5E0))
density_gas_initial.setup_potentials()
psi_gas_i = density_gas_initial.get_psi()
M_tot_gas_i = density_gas_initial.M_enclosed(r_array[-1])

density_gas_final = Density('density_gas_final', rho_gas_final, r_array, N_particles = int(5E0))
density_gas_final.setup_potentials()
psi_gas_f = density_gas_final.get_psi()
M_tot_gas_f = density_gas_final.M_enclosed(r_array[-1])

print("> ...Done.")

# plt.figure()
# plt.loglog(r_array, psi_NFW(r_array), c = rgb_palette_dict['rich black'], label = 'initial NFW')
# plt.loglog(r_array, psi_gas_i(r_array), c = rgb_palette_dict['pine green'], label = 'initial gas cloud')
# plt.loglog(r_array, psi_gas_f(r_array), c = rgb_palette_dict['turquiose'], label = 'collapsed gas cloud')
# plt.xlabel('$r$ (pc)')
# plt.ylabel(r'$\psi$')
# plt.legend()
# plt.savefig('./figures/density_NFW_gas_potentials.pdf')
# plt.show()

#%%

density_proto_poly = Density('density_proto_poly', rho_proto_poly_interp, r_array, N_particles = int(5E0))
density_proto_poly.setup_potentials()
psi_proto_poly = density_proto_poly.get_psi()
M_tot_proto_poly = density_proto_poly.M_enclosed(r_array[-1])

density_SMS_bloated = Density('density_SMS_bloated', rho_SMS_bloated_interp, r_array, N_particles = int(5E0))
#density_SMS_bloated.smoothen_density()
density_SMS_bloated.setup_potentials()
psi_SMS_bloated = density_SMS_bloated.get_psi()
M_tot_SMS_bloated = density_SMS_bloated.M_enclosed(r_array[-1])

#%%
plt.figure()
plt.loglog(r_array, psi_NFW(r_array), c = rgb_palette_dict['dark sienna'], label = 'initial NFW')
plt.loglog(r_array, psi_gas_i(r_array), c = rgb_palette_dict['pine green'], label = 'initial gas cloud')
plt.loglog(r_array, psi_gas_f(r_array), c = rgb_palette_dict['turquiose'], label = 'collapsed gas cloud')
plt.loglog(r_array, psi_SMS_bloated(r_array), c = rgb_palette_dict['flickr pink'], label = r'$10^5 M_\odot$ SMS')
plt.xlim(1E-9, 1E3)
plt.legend()
plt.xlabel('$r$ (pc)')
plt.ylabel(r'$\psi$')
plt.savefig('./figures/density_NFW_gas_star_potentials.pdf')
#plt.show()

#%%

m_BH = M_tot_SMS_bloated*1.
r_S = 2*G_N*m_BH/c_light**2
def psi_BH(r):
    return G_N*m_BH/r

r_sp = 0.122 * R_s * ( m_BH / ( rho_0_prime * R_s**3 ) )**(1./(3.-1))


# delta_psi = lambda r: psi_SMS_bloated(r) - psi_proto_poly(r)
print("> Supermassive star formation...")
print("> Calling Adiabatic Growth...")
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
density_after_SMS = density_NFW.adiabatic_growth(psi_SMS_bloated, refinement = 15, figures = False)
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#%%

plt.figure()
plt.loglog(r_array, rho_initial, c = rgb_palette_dict['dark sienna'], label = 'NFW')
#plt.loglog(r_array, rho_GS_no_r_S, c = rgb_palette_dict['amber'], label = r'GS, $m_{BH} = 10^5 M\odot$') #density_GS_pred
# plt.loglog(r_array, rho_after_proto, c = rgb_palette_dict['purple pizzazz'], label = 'after isothermal collapse')
plt.loglog(r_array, density_after_SMS(r_array), c = rgb_palette_dict['flickr pink'], label = 'after SMS formation')
plt.axvline(r_core, c = rgb_palette_dict['turquiose'], ls = '--')
plt.ylim(1E0, 1E25)
plt.xlim(1E-9, 1E3)
plt.legend()
plt.ylabel(r'$\rho$ [$M_\odot$pc$^{-3}$]')
plt.xlabel(r'$r$ (pc)')
plt.savefig('./figures/density_NFW_rho_SMS.pdf')
#plt.show()

