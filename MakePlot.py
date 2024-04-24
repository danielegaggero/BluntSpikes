#%%
import matplotlib.pyplot as plt
import numpy as np

from copy import copy
import os

from tqdm import tqdm
#os.chdir('/Users/daniele/Dropbox/Fisica/2023/EvolvingDensities-main/')

from scipy.interpolate import interp1d, UnivariateSpline, interpn
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from astropy.cosmology import Planck18
from evolving_densities import Density, smooth_Bspline
from polytropic import PolytropicSolver
from colourmap_maker import lch_colour_map, luv_colour_map

try:
    from scipy.integrate import cumulative_trapezoid
except:
    from scipy.integrate import cumtrapz as cumulative_trapezoid

from timeit import default_timer as timer

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

r_array = np.geomspace(1e-10, 1e0, num = int(1301))
#r_array = np.geomspace(1e-10, 1e3, num = 1301)

print("Check with different r_array from the start! Check with more particles!")

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

M_halo = 1.E7
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


#Number of particles in the sample
#---------------------------------
N_particles = int(5e6)

#N1 is the number of particles in the linear sampling...
N1 = int(N_particles-1)
N2 = N_particles - N1

#Calculate adiabatic phase-space from scratch
#---------------------------------
FROM_SCRATCH = False

#Calculate non-adiabatic samples from scratch
#---------------------------------
FROM_SCRATCH_NA = False


def lin_sampler(x_min, x_max, N):
    u = np.random.rand(N)
    x = (x_max - x_min)*u + x_min
    return x

def log_sampler(x_min, x_max, N):
    return np.exp(lin_sampler(np.log(x_min), np.log(x_max), N))

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

m_proto = M_tot
n = 3
rho_c = 1. * (3.0857E18)**3 / (1.9885E33) # M_sun/pc^3
poly_solver = PolytropicSolver(n)
r_array_poly, rho_array_poly = poly_solver(rho_c, m_proto)
rho_proto_poly_interp = interp1d(r_array_poly, rho_array_poly, fill_value = (rho_array_poly[0], 0), bounds_error = False)

r_poly = np.min(r_array_poly[rho_array_poly <= 0])
print("r_poly:", r_poly)

r_core = 5E-7

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

m_BH = M_tot
#m_BH = 1e-5

r_S = 2*G_N*m_BH/c_light**2
def psi_BH(r):
    return G_N*m_BH/r
    
#Final BH mass after final adiabatic re-growth phase...
#m_BH_final = 1e6
#r_S_final = 2*G_N*m_BH_final/c_light**2




#r_sp = 0.122 * R_s * ( m_BH / ( rho_0_prime * R_s**3 ) )**(1./(3.-1))

gamma_PL = 1.
r_0   = R_s
#rho_0 = rho_0_prime/4.
rho_0 = 1.0*rho_0_prime
gamma_sp_GS = (9. - 2.*gamma_PL)/(4. - gamma_PL)

#alpha_gamma = 0.293 * (gamma_PL)**(4./9.)
alpha_gamma = 0.135

def g_GS(r, m_BH, k):
    r_S = 4*G_N*m_BH/c_light**2
    return np.clip((1. - 2.*r_S/r), 0, None)**k

#print(">------- NOTE THAT WE'RE USING A FACTOR OF 2 TO FUDGE THE RESULTS: CHECK WHERE IT COMES FROM!")
def rho_GS(r, m_BH, k):
    r_sp_GS = alpha_gamma * r_0 * (m_BH / (rho_0 * r_0**3.))**(1./(3. - gamma_PL))
    rho_R = rho_0*(r_sp_GS/r_0)**(-gamma_PL)#0.5*
    return rho_0_prime * (R_s/r) / (1 + r/R_s)**2 * np.exp(-r/R_vir) + g_GS(r, m_BH, k) * rho_R*(r_sp_GS/r)**(gamma_sp_GS)


print(m_BH)
#rho_GS2_array = rho_GS(r_array, m_BH, k=1.00)


rho_r_C = np.loadtxt("results/rho_DCBH.txt", unpack=True, usecols=(1,))


rho_r_2 = np.loadtxt("results/rho_regrowth_2.txt", unpack=True, usecols=(1,))
rho_r_3 = np.loadtxt("results/rho_regrowth_3.txt", unpack=True, usecols=(1,))
rho_r_5 = np.loadtxt("results/rho_regrowth_5.txt", unpack=True, usecols=(1,))
rho_r_10 = np.loadtxt("results/rho_regrowth_10.txt", unpack=True, usecols=(1,))


#-------------------------------------
#-------------------------------------
#-------------------------------------

plt.figure(figsize=(5.5,5.5))

#plt.loglog(r_array, rho_initial, c = rgb_palette_dict['dark sienna'], label = 'Initial NFW')

#plt.loglog(r_array, rho_GS_no_r_S, c = rgb_palette_dict['amber'], label = r'GS, $m_{BH} = 10^5 M\odot$') #density_GS_pred
# plt.loglog(r_array, rho_after_proto, c = rgb_palette_dict['purple pizzazz'], label = 'after isothermal collapse')
#plt.loglog(r_array, rho_array, c = rgb_palette_dict['flickr pink'], label = 'After SMS formation')
#plt.loglog(r_array, rho_r,c='C1', linestyle=':')
#plt.loglog(r_array, rho_r_A,c='C2', linestyle='--', label="After SMS formation (sampled)")
#plt.loglog(r_array, rho_r_B,c='C3', linestyle=':',lw=2, label="After DCBH formation (all)")
plt.loglog(r_array, rho_r_C, c='blue', linestyle='-',lw=2, label=r"After DCBH formation, $m_\mathrm{BH} = 10^5\,M_\odot$")
rho_GS_array = rho_GS(r_array, m_BH*2, k=3.00)
plt.loglog(r_array, rho_GS_array, c='plum', linestyle='--',lw=1.5)#, label=r"After BH growth ($m_\mathrm{BH} = 10^6\,M_\odot$)")
plt.loglog(r_array, rho_r_2,  c='plum', linestyle='-',lw=2, label=r"Subsequent growth to $m_\mathrm{BH} = 2 \times 10^5\,M_\odot$")
#, label=r"Subsequent growth ($m_\mathrm{BH} = 2\times 10^5\,M_\odot$)")#, label=r"After BH growth ($m_\mathrm{BH} = 10^6\,M_\odot$)")
#plt.loglog(r_array, rho_r_3, c='midnightblue', linestyle='-',lw=2, label=r"After BH growth ($m_\mathrm{BH} = 10^6\,M_\odot$)")
#plt.loglog(r_array, rho_r_5, c='midnightblue', linestyle='-',lw=2, label=r"After BH growth ($m_\mathrm{BH} = 10^6\,M_\odot$)")

#rho_GS_array = rho_GS(r_array, m_BH*5, k=1.00)
#plt.loglog(r_array, rho_GS_array, c='royalblue', linestyle='--',lw=2)#, label=r"After BH growth ($m_\mathrm{BH} = 10^6\,M_\odot$)")
#plt.loglog(r_array, rho_r_5,  c='royalblue', linestyle='-',lw=2)#, label=r"After BH growth ($m_\mathrm{BH} = 10^6\,M_\odot$)")

rho_GS_array = rho_GS(r_array, m_BH*10, k=3.00)
plt.loglog(r_array, rho_GS_array, c='darkmagenta', linestyle='--',lw=1.5)#, label=r"After BH growth ($m_\mathrm{BH} = 10^6\,M_\odot$)")
plt.loglog(r_array, rho_r_10, c='darkmagenta', linestyle='-',lw=2, label=r"Subsequent growth to $m_\mathrm{BH} = 10^6\,M_\odot$")
#, label=r"Subsequent growth ($m_\mathrm{BH} = 10^6\,M_\odot$)")#, label=r"After BH growth ($m_\mathrm{BH} = 10^6\,M_\odot$)")

#plt.loglog(r_array, rho_GS2_array, c = 'royalblue', label = r'GS Profile ($m_\mathrm{BH} = 10^6\,M_\odot$)', linestyle='--')
#plt.axvline(r_core, c = rgb_palette_dict['turquiose'], ls = '--')
#plt.axvline(2*r_S, c = 'k', ls = '--')
#plt.axvline(r_poly, c = 'k', ls = '--')

#plt.text(1.4*r_S, 8e19, r"$2 r_\mathrm{s}$", rotation=90)
#plt.text(0.7*r_poly, 8e19, r"$r_\mathrm{SMS}$", rotation=90)
#plt.axvline(3*r_S, c = 'k', ls = '--')
#plt.axvline(4*r_S, c = 'k', ls = '--')

plt.ylim(1E14, 1E21)
plt.xlim(8E-9, 2E-5)
plt.yticks(np.geomspace(1e14, 1e21, 8))
plt.legend(fontsize=13)
plt.ylabel(r'$\rho_\mathrm{DM}$ [$M_\odot \, \mathrm{pc}^{-3}$]')
plt.xlabel(r'$r$ [pc]')
plt.savefig('./figures/density_final_v5.pdf', bbox_inches='tight')


#-------------------------------------
#-------------------------------------
#-------------------------------------

plt.figure(figsize=(5.5,5.5))

#plt.loglog(r_array, rho_initial, c = rgb_palette_dict['dark sienna'], label = 'Initial NFW')

#plt.loglog(r_array, rho_GS_no_r_S, c = rgb_palette_dict['amber'], label = r'GS, $m_{BH} = 10^5 M\odot$') #density_GS_pred
# plt.loglog(r_array, rho_after_proto, c = rgb_palette_dict['purple pizzazz'], label = 'after isothermal collapse')
#plt.loglog(r_array, rho_array, c = rgb_palette_dict['flickr pink'], label = 'After SMS formation')
#plt.loglog(r_array, rho_r,c='C1', linestyle=':')
#plt.loglog(r_array, rho_r_A,c='C2', linestyle='--', label="After SMS formation (sampled)")
#plt.loglog(r_array, rho_r_B,c='C3', linestyle=':',lw=2, label="After DCBH formation (all)")
plt.loglog(r_array, rho_r_C, c=rgb_palette_dict['dark goldenrod'], linestyle='-',lw=2, label=r"After DCBH formation ($m_\mathrm{BH} = 10^5\,M_\odot$)")

rho_GS_array = rho_GS(r_array, m_BH*2, k=3.00)
#plt.loglog(r_array, rho_GS_array, c='cornflowerblue', linestyle='--',lw=2)#, label=r"After BH growth ($m_\mathrm{BH} = 10^6\,M_\odot$)")
plt.loglog(r_array, rho_r_2,  c='lightsteelblue', linestyle='-',lw=2, label=r"Subsequent growth ($m_\mathrm{BH} = 2\times 10^5\,M_\odot$)")#, label=r"After BH growth ($m_\mathrm{BH} = 10^6\,M_\odot$)")

rho_GS_array = rho_GS(r_array, m_BH*3, k=3.00)
#plt.loglog(r_array, rho_GS_array, c='cornflowerblue', linestyle='--',lw=2)#, label=r"After BH growth ($m_\mathrm{BH} = 10^6\,M_\odot$)")
plt.loglog(r_array, rho_r_3,  c='cornflowerblue', linestyle='-',lw=2, label=r"Subsequent growth ($m_\mathrm{BH} = 3\times 10^5\,M_\odot$)")#, label=r"After BH growth ($m_\mathrm{BH} = 10^6\,M_\odot$)")

#plt.loglog(r_array, rho_r_3, c='midnightblue', linestyle='-',lw=2, label=r"After BH growth ($m_\mathrm{BH} = 10^6\,M_\odot$)")
#plt.loglog(r_array, rho_r_5, c='midnightblue', linestyle='-',lw=2, label=r"After BH growth ($m_\mathrm{BH} = 10^6\,M_\odot$)")

rho_GS_array = rho_GS(r_array, m_BH*5, k=3.00)
#plt.loglog(r_array, rho_GS_array, c='royalblue', linestyle='--',lw=2)#, label=r"After BH growth ($m_\mathrm{BH} = 10^6\,M_\odot$)")
plt.loglog(r_array, rho_r_5,  c='royalblue', linestyle='-',lw=2, label=r"Subsequent growth ($m_\mathrm{BH} = 5\times 10^5\,M_\odot$)")#, label=r"After BH growth ($m_\mathrm{BH} = 10^6\,M_\odot$)")

rho_GS_array = rho_GS(r_array, m_BH*10, k=3.00)
#plt.loglog(r_array, rho_GS_array, c='midnightblue', linestyle='--',lw=2)#, label=r"After BH growth ($m_\mathrm{BH} = 10^6\,M_\odot$)")
plt.loglog(r_array, rho_r_10, c='midnightblue', linestyle='-',lw=2, label=r"Subsequent growth ($m_\mathrm{BH} = 10^6\,M_\odot$)")#, label=r"After BH growth ($m_\mathrm{BH} = 10^6\,M_\odot$)")

#plt.loglog(r_array, rho_GS2_array, c = 'royalblue', label = r'GS Profile ($m_\mathrm{BH} = 10^6\,M_\odot$)', linestyle='--')
#plt.axvline(r_core, c = rgb_palette_dict['turquiose'], ls = '--')
#plt.axvline(2*r_S, c = 'k', ls = '--')
#plt.axvline(r_poly, c = 'k', ls = '--')

#plt.text(1.4*r_S, 8e19, r"$2 r_\mathrm{s}$", rotation=90)
#plt.text(0.7*r_poly, 8e19, r"$r_\mathrm{SMS}$", rotation=90)
#plt.axvline(3*r_S, c = 'k', ls = '--')
#plt.axvline(4*r_S, c = 'k', ls = '--')

plt.ylim(1E15, 2E21)
plt.xlim(8E-9, 2E-5)
plt.legend(fontsize=12)
plt.ylabel(r'$\rho_\mathrm{DM}$ [$M_\odot \, \mathrm{pc}^{-3}$]')
plt.xlabel(r'$r$ [pc]')
plt.savefig('./figures/density_final_all.pdf')
plt.show()

