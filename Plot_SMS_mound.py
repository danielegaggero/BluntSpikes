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
    "text.usetex": False
})

#%%

G_N = 4.302e-3 #Units of (pc/solar mass) (km/s)^2
c_light = 2.9979e05 #km/s
G_N_Mpc = 1e-6*4.302e-3 #(Mpc/solar mass) (km/s)^2

r_array = np.geomspace(1e-10, 1e0, num = int(1301))
#r_array = np.geomspace(1e-10, 1e3, num = 1301)


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



M_halo = 1.E7
z = 15
## Concentration
c = 3. #https://arxiv.org/pdf/1502.00391.pdf Fig. 7

#R_vir = 1E3 #0.784*(Planck18.Om0/Planck18.Om(15))**(-1/3) * (M_halo*1E-8*Planck18.h)**(1/3) * (10/(z + 1)) / Planck18.h *1E3
## Virial radius in pc
R_vir = (3.*M_halo/(4.*np.pi*200.*rho_critical(z)))**(1./3)  #pc

#print("Critical density today [MSun/pc**3] ", rho_critical(0))
#print("Critical density at z = 15", rho_critical(z), " [Msun/pc**3]")
#print("Virial radius at z = 15", R_vir, " [pc]")

"""
R_s = R_vir/c
a_NFW = np.log(1 + c) - c/(1 + c)
rho_s = M_halo/(4. * np.pi * R_s**3. * a_NFW)
def rho_NFW(r):
    return rho_s * (R_s/r) / (1 + r/R_s)**2 * np.exp(-r/R_vir)
"""


#Number of particles in the sample
#---------------------------------
#N_particles = int(5e6)
N_particles = int(1e6)


#Calculate adiabatic phase-space from scratch
#---------------------------------
FROM_SCRATCH = False

#Calculate non-adiabatic samples from scratch
#---------------------------------
FROM_SCRATCH_NA = True


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

m_proto = M_tot
n = 3
rho_c = 1. * (3.0857E18)**3 / (1.9885E33) # M_sun/pc^3
poly_solver = PolytropicSolver(n)
r_array_poly, rho_array_poly = poly_solver(rho_c, m_proto)
rho_proto_poly_interp = interp1d(r_array_poly, rho_array_poly, fill_value = (rho_array_poly[0], 0), bounds_error = False)

r_poly = np.min(r_array_poly[rho_array_poly <= 0])
#print("r_poly:", r_poly)

r_core = 5E-7

#%%

plt.figure()
plt.loglog(r_array, rho_NFW(r_array), c = rgb_palette_dict['dark sienna'], label = 'initial NFW')
#plt.loglog(r_array, rho_gas_initial(r_array), c = rgb_palette_dict['pine green'], label = 'initial gas cloud')
#plt.loglog(r_array, rho_gas_final(r_array), c = rgb_palette_dict['turquiose'], label = 'collapsed gas cloud')
#plt.loglog(r_array, rho_SMS_bloated_interp(r_array), c = rgb_palette_dict['flickr pink'], ls="--", label = r'$10^5 M_\odot$ SMS, Hosokawa solution')
plt.loglog(r_array_poly, rho_proto_poly_interp(r_array_poly), c = rgb_palette_dict['pine green'], label = r'$10^5 M_\odot$ SMS, Lane-Emden solution')
# plt.loglog(r_array, rho_proto_poly_interp(r_array), c = rgb_palette_dict['flickr pink'])
plt.xlim(1E-9, 1E3)
plt.ylim(bottom = 1E-4)
plt.xlabel(r'$r$ (pc)')
plt.ylabel(r'$\rho$ ($M_\odot$ pc$^{-3}$)')
plt.legend()
plt.savefig('./figures/density_NFW_gas_star_densities.pdf')

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
r_S_final = 2*G_N*m_BH/c_light**2
#m_BH = 1e-5

r_S = 2*G_N*m_BH/c_light**2
def psi_BH(r):
    return G_N*m_BH/r
    
#Final BH mass after final adiabatic re-growth phase...
#m_BH_final = 1e6


#%%
plt.figure()
plt.loglog(r_array, psi_NFW(r_array), c = rgb_palette_dict['dark sienna'], label = 'initial NFW')
plt.loglog(r_array, psi_gas_i(r_array), c = rgb_palette_dict['pine green'], label = 'initial gas cloud')
plt.loglog(r_array, psi_gas_f(r_array), c = rgb_palette_dict['turquiose'], label = 'collapsed gas cloud')
plt.loglog(r_array, psi_SMS_bloated(r_array), c = rgb_palette_dict['flickr pink'], label = r'$10^5 M_\odot$ SMS')
plt.loglog(r_array, psi_proto_poly(r_array), c = 'C0', label = r'$10^5 M_\odot$ Polytrope')
plt.loglog(r_array, psi_BH(r_array), c = 'C1', label = r'$10^5 M_\odot$ BH')
plt.xlim(1E-9, 1E3)
plt.legend()
plt.xlabel('$r$ (pc)')
plt.ylabel(r'$\psi$')
plt.savefig('./figures/density_NFW_gas_star_potentials.pdf')
#plt.show()



#r_sp = 0.122 * R_s * ( m_BH / ( rho_0_prime * R_s**3 ) )**(1./(3.-1))

gamma_PL = 1.
r_0   = R_s
#rho_0 = rho_0_prime/4.
rho_0 = 1.0*rho_0_prime
gamma_sp_GS = (9. - 2.*gamma_PL)/(4. - gamma_PL)

#alpha_gamma = 0.293 * (gamma_PL)**(4./9.)
alpha_gamma = 0.135

def g_GS(r, m_BH, k):
    r_S = 2*G_N*m_BH/c_light**2
    return np.clip((1. - 4.*r_S/r), 0, None)**k

def rho_GS(r, m_BH, k):
    r_sp_GS = alpha_gamma * r_0 * (m_BH / (rho_0 * r_0**3.))**(1./(3. - gamma_PL))
    rho_R = rho_0*(r_sp_GS/r_0)**(-gamma_PL)
    return rho_0_prime * (R_s/r) / (1 + r/R_s)**2 * np.exp(-r/R_vir) + g_GS(r, m_BH, k) * rho_R*(r_sp_GS/r)**(gamma_sp_GS)


rho_GS_array = rho_GS(r_array, m_BH, k=3.00)


#New functions which define the radial velocity in the Schwarzschild metric and the E, L cuts
def calc_vrsq(r, E, L, m_BH):
    vr_sq = 1 - 2*E/c_light**2 - (1 - 2*G_N*m_BH/(r*c_light**2))*(1 + L**2/(r**2*c_light**2))
    vr_sq *= c_light**2
    return vr_sq

def L_c(E, m_BH): 
    arg = 16*(G_N*m_BH/c_light**2)**2*(1 - 4*E/c_light**2)# - 2*E**2/c_light**4)
    arg = np.clip(arg, 0, None)
    return c_light*np.sqrt(arg)
    
def E_max_NR(r, m_BH):
    Rs = 2*G_N*m_BH/c_light**2
    return np.clip(0.5*c_light**2*(r - 2*Rs)*Rs/(r**2 + 2*r*Rs - 4*Rs**2), 0, None)
  

#-------------------------------------------------------------------------
#----------------- SUPERMASSIVE STAR FORMATION ---------------------------
#-------------------------------------------------------------------------
print("> Supermassive star formation...")

if ((FROM_SCRATCH) or (not os.path.isfile("f_grid.npy"))):
    print("> Calling Adiabatic Growth...")
    #BJK: Use refinement=15
    density_after_SMS, E_grid, L_grid, f_grid = density_NFW.adiabatic_growth(psi_proto_poly, refinement = 15, figures = False, return_DF=True)
    #######################################################################################################################
    #######################################################################################################################
    #######################################################################################################################

    rho_array = density_after_SMS(r_array)

    np.save("r_array.npy", r_array)
    np.save("rho_array.npy", rho_array)
    np.save("E_grid.npy", E_grid)
    np.save("L_grid.npy", L_grid)
    np.save("f_grid.npy", f_grid)
    
    _psi = psi_NFW(r_array) + psi_proto_poly(r_array)
    
    T_orb_grid = 0.0*f_grid
    for i, E in enumerate(tqdm(E_grid[:,0])):
        for j, L in enumerate(L_grid[0,:]):
            vr_sq_grid = 2*_psi - 2*E - L**2/r_array**2
            inds = vr_sq_grid > 0
            
            if ((np.sum(inds) > 0) and (np.sum(inds) < 100)):
                _r = np.geomspace(np.min(r_array[inds]), np.max(r_array[inds]), 250)
                _psi_new = psi_NFW(_r) + psi_proto_poly(_r)
                vr_sq_grid = 2*_psi_new - 2*E - L**2/_r**2
                inds = vr_sq_grid > 0
            
                integ = 0.0*vr_sq_grid
                integ[inds] = 1/np.sqrt(vr_sq_grid[inds])
                T_orb_grid[i,j] = np.trapz(integ, _r, axis=-1)
            else:
                integ = 0.0*vr_sq_grid
                integ[inds] = 1/np.sqrt(vr_sq_grid[inds])
                T_orb_grid[i,j] = np.trapz(integ, r_array, axis=-1)

    np.save("T_orb_grid.npy", T_orb_grid)

else:
    print("> Loading grids from file...")
    r_array   = np.load("r_array.npy")
    rho_array = np.load("rho_array.npy")
    E_grid    = np.load("E_grid.npy")
    L_grid    = np.load("L_grid.npy")
    f_grid    = np.load("f_grid.npy")
    
    T_orb_grid    = np.load("T_orb_grid.npy")

#print(f_grid.flags)

f_grid[np.isnan(f_grid)] = 0.0



#Sampling E, L points
#--------------------
_psi = psi_NFW(r_array) + psi_proto_poly(r_array)


E_min = np.min(E_grid[:,0])
E_max = np.max(E_grid[:,0])

L_max = np.max(L_grid[0,:])
L_min = 1e-5*2*c_light*r_S_final

deltaE = E_max - E_min
deltalogL = np.log(L_max) - np.log(L_min)

E_samps = lin_sampler(E_min, E_max, N_particles)
p_samps = 0.0*E_samps + 1/deltaE

L_samps = log_sampler(L_min, L_max, N_particles)
p_samps *= 1/(L_samps*deltalogL)


points = (E_grid[:,0], L_grid[0,:])
#new_points = np.array((E_samps, L_samps)).T
new_points = (E_samps, L_samps)

f_E_L = interpn(points,  (4*np.pi)**2*(L_grid*T_orb_grid)*f_grid, new_points, bounds_error=False, fill_value=0.0)
weights = (1/p_samps)*f_E_L

Vol = 1.0

print("Total:", Vol*np.sum(weights)/N_particles/M_halo)

_psi = psi_NFW(r_array) + psi_proto_poly(r_array)

# Reconstructing Density Profile
#-------------------------------

T_orb_samps = 0.0*E_samps
P_r_A = 0.0*r_array

for i in tqdm(range(N_particles),desc="Calculating density profile"):
    if (weights[i] > 0):
        E = E_samps[i]
        L = L_samps[i]
        vr_sq_grid = 2*_psi - 2*E - L**2/r_array**2
        inds = vr_sq_grid > 0
        integ = 0.0*vr_sq_grid
        
        if ((np.sum(inds) > 0) and (np.sum(inds) < 100)):
            _r = np.geomspace(np.min(r_array[inds]), np.max(r_array[inds]), 250)
            _psi_new = psi_NFW(_r) + psi_proto_poly(_r)
            vr_sq_grid = 2*_psi_new - 2*E - L**2/_r**2
            inds = vr_sq_grid > 0
        
            integ = 0.0*vr_sq_grid
            integ[inds] = 1/np.sqrt(vr_sq_grid[inds])
            T_orb_samps[i] = np.trapz(integ, _r, axis=-1)
            
            integ2 = np.interp(r_array, _r, integ, left=0.0, right=0.0)
            
        else:
            integ = 0.0*vr_sq_grid
            integ[inds] = 1/np.sqrt(vr_sq_grid[inds])
            T_orb_samps[i] = np.trapz(integ, r_array, axis=-1)
            integ2 = 1.0*integ
        
        if (T_orb_samps[i] > 0):
            P_r_A += (Vol/N_particles)*weights[i]*integ2/T_orb_samps[i]
        
#P_r_A = (Vol/N_particles)*np.sum(weights[:,np.newaxis]*integ/T_orb_samps[:,np.newaxis], axis=0)

rho_r_A = P_r_A/(4*np.pi*r_array**2)


np.savetxt("results/rho_SMS.txt", rho_r_A)

plt.figure()

plt.loglog(r_array, rho_initial, c = rgb_palette_dict['dark sienna'], label = 'NFW')
#plt.loglog(r_array, rho_GS_no_r_S, c = rgb_palette_dict['amber'], label = r'GS, $m_{BH} = 10^5 M\odot$') #density_GS_pred
# plt.loglog(r_array, rho_after_proto, c = rgb_palette_dict['purple pizzazz'], label = 'after isothermal collapse')
plt.loglog(r_array, rho_array, c = rgb_palette_dict['flickr pink'], label = 'After SMS formation')
plt.loglog(r_array, rho_GS_array, c = rgb_palette_dict['amber'], label = 'GS profile')
#plt.loglog(r_array, rho_r,c='C1', linestyle=':')
plt.loglog(r_array, rho_r_A,c='C2', linestyle='--', label="After SMS formation (sampled)")

plt.ylim(1E0, 1E25)
plt.xlim(1E-9, 1E3)
plt.legend()
plt.ylabel(r'$\rho$ [$M_\odot \,\mathrm{pc}^{-3}$]')
plt.xlabel(r'$r$ [pc]')
plt.savefig('./figures/density_check.pdf')

#assert 1 == 0

#-------------------------------------------------------------------------
#----------------- NON-ADIABATIC GROWTH ----------------------------------
#-------------------------------------------------------------------------
print("> Non-adiabatic growth...")

if ((FROM_SCRATCH_NA) or (not os.path.isfile("E_f_samps.npy"))):

    #Sampling r-positions
    #--------------------

    r_samps = 0.0*E_samps

    psi_fun = lambda x: psi_NFW(x) + psi_proto_poly(x)
    for i in tqdm(range(N_particles), desc="Sampling orbits and performing non-adiabatic growth"):
        found = False
        if (weights[i] < 1e-20):
            r_samps[i] = 1.0
            continue
    
        _Nr = 1000
        while not found:
            _rs = log_sampler(np.min(r_array), np.max(r_array), _Nr)
            vr_sq = 2*psi_fun(_rs) - 2*E_samps[i] - L_samps[i]**2/_rs**2
            inds = vr_sq > 0
            #print(inds)
            if np.sum(inds) > 0:
                found = True
            else:
                _Nr *= 10
            #print(weights[i])
            if (_Nr > 10000):
                r_samps[i] = 1.0
                weights[i] = 0.0
                found = True
            #assert(_Nr < 100000)
    
        #assert(np.sum(inds) > 0)

        if (np.sum(inds) == 0):
            continue

        i1 = np.random.choice(np.arange(len(_rs))[inds])
        r_samps[i] = _rs[i1]
        if (T_orb_samps[i] > 0):
            weights[i] *= (_rs[i1]/np.sqrt(vr_sq[i1]))/T_orb_samps[i]
        else:
            weights[i] = 0

    #Instantaneous BH formation
    #--------------------------

    E_f_samps = E_samps - psi_proto_poly(r_samps) + psi_BH(r_samps)

    np.save("E_f_samps.npy", E_f_samps)
    np.save("L_samps.npy", L_samps)
    np.save("weights.npy", weights)

else:
    print("> Loading samples from file...")
    E_f_samps   = np.load("E_f_samps.npy")
    L_samps = np.load("L_samps.npy")
    weights   = np.load("weights.npy")


_psi = psi_NFW(r_array) + psi_BH(r_array)

T_orb_samps_B = 0.0*E_samps
P_r_B = 0.0*r_array
P_r_C = 0.0*r_array

#Reconstructing density profile
#------------------------------

def calc_radial_weights(r, E, L, m_BH, energy_cut = False):
    vr_sq_grid = calc_vrsq(r, E, L, m_BH)
    if (energy_cut):
        vr_sq_grid *= (E < E_max_NR(r, m_BH))
        
    inds = vr_sq_grid > 0
    integ = 0.0*vr_sq_grid
    
    if ((np.sum(inds) > 0) and (np.sum(inds) < 100)):
        _r = np.geomspace(np.min(r[inds]), np.max(r[inds]), 250)

        vr_sq_grid = calc_vrsq(_r, E, L, m_BH)
        if (energy_cut):
            vr_sq_grid *= (E < E_max_NR(_r, m_BH))
        
        inds = vr_sq_grid > 0
    
        integ = 0.0*vr_sq_grid
        integ[inds] = 1/np.sqrt(vr_sq_grid[inds])
        T_orb = np.trapz(integ, _r, axis=-1)
        
        integ2 = np.interp(r, _r, integ, left=0.0, right=0.0)
        
    elif (np.sum(inds) == 0):
        T_orb = 1.0
        integ2 = 0.0*vr_sq_grid
        
    else:
        integ = 0.0*vr_sq_grid
        integ[inds] = 1/np.sqrt(vr_sq_grid[inds])
        T_orb = np.trapz(integ, r, axis=-1)
        integ2 = 1.0*integ
    
    if (T_orb <= 0):
        return 0.0*integ2
    else:
        return integ2/T_orb

for i in tqdm(range(N_particles),desc="Calculating density profile"):
    #_psi = psi_NFW(_r1) + psi_SMS_bloated(_r1)
    if (weights[i] > 0):
        E = E_f_samps[i]
        L = L_samps[i]
        
        #BJK - this seems to be zero everywhere!
        if (L < L_c(E, m_BH)):
            L_cut = 0
        else:
            L_cut = 1
    
        p_r     = calc_radial_weights(r_array, E, L, m_BH)
        p_r_cut = calc_radial_weights(r_array, E, L, m_BH, energy_cut = True)
    
        P_r_B += (Vol/N_particles)*weights[i]*p_r
        P_r_C += (Vol/N_particles)*weights[i]*p_r_cut*L_cut

rho_r_B = P_r_B/(4*np.pi*r_array**2)
rho_r_C = P_r_C/(4*np.pi*r_array**2)

rho_r_C[r_array < 4*G_N*m_BH/c_light**2] = 0.0*rho_r_C[r_array < 4*G_N*m_BH/c_light**2] 

np.savetxt("results/rho_DCBH.txt", np.c_[rho_r_B, rho_r_C])


#-------------------------------------------------------------------------
#----------------- ADIABATIC RE-GROWTH ----------------------------------
#-------------------------------------------------------------------------
print("Performing adiabatic regrowth...")

#Reconstruct enclosed mass:
Menc_DM_arr = cumulative_trapezoid(P_r_C, r_array, initial=0)

psi_DM_arr = cumulative_trapezoid(G_N*Menc_DM_arr/r_array**2, r_array, initial = 0.0)
psi_DM_arr = psi_DM_arr[-1] - psi_DM_arr


psi_func = interp1d(r_array, psi_DM_arr + G_N*m_BH/r_array, bounds_error=False, fill_value=0.0)

def adiabatic_regrowth(m_BH_final):
    r_S_final = 2*G_N*m_BH_final/c_light**2
    
    N_particles = len(E_f_samps)
    E_f2_samps = 0.0*E_f_samps
    #Here, it doesn't matter that we're using `density_NFW`, we just need to call the function `calc_final_energy` which lives inside that class
    for i in tqdm(range(N_particles), desc="Calculating final energies"):
        E_f2_samps[i] = density_NFW.calc_final_energy(E_f_samps[i], L_samps[i], psi_func, m_BH_final)

    _psi = G_N*m_BH_final/r_array

    T_orb_samps_C = 0.0*E_samps
    P_r_D = 0.0*r_array
    P_r_E = 0.0*r_array

    #Reconstructing density profile
    #------------------------------

    for i in tqdm(range(N_particles),desc="Calculating final density profile"):
        #_psi = psi_NFW(_r1) + psi_SMS_bloated(_r1)
        if (weights[i] > 0):
            E = E_f2_samps[i]
            L = L_samps[i]
            #vr_sq_grid = 2*_psi - 2*E - L**2/r_array**2
        
            if (L < L_c(E, m_BH_final)):
                L_cut = 0
            else:
                L_cut = 1
    
            p_r     = calc_radial_weights(r_array, E, L, m_BH_final)
            p_r_cut = calc_radial_weights(r_array, E, L, m_BH_final, energy_cut = True)
    
            P_r_D += (Vol/N_particles)*weights[i]*p_r
            P_r_E += (Vol/N_particles)*weights[i]*p_r_cut*L_cut

    #D corresponds to the final density profile, ignoring capture by the central BH
    rho_r_D = P_r_D/(4*np.pi*r_array**2)
    #E corresponds to the final density profile, excluding captured orbits
    rho_r_E = P_r_E/(4*np.pi*r_array**2)
    rho_r_E[r_array < 4*G_N*m_BH_final/c_light**2] = 0.0*rho_r_E[r_array < 4*G_N*m_BH_final/c_light**2] 
    
    return rho_r_D, rho_r_E

growth_factor = [2, 5, 10, 3]
for g in growth_factor:
    
    rho_r_D, rho_r_E = adiabatic_regrowth(m_BH*g)
    np.savetxt(f"results/rho_regrowth_{g}.txt", np.c_[rho_r_D, rho_r_E])


plt.figure()

plt.loglog(r_array, rho_initial, c = rgb_palette_dict['dark sienna'], label = 'NFW')
#plt.loglog(r_array, rho_GS_no_r_S, c = rgb_palette_dict['amber'], label = r'GS, $m_{BH} = 10^5 M\odot$') #density_GS_pred
# plt.loglog(r_array, rho_after_proto, c = rgb_palette_dict['purple pizzazz'], label = 'after isothermal collapse')
plt.loglog(r_array, rho_array, c = rgb_palette_dict['flickr pink'], label = 'After SMS formation')
plt.loglog(r_array, rho_GS_array, c = rgb_palette_dict['amber'], label = 'GS profile')
#plt.loglog(r_array, rho_r,c='C1', linestyle=':')
plt.loglog(r_array, rho_r_A,c='C2', linestyle='--', label="After SMS formation (sampled)")
plt.loglog(r_array, rho_r_B,c='C3', linestyle=':',lw=2, label="After DCBH formation (all)")
plt.loglog(r_array, rho_r_C,c='C4', linestyle=':',lw=2, label="After DCBH formation (uncaptured)")
plt.loglog(r_array, rho_r_D,c='C5', linestyle=':',lw=2, label="After regrowth (all)")
plt.loglog(r_array, rho_r_E,c='C6', linestyle=':',lw=2, label="After regrowth (uncaptured)")
plt.axvline(r_core, c = rgb_palette_dict['turquiose'], ls = '--')
plt.axvline(2*r_S, c = 'k', ls = '--')
plt.axvline(3*r_S, c = 'k', ls = '--')
plt.axvline(4*r_S, c = 'k', ls = '--')
plt.ylim(1E0, 1E25)
plt.xlim(1E-9, 1E3)
plt.legend()
plt.ylabel(r'$\rho$ [$M_\odot \,\mathrm{pc}^{-3}$]')
plt.xlabel(r'$r$ [pc]')
plt.savefig('./figures/density_NFW_rho_SMS_v2.pdf')


plt.figure(figsize=(6,6))
#fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9))
plt.loglog(r_array, rho_GS_array, c = rgb_palette_dict['amber'], label = 'GS Profile')
plt.loglog(r_array, rho_array, c = rgb_palette_dict['flickr pink'], label = 'After SMS formation')
plt.loglog(r_array, rho_r_C, c=rgb_palette_dict['dark goldenrod'], linestyle='-',lw=2, label="After DCBH formation")

plt.axvline(2*r_S, c = 'k', ls = '--')
plt.axvline(r_poly, c = 'k', ls = '--')
plt.text(1.4*r_S, 8e19, r"$2 r_\mathrm{s}$", rotation=90)
plt.text(0.7*r_poly, 8e19, r"$r_\mathrm{SMS}$", rotation=90)
plt.ylim(1E14, 1E21)
plt.xlim(1E-8, 1E-5)
plt.legend(loc='best')
plt.ylabel(r'$\rho_\mathrm{DM}$ [$M_\odot\, \mathrm{pc}^{-3}$]')
plt.xlabel(r'$r$ [pc]')
plt.savefig('./figures/density_NFW_rho_SMS_zoom_v3.pdf')


#Calculate the GS profile of the final BH
rho_GS2_array = rho_GS(r_array, 3e5, k = 3.00)



plt.figure(figsize=(5.5,5.5))

#plt.loglog(r_array, rho_initial, c = rgb_palette_dict['dark sienna'], label = 'Initial NFW')

#plt.loglog(r_array, rho_GS_no_r_S, c = rgb_palette_dict['amber'], label = r'GS, $m_{BH} = 10^5 M\odot$') #density_GS_pred
# plt.loglog(r_array, rho_after_proto, c = rgb_palette_dict['purple pizzazz'], label = 'after isothermal collapse')
#plt.loglog(r_array, rho_array, c = rgb_palette_dict['flickr pink'], label = 'After SMS formation')
#plt.loglog(r_array, rho_r,c='C1', linestyle=':')
#plt.loglog(r_array, rho_r_A,c='C2', linestyle='--', label="After SMS formation (sampled)")
#plt.loglog(r_array, rho_r_B,c='C3', linestyle=':',lw=2, label="After DCBH formation (all)")
plt.loglog(r_array, rho_r_C, c=rgb_palette_dict['dark goldenrod'], linestyle='-',lw=2, label=r"After DCBH formation ($m_\mathrm{BH} = 10^5\,M_\odot$)")
plt.loglog(r_array, rho_r_E, c='midnightblue', linestyle='-',lw=2, label=r"After BH growth ($m_\mathrm{BH} = 10^6\,M_\odot$)")
plt.loglog(r_array, rho_GS2_array, c = 'royalblue', label = r'GS Profile ($m_\mathrm{BH} = 10^6\,M_\odot$)', linestyle='--')
#plt.axvline(r_core, c = rgb_palette_dict['turquiose'], ls = '--')
#plt.axvline(2*r_S, c = 'k', ls = '--')
#plt.axvline(r_poly, c = 'k', ls = '--')

#plt.text(1.4*r_S, 8e19, r"$2 r_\mathrm{s}$", rotation=90)
#plt.text(0.7*r_poly, 8e19, r"$r_\mathrm{SMS}$", rotation=90)
#plt.axvline(3*r_S, c = 'k', ls = '--')
#plt.axvline(4*r_S, c = 'k', ls = '--')

plt.ylim(1E14, 1E21)
plt.xlim(1E-8, 1E-4)
plt.legend()
plt.ylabel(r'$\rho_\mathrm{DM}$ [$M_\odot \, \mathrm{pc}^{-3}$]')
plt.xlabel(r'$r$ [pc]')
plt.savefig('./figures/density_final_v1.pdf')
plt.show()


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9))
ax1.loglog(r_array, rho_GS_array, c = rgb_palette_dict['amber'], label = 'GS Profile')
ax1.loglog(r_array, rho_array, c = rgb_palette_dict['flickr pink'], label = 'After SMS formation')
ax1.loglog(r_array, rho_r_C, c=rgb_palette_dict['dark goldenrod'], linestyle='-',lw=2, label="After DCBH formation")
ax1.axvline(2*r_S, c = 'k', ls = '--')
ax1.axvline(r_poly, c = 'k', ls = '--')
ax1.text(1.4*r_S, 8e19, r"$2 r_\mathrm{s}$", rotation=90)
ax1.text(0.7*r_poly, 8e19, r"$r_\mathrm{SMS}$", rotation=90)
ax1.set_ylim(1E14, 1E21)
ax1.set_xlim(1E-8, 1E-5)
ax1.legend()
ax1.set_ylabel(r'$\rho_\mathrm{DM}$ [$M_\odot$pc$^{-3}$]')
ax1.set_xlabel(r'$r$ [pc]')
slope_1 = np.gradient(np.log(rho_array), np.log(r_array))
slope_2 = np.gradient(np.log(rho_r_C), np.log(r_array))
ax2.plot(r_array, np.log(slope_1), c = rgb_palette_dict['flickr pink'])
ax2.plot(r_array, np.log(slope_2), c = rgb_palette_dict['dark goldenrod'], linestyle='-',lw=2)
ax2.set_ylabel("Log Slope")
ax2.set_xlim(1E-8, 1E-5)
plt.tight_layout()
plt.savefig('./figures/density_NFW_rho_SMS_zoom_2.pdf')
plt.show()

