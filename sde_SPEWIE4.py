import inspect
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import exponential, lognormal, normal, uniform
from scipy.stats import lomax

import random


SEED = 433322
np.random.seed(SEED)

rng = np.random.default_rng(SEED)


# If you are using scipy.stats, they rely on the numpy seed by default, 
# but it's safer to be explicit:

from scipy import stats
from scipy.stats import poisson
import os, h5py
from functools import partial
from numba import njit
import time
from astropy.cosmology import Planck18 as cosmo

################################################################################

#c *= 1e2 # in cm/s
keV_to_erg = 1.60218e-9 # in erg/keV

################################################################################
# Set the paths
################################################################################

BASE_DIR = '/astrodata/romain/sde_GA/rest_frame/'

################################################################################
# Spectral indices
################################################################################

# Alpha
alpha_values_path = BASE_DIR + "spectra/alphas.txt"
alpha_values = np.loadtxt(alpha_values_path)

# Beta
beta_values_path = BASE_DIR + "spectra/betas.txt"
beta_values = np.loadtxt(beta_values_path)

################################################################################
# Conversion factors
################################################################################

# CGRO/BATSE
k_values_batse_path = BASE_DIR + "k_values/log10_fluence_over_counts_CGRO_BATSE.txt"
k_values_batse = np.loadtxt(k_values_batse_path)

# Swift/BAT
k_values_bat_path = BASE_DIR + "k_values/log10_fluence_over_counts_Swift_BAT.txt"
k_values_bat = np.loadtxt(k_values_bat_path)

# Fermi/GBM
k_values_gbm_path = BASE_DIR + "k_values/log10_fluence_over_counts_Fermi_GBM.txt"
k_values_gbm = np.loadtxt(k_values_gbm_path)

###################################################################################
###################################################################################


def sample_broken_power_law(n_samples, x_min, x_break, x_max, alpha1, alpha2):
    """
    Samples from a broken power law distribution.
    Assumes alpha1 != 1 and alpha2 != 1.
    """

    # 1. Calculate unnormalized weights of both segments to ensure continuity
    w1 = (x_break**(1 - alpha1) - x_min**(1 - alpha1)) / (1 - alpha1)
    w2 = (x_break**(1 - alpha1)) * ((x_max / x_break)**(1 - alpha2) - 1) / (1 - alpha2)
    
    # Probability of falling into the first segment
    p1 = w1 / (w1 + w2)
    
    # 2. Determine which segment each sample belongs to
    u1 = np.random.uniform(0, 1, n_samples)
    samples = np.zeros(n_samples)
    
    mask1 = u1 < p1
    n1 = np.sum(mask1)
    n2 = n_samples - n1
    
    # 3. Sample within Segment 1 (x_min to x_break)
    u2_1 = np.random.uniform(0, 1, n1)
    samples[mask1] = (x_min**(1 - alpha1) + u2_1 * (x_break**(1 - alpha1) - x_min**(1 - alpha1)))**(1 / (1 - alpha1))
    
    # 4. Sample within Segment 2 (x_break to x_max)
    u2_2 = np.random.uniform(0, 1, n2)
    samples[~mask1] = (x_break**(1 - alpha2) + u2_2 * (x_max**(1 - alpha2) - x_break**(1 - alpha2)))**(1 / (1 - alpha2))
    
    return samples

#################################################################################
#################################################################################

def amati_relation(L_iso):
    """Amati correlation (E_pi - L_iso):

        log(E_pi/E_pivot) = K + S * log(L_iso / L_pivot)

    Parameter values are taken from Ghirlanda et al. (2022).

    Args:
        L_iso (float): Isotropic-equivalent peak luminosity.

    Returns:
        float: Intrinsic peak energy.
    """

    L_pivot = 3.38e52 # pivot isotropic-equivalent peak luminosity in erg/s
    E_pivot = 406.7 # pivot intrinsic peak energy in keV

    S = 0.56
    K = 0.27

    sigma = 0.18 # intrinsic dispersion of the correlation
    sigma_drawn = rng.normal(0.0, sigma)

    E_pi = E_pivot * (10**(K + sigma_drawn) * (L_iso / L_pivot)**S)

    return E_pi

################################################################################

def band_function(E, E_p, alpha, beta, A=1):
    """Band GRB function (Band 1993).

    Args:
        E (float): Energy.
        E_p (float): Peak energy.
        alpha (float): Low-energy power-law index.
        beta (float): High-energy power-law index.
        A (float): Normalization constant. Default value 1.

    Returns:
        float: Photon spectrum (photons/s/cm^2).
    """
    E = np.asarray(E)
    spectrum = np.zeros_like(E, dtype=float)

    E_pivot = 100 # keV
    E_break = E_p / (alpha + 2)
    delta = alpha - beta
    E_char = delta * E_break

    # Masks
    low_mask = E < E_char
    high_mask = E >= E_char

    # Low-energy branch
    spectrum[low_mask] = A * (E[low_mask] / E_pivot)**alpha * np.exp(-E[low_mask] / E_break)

    # High-energy branch
    spectrum[high_mask] = A * (E[high_mask] / E_pivot)**beta * np.exp(-delta) * (delta * E_break / E_pivot)**delta

    return spectrum

################################################################################

def luminosity_distance(z):
    """Calculate the luminosity distance at a given redshift.

    Args:
        z (float): Redshift.

    Returns:
        float: Luminosity distance in cm.
    """
    d_L = cosmo.luminosity_distance(z).to("cm").value # in cm

    return d_L

################################################################################

def luminosity_function(L_iso, z=0):
    """Luminosity function of long GRBs (Ghirlanda et al. 2022).

    Args:
        L_iso (float): Isotropic-equivalent peak luminosity.
        z (float): Redshift. Default to 0.

    Returns:
        float: Luminosity function.
    """
    a_1 = 0.97
    a_2 = 2.21
    log_L_iso_0 = 52.02 # in erg/s
    delta = 0.64

    L_iso_b = 10**log_L_iso_0 * (1 + z)**delta

    L_iso = np.asarray(L_iso)
    phi = np.zeros_like(L_iso)

    low = L_iso <= L_iso_b
    high = L_iso > L_iso_b

    phi[low] = L_iso[low]**(-a_1)
    phi[high] = L_iso_b**(a_2 - a_1) * L_iso[high]**(-a_2)

    return phi

################################################################################


###############################################################################

def flux(L, d_L, k):
    """Convert the luminosity into flux.

    Args:
        L (float): Luminosity.
        d_L (float): Luminosity distance.
        k (float): Flux correction factor

    Returns:
        float: Flux.
    """
    flx = L / (4 * np.pi * d_L**2) * k # in erg/s/cm^2

    return flx

################################################################################


def formation_rate(z):
    """GRB formation rate (Salvaterra et al. 2007, Ghirlanda & Salvaterra 2022).

    Args:
        z (float): GRB redshift

    Returns:
        float: GRB formation rate (yr^-1) per unit redshift.
    """
    # Differential comoving volume element
    dV_dz = 4 * np.pi * cosmo.differential_comoving_volume(z).to("Gpc^3/sr").value # in Gpc^3/sr

    # Formation rate per comoving volume element
    rho_0 = 79 # in Gpc^-3 yr^-1
    p_1 = 3.33
    p_2 = 3.42
    p_3 = 6.21
    rho = rho_0 * (1 + z)**p_1 / (1 + ((1 + z) / p_2)**p_3)

    # Absolute formation rate
    rate = dV_dz * rho / (1 + z)

    return rate



###############################################################################

def flux_to_counts(F, bin_time, k_factors):
    """Convert flux to counts per bin.

    Args:
        F (float): Flux.
        bin_time (float): Bin-time.
        k_factors (float): Array of conversion factors.

    Returns:
        float: Counts per bin.
    """
    k = rng.choice(k_factors)

    cnt = 10**(-k) * F * bin_time # in cnt

    return cnt, k

################################################################################


def flux_to_photon_flux(F, alpha, beta, E_p, e_1, e_2):
    """Convert flux to photon flux in the selected instrument's energy band.

    Args:
        F (float): Flux in erg/s/cm^2.
        alpha (float): Low-energy spectral index.
        beta (float): High-energy spectral index.
        E_p (float): Peak energy in keV.
        e_1 (float): Left extreme of the instrumental energy passband.
        e_2 (float): Right extreme of the instrumental energy passband.

    Returns:
        float: Photon flux in photons/s/cm^2.
    """
    e_min = np.log10(e_1)
    e_max = np.log10(e_2)

    e_grid = np.linspace(e_min, e_max, num=10000)

    def photon_spectrum(logE):
        E = 10**logE
        return np.log(10) * E * band_function(E, E_p, alpha, beta)

    def energy_spectrum(logE):
        E = 10**logE
        return np.log(10) * E**2 * band_function(E, E_p, alpha, beta)

    photon_spec = photon_spectrum(e_grid)
    energy_spec = energy_spectrum(e_grid)

    a_photon = np.trapz(photon_spec, e_grid)
    a_energy = np.trapz(energy_spec, e_grid)
    if a_photon <= 0.0:
        return 0.0
    E_mean = a_energy / a_photon # in keV
    E_mean *= keV_to_erg # in erg
    P_ph = F / E_mean # in ph/s/cm^2

    return P_ph

################################################################################
################################################################################

def flux_correction_factor(lc, z, alpha_values, beta_values, e_1, e_2):
    """Calculate the flux correction factor corresponding to the selected
    instrument's energy band.

    Args:
        lc (float): GRB light curve in luminosity (erg/s).
        z (float): GRB redshift.
        alpha_values (float): Array of low-energy indices of the Band spectrum.
        beta_values (float): Array of high-energy indices of the Band spectrum.
        e_1 (float): Left extreme of the instrumental energy passband.
        e_2 (float): Right extreme of the instrumental energy passband.

    Returns:
        float: Flux correction factor.
    """
    min_instr = np.log10(e_1 * (1 + z)) # in keV
    max_instr = np.log10(e_2 * (1 + z)) # in keV

    min_comov = 0 # log10(1) in keV
    max_comov = 4 # log10(1e4) in keV

    e_grid_instr = np.linspace(min_instr, max_instr, num=10000)
    e_grid_comov = np.linspace(min_comov, max_comov, num=10000)

    lc = np.asarray(lc)
    L_iso = lc.max()
    E_pi = amati_relation(L_iso)

    while True:
        alpha = rng.choice(alpha_values)
        beta = rng.choice(beta_values)

        if alpha <= -2:
            continue

        if beta >= alpha:
            continue

        break

    def energy_spectrum(logE):
        E = 10**logE
        return np.log(10) * E**2 * band_function(E, E_pi, alpha, beta)

    instr_spec = energy_spectrum(e_grid_instr)
    comov_spec = energy_spectrum(e_grid_comov)

    a_instr = np.trapz(instr_spec, e_grid_instr)
    a_comov = np.trapz(comov_spec, e_grid_comov)
    if a_comov <= 0.0:
        return 0.0, E_pi, alpha, beta

    k_flx = a_instr / a_comov

    return k_flx, E_pi, alpha, beta


################################################################################

### Load the (gaussian) errors of the Swift GRBs
#path_swift_errs = '../lc_pulse_avalanche/'                                                # LB
#path_swift_errs = '/home/bazzanini/PYTHON/genetic/lc_pulse_avalanche/lc_pulse_avalanche/' # bach
#path_swift_errs = '/home/bazzanini/PYTHON/genetic3/lc_pulse_avalanche/'                    # gravity
path_swift_errs = '/astrodata/romain/sde_GA/new_swift_errs/'

bins_swift_errs = np.array([  0.1, 0.21544347, 0.46415888, 1., 2.15443469, 4.64158883, 10.])
dict_errs_swift = {}
for i in range(1, len(bins_swift_errs)+1):
    with open(path_swift_errs+'new_swift_errs_'+str(i)+'.txt', 'r') as f:
        dict_errs_swift[str(i)] = f.readlines()
for key in dict_errs_swift.keys():
    for i, line in enumerate(dict_errs_swift[key]):
        line       = line.rstrip(' \n')
        errs_split = list(map(float, line.split(' ')))
        dict_errs_swift[key][i] = errs_split


###############################################################################
###############################################################################

z_values_path = BASE_DIR + "redshifts/redshifts.txt"
z_values = np.loadtxt(z_values_path)

################################################################################
################################################################################


#==============================================================================#
# Define the class LC describing the light curve.                              #
#==============================================================================#

class LC(object):
    """
    A class to generate gamma-ray burst light curves (GRB LCs) using a stochastic
    differential equation
    :tau_i: timescale over which a random instability leads to change in Xt comparable with itself
    :tau_se: timescale of the stretched-exponential 
    :gamma: exponent of the stretched-exponential
    :xi: exponent of the power-law rise 
    :alpha1: Low value PL exponent of the normalization distribution
    :Lb: break luminosity of the normalization distribution
    :alpha2: High value PL exponent of the normalization distribution
    :t_min: GRB LC start time
    :t_max: GRB LC stop time
    :res: GRB LC time resolution (s) (i.e., bin time)
    :eff_area: effective area of instrument (cm2)
    :bg_level: background level rate per unit area of detector (cnt/cm2/s)
    :min_photon_rate: left  boundary of -3/2 log N - log S distribution (ph/cm2/s)
    :max_photon_rate: right boundary of -3/2 log N - log S distribution (ph/cm2/s)
    :sigma: signal above background level
    :with_bg: boolean flag for keeping or removing the background level at the 
              end of the generation
    :use_poisson: boolean flag for using the Poisson or the (rounded) 
                  exponential for sampling the number of initial pulses and child
    """
    
    def __init__(self,tau_i,tau_se,gamma,xi,alpha1,Lb,alpha2,E_pi=500,alpha_band=-1,beta_band=-2.3,e_1=15,e_2=150,
    t_min=1e-6, t_max=1000, res=0.064,eff_area= 1400, bg_level=7.142857, with_bg=False, use_poisson=True,
                 min_photon_rate=1.3, max_photon_rate=1300, sigma=5, instrument='swift', verbose=False,seed=None): #New parameters of the BPL count distrib
        
        self._seed = seed
        if seed is not None:
            np.random.seed(seed) # This locks np.random.gamma, poisson, etc.
        
        self._tau_i =tau_i
        self._tau_se = tau_se
        self._gamma = gamma
        self._xi = xi
        self._alpha1 = alpha1
        self._Lb = Lb
        self._alpha2 = alpha2

        # Spectral parameters
        self.E_pi = E_pi
        self.alpha_band = alpha_band
        self.beta_band = beta_band
        #
        #self._instr = instrument
        self._e_1 = e_1
        self._e_2 = e_2

        self._eff_area = eff_area 
        self._bg = bg_level * self._eff_area # cnt/s
        self._min_photon_rate = min_photon_rate  
        self._max_photon_rate = max_photon_rate 
        self._verbose = verbose
        self._res = res # s
        self._n = int(np.ceil((t_max - t_min)/self._res)) + 1 # time steps
        self._t_min = t_min # ms
        self._t_max = (self._n - 1) * self._res + self._t_min # ms
        self._times, self._step = np.linspace(self._t_min, self._t_max, self._n, retstep=True)
        # Arrays of COUNT RATES
        self._rates       = np.zeros(len(self._times))
        self._total_rates = np.zeros(len(self._times))
        
        # Other parameters
        self._lc_params   = list()
        
        self._sigma       = sigma
        self._with_bg     = with_bg
        self._use_poisson = use_poisson
        self._instrument  = instrument
        self._bin_time    = res
        # Conversion factors
        # CGRO/BATSE
        if self._instrument == "batse":
            self.k_values_path = k_values_batse_path
            self.k_values = k_values_batse
        # Swift/BAT
        elif self._instrument == "swift":
            self.k_values_path = k_values_bat_path
            self.k_values = k_values_bat
        # Fermi/GBM
        elif self._instrument == "fermi":
            self.k_values_path = k_values_gbm_path
            self.k_values = k_values_gbm
        else:
            raise ValueError("Instrument not found.")
        
        if self._verbose:
             print("Time resolution: ", self._step)
    #--------------------------------------------------------------------------#

    def generate_LC_from_sde(self):

        tau_i,tau_se,gamma,xi,alpha1,Lb,alpha2 = self._tau_i,self._tau_se,self._gamma,self._xi,self._alpha1,self._Lb,self._alpha2

        norm_A = sample_broken_power_law(1, 1e47, Lb, 1e56, alpha1, alpha2)[0]
        cpt_A = 0
        

        # GRB redshift
        self._z = rng.choice(z_values)
        comov_t = self._times/(1+self._z)

        deterministic_term = ((comov_t/10)**xi)*np.exp(-comov_t/(2*tau_i) - (comov_t / tau_se)**(gamma))
        dt=comov_t[1]-comov_t[0]
        dW = rng.normal(0,1,len(comov_t)) 
        beta = np.cumsum(dW) * np.sqrt((1.0/tau_i) * dt)
        # Bolometric light curve generated with SDE (erg/s in source rest frame)
        curve_shape  = deterministic_term * np.exp(beta)
        curve_shape /= curve_shape.max() # normalize the shape to 1
        
        self._luminosity_lc = norm_A * curve_shape
        self._max_raw_pc = self._luminosity_lc.max()

        self._peak_value = self._max_raw_pc

        if (self._max_raw_pc<1e-12):
            self.check=0
            return 0
        else:
            self.check=1

        # Isotropic-equivalent peak luminosity (comoving frame)
        L_iso = self._peak_value

        if L_iso <= 0.0:
            self.check_flag = False
            return 

        self.log_L_iso = np.log10(L_iso)

        # Isotropic-radiated gamma-ray energy (comoving frame)
        E_iso = np.trapz(self._luminosity_lc,comov_t)
        if E_iso <= 0.0:
            self.check_flag = False
            return 

        self.log_E_iso = np.log10(E_iso)

        # Luminosity distance
        d_L = luminosity_distance(self._z) # in cm
        self.log_d_L = np.log10(d_L)

        # Calculate the flux correction factor, intrisic peak energy,
        # low-energy index alpha, and high-energy index beta
        k_corr, E_pi, alpha_band, beta_band = flux_correction_factor(
            self._luminosity_lc,
            self._z,
            alpha_values,
            beta_values,
            self._e_1,
            self._e_2
        )

        if E_pi <= 0.0:
            self.check_flag = False
            return 

        self.k_corr = k_corr
        self.E_pi = E_pi
        self.alpha_band = alpha_band
        self.beta_band = beta_band

        # Flux (observer frame) in erg/s/cm^2
        self.flux_lc = flux(self._luminosity_lc, d_L, k_corr)


        # Fluence (observer frame) in erg/cm^2
        fluence = np.trapz(self.flux_lc, self._times)
        if fluence <= 0.0:
            self.check_flag = False
            return 

        self.log_fluence = np.log10(fluence)

        # Peak photon flux (observer frame)
        peak_ph_flux = flux_to_photon_flux(
            self.flux_lc.max(), # peak flux
            alpha_band,
            beta_band,
            self.E_pi / (1 + self._z), # peak energy (observer frame)
            self._e_1,
            self._e_2
        )
        if peak_ph_flux <= 0.0:
            self.check_flag = False
            return 

        self.log_peak_ph_flux = np.log10(peak_ph_flux)

        # Convert flux into counts (observer frame)
        cnt, self.k_conv = flux_to_counts(self.flux_lc, self._bin_time, self.k_values)
        if self._instrument == 'batse' or self._instrument == 'fermi':
            self._model           = cnt# model COUNTS 
            self._modelbkg        = self._model + (self._bg * self._res)# model COUNTS + constant bgk counts
            if self._max_raw_pc > 1e12:
                self._plot_lc=rng.normal(loc=self._modelbkg, scale=np.sqrt(self._modelbkg))
            else:
                self._plot_lc = rng.poisson(self._modelbkg).astype('float')   # total COUNTS (signal+noise) with Poisson

            self._plot_lc_with_bg = self._plot_lc  
            self._err_lc          = np.sqrt(self._plot_lc)
            if self._with_bg: # lc with background
                pass
            else: # background-subtracted lc
                self._plot_lc = self._plot_lc - (self._bg * self._res)   # total COUNTS (removed the constant bkg level)
        
        # For Swift, the variable `_plot_lc` contains the COUNTS RATES (and not the counts!)
        elif self._instrument == 'swift':
            self._model           = cnt      # model COUNTS 
            self._model_rate      = self._model / self._res  # model COUNT RATES
            self._modelbkg        = self._model              # bkg 0 in Swift
            self._modelbkg_rate   = self._model_rate         # bkg 0 in Swift
            #
            if np.max(self._model_rate)<bins_swift_errs[1]:
                errs_swift_list = dict_errs_swift['1']
            elif np.max(self._model_rate)<bins_swift_errs[2]: 
                errs_swift_list = dict_errs_swift['2']
            elif np.max(self._model_rate)<bins_swift_errs[3]: 
                errs_swift_list = dict_errs_swift['3']
            elif np.max(self._model_rate)<bins_swift_errs[4]: 
                errs_swift_list = dict_errs_swift['4']
            elif np.max(self._model_rate)<bins_swift_errs[5]: 
                errs_swift_list = dict_errs_swift['5']
            elif np.max(self._model_rate)<bins_swift_errs[6]: 
                errs_swift_list = dict_errs_swift['6']
            else: 
                errs_swift_list = dict_errs_swift['7']
            #
             ### new mode of computing errors on swift data (taken from Manuele's github)
            idx_err_to_apply = rng.integers(len(errs_swift_list))
            mean_err_to_apply, std_err_to_apply = errs_swift_list[idx_err_to_apply]
            self._err_lc = np.abs(rng.normal(mean_err_to_apply, std_err_to_apply, size=len(self._model_rate)))
            #######

            self._plot_lc         = rng.normal(loc=self._model_rate, scale=self._err_lc) # total COUNTS RATE (signal+noise) with Gauss
            self._plot_lc_with_bg = self._plot_lc  
        
        self._get_lc_properties()
        return self._lc_params
    #--------------------------------------------------------------------------#

    def _get_lc_properties(self):
        """
        Calculates T90 and T100 durations along with their start and stop times, 
        total number of counts per T100, mean, max, and background count rates.
        """       
        threshold = self._model.max() * 1e-4
        mask = self._model > threshold

        # Find first and last True values without scanning the whole array twice
        idx_start = np.argmax(mask)
        # Find last index by reversing the mask
        idx_stop = len(mask) - np.argmax(mask[::-1]) - 1
        
        self._max_snr   = ((self._plot_lc_with_bg/self._res - self._bg) * self._res / (self._bg * self._res)**0.5).max()
        self._aux_times = self._times[idx_start:idx_stop] # +1 in the index
        # as `self._aux_lc` lc, we use the 'model' one, which is just the sum of the single pulses
        self._aux_lc    = self._model[idx_start:idx_stop] / self._res  # count RATES
        self._t_start = self._times[idx_start]
        self._t_stop  = self._times[idx_stop]
        self._t100    = self._t_stop - self._t_start
        
        self._total_cnts = np.sum(self._aux_lc) * self._res
          
        try:

            cumulative_counts = np.cumsum(self._aux_lc) * self._res
            total = self._total_cnts
            
            # Find the 5% and 95% indices instantly via binary search in C
            idx_i = np.searchsorted(cumulative_counts, 0.05 * total)
            idx_f = np.searchsorted(cumulative_counts, 0.95 * total)
            
            self._t90_i = self._aux_times[idx_i]
            self._t90_f = self._aux_times[idx_f]
            self._t90 = self._t90_f - self._t90_i
            assert self._t90 > 0
 
        except:
            #print('Weird stuff happened...')
            self._t90      = self._t100
            self._t90_i    = self._t_start
            self._t90_f    = self._t_stop
            self._t90_cnts = self._total_cnts
    #--------------------------------------------------------------------------#


    def hdf5_lc_generation(self, outfile, overwrite=False, seed=SEED):
        """
        Generates a new avalanche and writes it to an hdf5 file
        
        :n_lcs: number of light curves we want to simulate
        :outfile: file name
        :overwrite: overwrite existing file
        :seed: random seed for the avalanche generation, int or list
        """
        
        if overwrite == False:
            assert os.path.isfile(outfile), 'ERROR: file already exists!'

        self._f = h5py.File(outfile, 'w')

        
        self._f.create_group('GRB_PARAMETERS')
        self._f['GRB_PARAMETERS'].attrs['PARAMETER_ORDER'] = '[K, t_start, t_rise, t_decay]'

        self._grb_counter = 1
            
        if isinstance(seed, list):
            for sd in seed:
                self.aux_hdf5(seed=sd)
                
        else:
            self.aux_hdf5(seed=seed)

        self._f.close()
        
    #--------------------------------------------------------------------------#
        
    def aux_hdf5(self, seed):
        norms, t_delays, taus, tau_rs, peak_value = self.generate_avalanche(seed=seed, return_array=True)
        n_pulses = norms.size

        grb_array = np.concatenate((
                    norms.reshape(n_pulses,1),
                    t_delays.reshape(n_pulses,1),
                    tau_rs.reshape(n_pulses,1),
                    taus.reshape(n_pulses,1)),
                    axis=1
                    )

        self._f.create_dataset(f'GRB_PARAMETERS/GRB_{self._grb_counter}', data=grb_array)
        self._f[f'GRB_PARAMETERS/GRB_{self._grb_counter}'].attrs['PEAK_VALUE'] = peak_value
        self._f[f'GRB_PARAMETERS/GRB_{self._grb_counter}'].attrs['N_PULSES']   = n_pulses
        self._grb_counter += 1


#==============================================================================#
# Define the class Restore_LC
#==============================================================================#
class Restored_LC(LC):
    """
    Class to restore an avalanche from yaml file
    
    :res: GRB LC time resolution
    """
    
    def __init__(self, par_list, res=0.256, t_min=-10, t_max=1000, sigma=5):
        
        super(Restored_LC, self).__init__(res=res, t_min=t_min, t_max=t_max, sigma=sigma)

        if not par_list:
            raise TypeError("Avalanche parameters should be given")
        elif not isinstance(par_list, list):
            raise TypeError("The avalanche parameters should be a list of dictionaries")
        else:
            self._lc_params = par_list
 
        self._restore_lc()
#==============================================================================#
#==============================================================================#