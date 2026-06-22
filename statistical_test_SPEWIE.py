###############################################################################
# IMPORT LIBRARIES
################################################################################
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from matplotlib import rc
from pathlib import Path
import datetime
import ctypes
from scipy.signal import savgol_filter
from scipy import signal
from scipy import stats
from scipy.stats import anderson_ksamp, ks_2samp
from scipy.special import expit
from tqdm import tqdm
import pandas as pd

sys.path.append('/astrodata/romain/')
from sax_utils import rebin_lc_genau
instrument = 'fermi'


df_peak_flx_gbm = pd.read_csv('/astrodata/romain/sde_GA/fermi_real_data/peak_flux_4th_gbm_catalogue.txt',sep='\s+')
log_fluence_obs_list = df_peak_flx_gbm.Fl.values

eps_log = 1e-15
if instrument == 'swift':
    times_tot = np.linspace(-150,150,4686)

elif instrument == 'fermi':
    times_tot = np.linspace(-150,150,4686)

def rebin_fact(snr):
    """
    Calculate the rebinning factor based on the S/N value.
    """
    if snr >19.02:
        return 1
    elif (snr < 19.02) and (snr > 9.09):
        return 2
    elif (snr > 6.28) and (snr <= 9.09):
        return 4
    elif (snr > 4.34) and (snr <= 6.28):
        return 7
    elif snr <= 4.34:
        return 16


def get_normalized_profile(grb_test,bin_time=0.064,npts_before=2343,npts_after=2343):
    x,y,z = grb_test.times, grb_test.counts, grb_test.errs


    x = grb_test.times
    y = grb_test.counts

    imax = np.argmax(y)
    xmax = x[imax]
    
    xini = xmax-200
    xfini = xmax + 200

    mask_norm = (x>xini) &  (x<xfini)
    x_norm = x[mask_norm]
    y_norm = y[mask_norm]

    # Initialize padding arrays to avoid UnboundLocalError
    x_pad_bef = np.array([], dtype=np.float64)
    y_pad_bef = np.array([], dtype=np.float64)
    x_pad_aft = np.array([], dtype=np.float64)
    y_pad_aft = np.array([], dtype=np.float64)

    if xini < x_norm[0]:
        # zero-pad the light curve before the peak
        n_points_to_pad = int( np.ceil((x_norm[0] - xini)/bin_time))
        x_pad_bef = np.linspace(xini, x_norm[0]-bin_time, n_points_to_pad)
        y_pad_bef = np.zeros_like(x_pad_bef)
    if xfini > x_norm[-1]:
        n_points_to_pad = int( np.ceil((xfini - x_norm[-1])/bin_time))
        x_pad_aft = np.linspace(x_norm[-1]+bin_time, xfini, n_points_to_pad)
        y_pad_aft = np.zeros_like(x_pad_aft)

    x_new = np.concatenate([x_pad_bef, x_norm, x_pad_aft])
    y_new = np.concatenate([y_pad_bef, y_norm, y_pad_aft])
    imax = np.argmax(y_new)
    x_centered = x_new[imax-npts_before:imax+npts_after]
    y_centered = y_new[imax-npts_before:imax+npts_after]
    y_centered /= np.max(y_centered)
    return x_centered, y_centered


def compute_avgd_profile(grb_list,npts_before=2343,npts_after=2343):
    avgd_profile = np.zeros(npts_before+npts_after)
    avgd_profile_square = np.zeros(npts_before+npts_after)
    avgd_profile_cube = np.zeros((npts_before+npts_after))
    N = len(grb_list)
    for i in range(N):
        grb_test = grb_list[i]
        x_centered, y_norm = get_normalized_profile(grb_test,npts_before=npts_before,npts_after=npts_after)
        avgd_profile += y_norm
        avgd_profile_square += y_norm**2
        avgd_profile_cube += y_norm**3
       
    avgd_profile /= N
    avgd_profile = savgol_filter(avgd_profile, window_length=21, polyorder=2)
    avgd_profile /= np.nanmax(avgd_profile)
    
    avgd_profile_square /= N
    avgd_profile_square = savgol_filter(avgd_profile_square, window_length=63, polyorder=2)
    avgd_profile_square /= np.nanmax(avgd_profile_square)


    avgd_profile_cube /= N
    avgd_profile_cube = savgol_filter(avgd_profile_cube, window_length=63, polyorder=3)
    avgd_profile_cube /= np.nanmax(avgd_profile_cube)

    return avgd_profile, avgd_profile_square, avgd_profile_cube

#user = 'external user'
#user='LB'
#user='AF'
#user='bach'
#user='gravity'
#user = 'MM'
user='romano'

if user=='romano':
    print('User = romano')
    #sys.path.append('/astrodata/romain/sde_GA/geneticgrbs_v2/lc_pulse_avalanche')

else:
    raise ValueError('Assign to the variable "user" a correct username!')
    
from sde_SPEWIE4 import LC


test_times = np.linspace(0, 150, int(150/0.064))

# write a function that can find if the peak is before 0.128 s:

@njit(fastmath=True)
def adaptive_bin_curve(x, y, min_bin_points=5, max_bin_width=0.5, signal_threshold=1e-2):
    # Convert to contiguous arrays for speed
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Sort arrays by x
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    N = len(x)
    # Preallocate with a conservative estimate
    x_bins = np.empty(N)
    y_bins = np.empty(N)
    counts = np.empty(N, dtype=np.int64)
    edges_left = np.empty(N)
    edges_right = np.empty(N)

    # Precompute cumulative sums for fast means
    cumsum_y = np.empty(N + 1)
    cumsum_x = np.empty(N + 1)
    cumsum_y[0] = 0.0
    cumsum_x[0] = 0.0
    for i in range(N):
        cumsum_y[i + 1] = cumsum_y[i] + y[i]
        cumsum_x[i + 1] = cumsum_x[i] + x[i]

    i = 0
    n_bins = 0

    while i < N:
        j = i + min_bin_points
        if j > N:
            j = N
        while j < N:
            width = x[j - 1] - x[i]
            mean_y = (cumsum_y[j] - cumsum_y[i]) / (j - i)
            if (j - i) >= min_bin_points and (mean_y >= signal_threshold or width >= max_bin_width):
                break
            j += 1

        # Final bin if near the end
        if j >= N:
            j = N

        # Compute means using cumsums (avoid np.mean)
        mean_x = (cumsum_x[j] - cumsum_x[i]) / (j - i)
        mean_y = (cumsum_y[j] - cumsum_y[i]) / (j - i)

        x_bins[n_bins] = mean_x
        y_bins[n_bins] = mean_y
        counts[n_bins] = j - i
        edges_left[n_bins] = x[i]
        edges_right[n_bins] = x[j - 1]
        n_bins += 1
        i = j

    # Trim arrays to used bins
    return (
        x_bins[:n_bins],
        y_bins[:n_bins],
        counts[:n_bins],
        edges_left[:n_bins],
        edges_right[:n_bins]
    )


def rebin_with_given_edges_fast(x_sim, y_sim, bin_edges_left, bin_edges_right, stat='mean'):
    # Ensure inputs are numpy arrays
    x_sim = np.asarray(x_sim)
    y_sim = np.asarray(y_sim)
    
    # If arrays are empty, return early to avoid crash
    if x_sim.size == 0:
        midpoints = (bin_edges_left + bin_edges_right) / 2
        return midpoints, np.full(len(midpoints), np.nan), np.zeros(len(midpoints))

    if len(x_sim) != len(y_sim):
        raise ValueError("x_sim and y_sim must have the same length")
    # Define the bins
    # We use all the left edges and the very last right edge
    bins = np.append(bin_edges_left, bin_edges_right[-1])
    
    # Assign each x_sim to a bin index
    bin_indices = np.digitize(x_sim, bins)
    
    # CRITICAL FIX: bin_indices might include 0 (underflow) or len(bins) (overflow)
    # We only care about indices 1 to len(bin_edges_left)
    mask = (bin_indices > 0) & (bin_indices <= len(bin_edges_left))
    
    # Squeeze the data to only things that fall inside your edges
    x_masked = x_sim[mask]
    y_masked = y_sim[mask]
    b_masked = bin_indices[mask]
    
    # Now create the DataFrame with guaranteed matching lengths
    df = pd.DataFrame({
        'y': y_masked, 
        'x': x_masked, 
        'bin': b_masked
    })
    
    # Aggregate
    if stat == 'mean':
        group = df.groupby('bin').agg({'y': 'mean', 'x': 'mean', 'bin': 'count'})
    elif stat == 'median':
        group = df.groupby('bin').agg({'y': 'median', 'x': 'mean', 'bin': 'count'})
    
    # Reindex to catch the empty bins
    all_bins = np.arange(1, len(bin_edges_left) + 1)
    group = group.reindex(all_bins)
    
    # Fill in the empty bins with the requested midpoints
    midpoints = (bin_edges_left + bin_edges_right) / 2
    group['x'] = group['x'].fillna(pd.Series(midpoints, index=all_bins))
    
    return group['x'].values, group['y'].values, group['bin'].fillna(0).values

################################################################################

class GRB:
    """
    Class for GRBs to save their properties.
    """
    def __init__(self,
                 grb_name,
                 times,
                 counts,
                 errs,
                 t90,
                 t90_start = None,
                 t90_stop = None,
                 t20=-1,
                 s2n = -1,
                 log_E_iso=None,
                 log_L_iso=None,
                 z=None,
                 log_peak_ph_flux=None,
                 log_fluence=None,
                 k_corr= None,
                 k_conv = None,
                 alpha_band = None,
                 beta_band = None,
                 E_pi = None,
                 model = [],
                 modelbkg = [],
                 bg = [],
                 num_of_sig_pulses=-1,
                 grb_data_file_path='./',
                 minimum_peak_rate_list=[],
                 peak_rate_list=[],
                 current_delay_list=[],
                 minimum_pulse_delay_list=[]):

        self.name                     = grb_name
        self.times                    = times
        self.counts                   = counts
        self.errs                     = errs
        self.model                    = model
        self.modelbkg                 = modelbkg
        self.bg                       = bg
        self.t90                      = t90
        self.t90_start                   = t90_start
        self.t90_stop                    = t90_stop
        self.t20                      = t20
        self.s2n                      = s2n
        self.log_E_iso                = log_E_iso
        self.log_L_iso                = log_L_iso
        self.z                        = z
        self.log_peak_ph_flux         = log_peak_ph_flux
        self.log_fluence              = log_fluence
        self.k_corr                   = k_corr
        self.k_conv                   = k_conv
        self.alpha_band               = alpha_band
        self.beta_band                = beta_band
        self.E_pi                      = E_pi
        self.minimum_peak_rate_list   = minimum_peak_rate_list
        self.peak_rate_list           = peak_rate_list
        self.current_delay_list       = current_delay_list
        self.minimum_pulse_delay_list = minimum_pulse_delay_list
        self.grb_data_file_path       = grb_data_file_path
        self.num_of_sig_pulses        = num_of_sig_pulses

    def copy(self):
        copy_grb = GRB(grb_name=self.name, times=self.times, counts=self.counts, 
                       errs=self.errs, t90=self.t90, grb_data_file_path=self.grb_data_file_path, 
                       t20=self.t20, log_E_iso = self.log_E_iso,log_L_iso=self.log_L_iso,z=self.z,
                       log_peak_ph_flux=self.log_peak_ph_flux,
                       log_fluence=self.log_fluence,k_corr=self.k_corr,k_conv=self.k_conv,alpha_band=self.alpha_band,beta_band=self.beta_band,E_pi=self.E_pi,model=self.model,modelbkg=self.modelbkg,bg=self.bg,
                       num_of_sig_pulses=self.num_of_sig_pulses, 
                       minimum_peak_rate_list=self.minimum_peak_rate_list, 
                       peak_rate_list=self.peak_rate_list, 
                       current_delay_list=self.current_delay_list, 
                       minimum_pulse_delay_list=self.minimum_pulse_delay_list)
        return copy_grb

################################################################################
################################################################################

# Dictionaries where the properties of instruments are stored:
# - name         : name of the instrument
# - res          : time resolution of the instrument [s]
# - eff_area     : effective area of instrument [cm2]
# - bg_level     : background level [cnt/cm2/s]
# - t90_threshold: used to select only _long_ GRBs [s]
# - sn_threshold : used to select only lc with high S2N    
#------------------------------------------------------------------------------#
# BATSE (counts)
#------------------------------------------------------------------------------#
# - Effective area: 2025 cm^2 (https://heasarc.gsfc.nasa.gov/docs/cgro/nra/appendix_g.html#V.%20BATSE%20GUEST%20INVESTIGATOR%20PROGRAM)
# - Background: 2.8 (computed from the average of the background LC, see `statistical_test.ipynb`)
name_batse          = 'batse'
res_batse           = 0.064
eff_area_batse      = 2025
bg_level_batse      = 2.8 
t90_threshold_batse = 2
sn_threshold_batse  = 15
e_1_batse           = 25
e_2_batse           = 2000
N_accepted_batse    = 25
instr_batse         = {
    "name"         : name_batse,
    "res"          : res_batse,
    "eff_area"     : eff_area_batse,
    "bg_level"     : bg_level_batse,
    "t90_threshold": t90_threshold_batse,
    "sn_threshold" : sn_threshold_batse,
    "e_1"          : e_1_batse,
    "e_2"          : e_2_batse,
    "N_accepted"   : N_accepted_batse
}
#------------------------------------------------------------------------------#
# Swift (count rates)
#------------------------------------------------------------------------------#
# - Effective area:
#   https://swift.gsfc.nasa.gov/proposals/tech_appd/swiftta_v17/node27.html)
#   Approximately 1400 cm^2.
# - Background:
#   https://swift.gsfc.nasa.gov/proposals/tech_appd/swiftta_v17/node32.html
#   The typical BAT background event rate in the full array above threshold is 
#   about 10000 counts per second.
name_swift          = 'swift'
res_swift           = 0.064
eff_area_swift      = 1400
bg_level_swift      = (10000/eff_area_swift)

t90_threshold_swift = 2
sn_threshold_swift  = 10
e_1_swift = 15
e_2_swift = 150
N_accepted_swift = 15

instr_swift         = {
    "name"         : name_swift,
    "res"          : res_swift,
    "eff_area"     : eff_area_swift,
    "bg_level"     : bg_level_swift,
    "t90_threshold": t90_threshold_swift,
    "sn_threshold" : sn_threshold_swift,
    "e_1"          : e_1_swift,
    "e_2"          : e_2_swift,
    "N_accepted"   : N_accepted_swift
}

#------------------------------------------------------------------------------#
# Fermi (count rates)
#------------------------------------------------------------------------------#
name_fermi          = 'fermi'
res_fermi           = 0.064
eff_area_fermi      = 100 # effective area of a NaI detector at normal incidence 
bg_level_fermi      = 39.4 # see `statistical_test.ipynb` for the computation 
t90_threshold_fermi = 2
sn_threshold_fermi  = 15
e_1_fermi = 8.0
e_2_fermi = 1000.0
N_accepted_fermi = 25
instr_fermi         = {
    "name"         : name_fermi,
    "res"          : res_fermi,
    "eff_area"     : eff_area_fermi,
    "bg_level"     : bg_level_fermi,
    "t90_threshold": t90_threshold_fermi,
    "sn_threshold" : sn_threshold_fermi,
    "e_1"          : e_1_fermi,
    "e_2"          : e_2_fermi,
    "N_accepted"   : N_accepted_fermi
}


################################################################################
################################################################################
# Functions for evaluating GRB properties
################################################################################


def evaluateDuration20(times, counts, t90=None, t90_frac=15, bin_time=None, filter=True):
    """
    Compute the duration of the GRB event as described in [Stern et al., 1996].
    We define the starting time when the signal reaches the 20% of the value of
    the peak, and analogously for the ending time. The difference of those two
    times is taken as definition of the duration of the GRBs (T20%).
    If filter==True, we smooth the signal using a Savitzky-Golay filter on the
    light curves before computing the T20%.
    Inputs:
      - times: time values of the bins of the light-curve;
      - counts: counts per bin of the GRB;
      - t90: T90 duration of the GRB;
      - bin_time: temporal bin size of the instrument [s];
      - t90_frac: fraction of T90 to be used as window length;
      - filter: boolean variable. If True, it activates the smoothing savgol
                filter before computing the T20% duration;
    Output:
      - duration: T20%, that is, the duration at 20% level;
    """
    if filter:
        t90_frac = t90_frac
        window   = int(t90/t90_frac/bin_time)+2
        window   = window if window%2==1 else window+1

        try:
            counts = savgol_filter(x=counts,
                                   window_length=window,
                                   polyorder=2)
        except:
            #print('window_length =', window)
            print('Error in "evaluateDuration20()" during the "savgol_filter()"...')
            sys.exit()

    threshold_level = 0.20
    c_max           = np.max(counts)
    c_threshold     = c_max * threshold_level
    selected_times  = times[ np.where(counts>=c_threshold)[0] ]
    # check that selected_times is not empty
    if len(selected_times)==0:
        # return null duration if selected_times is empty
        return np.array( [0, 0, 0] )
    #selected_times = times[counts >= c_threshold]
    tstart          = selected_times[ 0]
    tstop           = selected_times[-1]
    duration        = tstop - tstart # T20%
    assert duration>0

    return duration, tstart, tstop

################################################################################

def evaluateGRB_SN(times, counts, errs, t90, t90_frac, bin_time, filter,bg_level=7.142857142857143, 
                   return_cnts=False):
    """
    Compute the S/N ratio between the total signal from a GRB and the background
    in a time interval equal to the GRB duration, as defined in Stern+96, i.e.,
    the time interval between the first and the last moments in which the signal
    reaches the 20% of the peak (T20%). The S2N ratio is defined in the 
    following way: we sum of the signal inside the time window defined by the 
    T20%, and we divide it by the square root of the squared sum of the errors
    in the same time interval.
    Input:
     - times: array of times;
     - counts: counts of the event;
     - errs: errors over the counts;
     - t90: T90 of the GRB;
     - bin_time: temporal bin size of the instrument [s];
     - filter: if True, apply savgol filter;
     - return_cnts: if True, return also the total counts inside the T20% interval;
    Output:
     - s2n: signal to noise ratio;
     - T20: duration of the GRB at 20% level;
     - T20_start: start time of the T20% interval;
     - T20_stop: stop time of the T20% interval; 
     - sum_grb_counts: total counts inside the T20% interval;
    """
    T20, tstart, tstop = evaluateDuration20(times=times, 
                                            counts=counts,
                                            t90=t90, 
                                            t90_frac=t90_frac, 
                                            bin_time=bin_time,
                                            filter=filter)
    
    i_start = np.searchsorted(times, tstart)
    i_stop = np.searchsorted(times, tstop, side='right')

    sum_grb_counts   = np.sum( counts[i_start:i_stop] )
    sum_errs         = np.sqrt( np.sum(errs[i_start:i_stop]**2) )
    sum_errs        = sum_errs if sum_errs>0 else 1e-10

    s2n              = np.abs( sum_grb_counts/sum_errs )
    
    if not(return_cnts):
        return s2n, T20, tstart, tstop
    else:
        return s2n, T20, tstart, tstop, sum_grb_counts


def evaluateGRB_SN_peak(counts, errs):
    """
    Compute the S/N ratio of the peak of the GRB.
    Input:
     - counts: counts of the event;
     - errs: errors over the counts;
    Output:
     - s2n: signal to noise ratio of the peak;
    """
    c_max    = np.max(counts)
    i_c_max  = np.argmax(counts)
    s_n_peak = c_max / errs[i_c_max]
    return s_n_peak

################################################################################

def load_lc_batse(path):
    """
    Load the BATSE light curves, and put each of them in an object inside
    a list. Since in the analysis we consider only the long GRBs, we load 
    only the light curves listed in the 'alltrig_long.list' file.
    N.B. BATSE LC are already in counts.
    Input:
    - path: path to the folder that contains a file for each BATSE GRB and the
            file containing all the T90s;
    Output:
    - grb_list_batse: list of GRB objects;
    """
    # load only the GRBs that are already classified as 'long'
    long_list_file     = 'alltrig_long.list'
    all_grb_list_batse = [grb_num.rstrip('\n') for grb_num in open(path+long_list_file).readlines()]
    # load T90s
    t90data = np.loadtxt(path+'T90_full.dat')
    # Bad GRBs
    GRBs_to_mask = {
        "01742": (   0, 250),
        "02344": (-100, 245),
        "02898": ( -50, 450),
        "05474": (-100, 200),
        "01924": ( -50, 100),
        "00753": (-100, 100),
        "02663": (-100, 100),
        "02863": (-100, 200),
        "02877": (-100, 250),
        "02922": ( -45, 400),
        "03084": (-100, 100),
        "03611": (-100, 100),
        "03637": (-100, 100),
        "05635": (-100, 100),
        "05867": (-100, 100),
        "06090": (-100, 100),
        "06096": (-100, 100),
        "06216": (-100, 100),
        "06273": (-100, 100),
        "06400": (-100, 100),
        "06422": ( -50, 100),
        "07219": (-100, 150),
        "07250": (-150, 100),
        "07404": (-100, 100),
        "07997": (-100, 100)
    }

    grb_list_batse = []
    grb_not_found  = []
    print("Loading BATSE data...")
    #print("Loading BATSE data (approx 90 s)...")
    for grb_name in all_grb_list_batse:
    #for grb_name in tqdm(all_grb_list_batse):
        try:
            times, counts, errs = np.loadtxt(path+grb_name+'_all_bs.out', unpack=True)
        except:
            # print(grb_name, ' not found!')
            grb_not_found.append(grb_name)
            continue
        
        if grb_name in GRBs_to_mask.keys():
            start      = GRBs_to_mask[grb_name][0]
            stop       = GRBs_to_mask[grb_name][1]
            times_mask = np.logical_and(times >= start, times <= stop)
            times      = np.float32(times[times_mask])
            counts     = np.float32(counts[times_mask])
            errs       = np.float32(errs[times_mask]) 
        else: 
            times  = np.float32(times)
            counts = np.float32(counts)
            errs   = np.float32(errs)
        t90    = t90data[t90data[:,0] == float(grb_name),1]
        t90    = np.float32(t90)
        grb    = GRB(grb_name=grb_name, times=times, counts=counts, errs=errs,
                     t90=t90, grb_data_file_path=path+grb_name+'_all_bs.out')
        grb_list_batse.append(grb)

    print("Total number of _long_ GRBs in BATSE catalogue: ", len(all_grb_list_batse))
    print("GRBs in the catalogue which are NOT present in the data folder: ", len(grb_not_found))
    print("Loaded GRBs: ", len(grb_list_batse))
    return grb_list_batse


def load_lc_swift(path):
    """
    Load the Swift light curves, and put each of them in an object inside
    a list. Since in the analysis we consider only the _long_ GRBs, we load 
    only the light curves listed in the 'merged_lien16-GCN_long_noshortEE_t90.dat'
    file.
    N.B. Swift LC are in counts/s, so we need to multiply them by the resolution
    to get the counts.
    Input:
    - path: path to the folder that contains a folder for each Swift GRB named
            with the name of the GRB, and the file containing all the T90s;
    Output:
    - grb_list_swift:  list of GRB objects;
    """

    # load only the GRBs that are already classified as 'long'
    long_list_file     = 'merged_lien16-GCN_long_noshortEE_t90.dat'
    all_grb_list_swift = []
    t90_dic            = {}
    with open(path+long_list_file) as f:
        for line in f:
            grb_name = line.split()[0]
            t90      = line.split()[1]
            all_grb_list_swift.append(grb_name)
            t90_dic[grb_name] = np.float32(t90)

    # bin time of Swift
    res = instr_swift['res']

    grb_list_swift = []
    grb_not_found  = []
    for grb_name in all_grb_list_swift:
        try:
            times, counts, errs = np.loadtxt(path+grb_name+'/'+'all_3col.out', unpack=True)
        except:
            # print(grb_name, ' not found!')
            grb_not_found.append(grb_name)
            continue
        t90    = t90_dic[grb_name]
        times  = np.float32(times)
        counts = np.float32(counts) * res # convert from counts/s to counts
        errs   = np.float32(errs)   * res # convert from counts/s to counts
        t90    = np.float32(t90)
        grb    = GRB(grb_name=grb_name, times=times, counts=counts, errs=errs, 
                     t90=t90, grb_data_file_path=path+grb_name+'/'+'all_3col.out')
        grb_list_swift.append(grb)

    print("Total number of GRBs in Swift catalogue: ", len(all_grb_list_swift))
    print("GRBs in the catalogue which are NOT present in the data folder: ", len(grb_not_found))
    print("Loaded GRBs: ", len(grb_list_swift))
    return grb_list_swift


def load_lc_fermi(path):
    # Load the data from von Kienlin catalogue
    path_vk_catalogue = '/astrodata/romain/sde_GA/geneticgrbs_v2/lc_pulse_avalanche/vk_catalog_list.txt'
    vk_grbs, vk_t90s  = np.loadtxt(path_vk_catalogue, dtype=str, unpack=True)
    vk_t90s           = vk_t90s.astype(float)
    
    # List the Fermi/GBM GRBs
    grb_dir_list = [ name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ]
    grb_dir_list.sort()

    # Bad GRBs (to be checked)
    grbs_to_mask = ['bn130206482',
                    'bn140603476',
                    'bn160720275',
                    'bn180610377']
    
    # Lists
    fermi_grb_list = []
    grb_with_no_bs = []

    for vk_grb, vk_t90 in zip(vk_grbs, vk_t90s):
        if (vk_grb in grb_dir_list) & (vk_grb not in grbs_to_mask):

            try:
                path_selected_units = path + vk_grb + '/LC/selected_units.txt'
                selected_units      = np.loadtxt(path_selected_units, dtype=str, ndmin=1)
                path_lc             = path + vk_grb + '/LC/' + vk_grb + '_LC_64ms_'
                for unit in selected_units:
                    path_lc += unit + '_'
                path_lc += 'bs.txt'
            except:
                grb_with_no_bs.append(vk_grb)
                continue
            
            try:
                times, counts, errs = np.loadtxt(path_lc, unpack=True)
            except:
                grb_with_no_bs.append(vk_grb)
                continue

            grb = GRB(grb_name=vk_grb, 
                      times=times,
                      counts=counts,
                      errs=errs,
                      t90=vk_t90,
                      grb_data_file_path=path_lc)
            fermi_grb_list.append(grb)

    print('GRBs in the von Kienlin catalogue: ', len(vk_grbs))
    print('GRBs without background-subtracted LC: ', len(grb_with_no_bs))
    print('Loaded GRBs: ', len(fermi_grb_list))

    return fermi_grb_list


def load_lc_sim(path):
    """
    Load the simulated light curves, which were previously generated and saved
    as files, named 'lcXXX.txt', one file for each simulated GRB ("XXX" is the 
    index of the GRB generated). The columns in the files are: 'times', 
    'counts', 'errs', 't90'. We put each light curve in a 'GRB' object inside
    a list. 
    Input:
    - path: path to the folder that contains a file for each simulated GRB;
    Output: 
    - grb_list_sim: list of GRB objects;
    """
    grb_sim_names = os.listdir(path)
    grb_list_sim  = []
    for grb_file in grb_sim_names:
        left_idx  = grb_file.find('lc') + len('lc')
        right_idx = grb_file.find('.txt')
        grb_name  = grb_file[left_idx:right_idx] # extract the ID of the GRB as string
        # read files
        print(path+grb_file)
        try: 
            times, counts, errs, model, modelbkg, bg, t90, n_sig_pulses, n_pls = np.genfromtxt(path+grb_file, unpack=True) # works with "export_grb()"
        except:
            times, counts, errs, t90 = np.genfromtxt(path+grb_file, unpack=True) # works with "export_LC()"
            n_sig_pulses = np.array([-1])

        times    = np.float32(times)
        counts   = np.float32(counts)
        errs     = np.float32(errs)
        t90      = np.float32(t90)
        model    = np.float32(model)
        modelbkg = np.float32(modelbkg)
        bg       = np.float32(bg)

        grb      = GRB(grb_name=grb_name, 
                       times=times, 
                       counts=counts, 
                       errs=errs, 
                       t90=t90[0], 
                       model=model,
                       modelbkg=modelbkg,
                       bg=bg[0],
                       grb_data_file_path=path+grb_file, 
                       num_of_sig_pulses=n_sig_pulses[0])
        grb_list_sim.append(grb)

    print("Total number of simulated GRBs: ", len(grb_sim_names))
    return grb_list_sim

################################################################################

def apply_constraints(grb_list, t90_threshold, sn_threshold,bin_time, t_f, 
                      t90_frac=15, sn_distr=False, filter=True, 
                      zero_padding=True, t_cut=200., verbose=True):
    """
    Given as input a list of GBR objects, the function outputs a list containing
    only the GRBs that satisfy the following constraint:
    - T90 > t90_threshold (2 sec);
    - GRB signal S2N > sn_threshold;
    - the measurement lasts at least for t_f (150 sec) after the peak;
    Input:
    - t90_threshold [s];
    - sn_threshold;
    - bin_time: temporal bin size of the instrument [s];
    - t_f: time after the peak that we need the signal to last [s];
    - sn_distr: if True, returns and export also the distribution of the s2n 
                (and the total counts inside the T20%) of _only_ the GRBs that 
                have passed the constraint selection;
    - filter: if True, it applies a savgol filter before computing the S2N;
    - zero_padding: if True, pads to zero outside the 5/3 of the T20% interval;
    - t_cut: time after the peak at which the zero padding stops.
    Output:
    - good_grb_list: list of GRB objects, where each one is a GRB that satisfies
                     the 3 constraints described above;
    - sn_levels: list containing the s2n ratio of _all_ the input GRBs (not only
                 of those selected); 
    """
    
    good_grb_list = []
    sn_levels     = {}
    total_cnts    = {}
    grb_with_neg_t20 = 0
    for grb in grb_list:
        times   = np.float32(grb.times)
        counts  = np.float32(grb.counts)
        errs    = np.float32(grb.errs)
        t90     = np.float32(grb.t90)
        i_c_max = np.argmax(counts)
        t_c_max = times[i_c_max]
        
        try:
                                                                                       
            s_n, T20, T20_start, T20_stop, sum_grb_counts = evaluateGRB_SN(times=times, 
                                                                              counts=counts, 
                                                                              errs=errs, 
                                                                              t90=t90,
                                                                              t90_frac=t90_frac,
                                                                         bin_time=bin_time,
                                                                          filter=filter,
                                                                          return_cnts=True)
        except AssertionError:
            #remove GRB if the t20% is negative
            grb_with_neg_t20 += 1
            s_n = 0
        if (zero_padding):
            padding_mask         = np.logical_or(times<T20_start-T20/3., 
                                                 times>T20_stop +T20/3.)
            counts[padding_mask] = 0.
            errs[padding_mask]   = 0.
            t_max                = np.max(times)
            if(t_max<=t_cut):
                missing_bins = np.zeros(int((t_cut-t_max)/bin_time)+1)
                counts       = np.concatenate([counts, missing_bins])
                errs         = np.concatenate([errs,   missing_bins])
                times        = np.linspace(min(times), t_cut, len(counts))
                
                # re-define times, counts, and errors after the padding 
                grb.times  = times
                grb.counts = counts
                grb.errs   = errs         
                         

        cond_1 = t90>t90_threshold
        cond_2 = s_n>sn_threshold
        cond_3 = len(counts[i_c_max:])>=(t_f/bin_time)
        
        if (cond_1 and cond_2 and cond_3):
            grb.t20 = T20
            grb.s2n = s_n
            good_grb_list.append(grb)
            if sn_distr:
                sn_levels[grb.name]  = s_n
                total_cnts[grb.name] = sum_grb_counts

    if verbose:
        print("Total number of input GRBs: ", len(grb_list))
        print("GRBs with negative duration: ", grb_with_neg_t20)
        print("GRBs that satisfy the constraints: ", len(good_grb_list)) 

    # Export the s2n distribution of the GRBs that passed the constraints
    if sn_distr:
        with open('./sn_distr.txt', 'w') as f:
            print("# grb_name    s2n    total_cnts", file=f)
            for key in sn_levels:
                print(f"{key}    {sn_levels[key]}    {total_cnts[key]}", file=f)
    if sn_distr:
        return good_grb_list, sn_levels
    else:
        return good_grb_list

################################################################################
################################################################################

def compute_autocorrelation(grb_list, N_lim, t_min=0, t_max=150, 
                            bin_time=0.064, mode='scipy', compute_rms=False):
    """
    Compute the autocorrelation (ACF) of the GRBs. The ACF is computed up to
    a shift of the light curve of t_max = 150 seconds. 
    The correct way to compute the ACF is by using:
    - for the REAL data, the Link93 formula (using BOTH counts and errs, i.e.,
    col 2 (`grb.counts`) and 3 (`grb.errs`) of data files, respectively). 
    - for the SIMULATED data, the `scipy.signal.correlate` function on the _clean_ 
    model curve (i.e., `grb.model`, with no bkg and no poisson added). 
    Inputs:
    - grb_list: list of GRB objects;
    - N_lim: max number of GRBs with which we compute the average ACF;
    - t_min: min time lag for the autocorrelation [s], set by default to zero;
    - t_max: max time lag for the autocorrelation [s];
    - bin_time: temporal bin size of the instrument [s];
    - mode: choose the method to compute the ACF between:
            'scipy': use the scipy.signal.correlate() function. method='auto' 
            automatically chooses direct or Fourier method based on an estimate
            of which is faster.
            'link93': use the method described in Link et al., 1993;
    - compute_rms: if True, the function computes and returns also the rms of 
                   the autocorrelation;
    Outputs:
    - steps: time lags of the autocorrelation;
    - acf: autocorrelation;
    - acf_rms: autocorrelation rms.
    """

    steps   = int((t_max-t_min)/bin_time) # number of steps for ACF
    acf_sum = np.zeros(steps)
    if compute_rms:
        acf_sum_square = np.zeros(steps)

    # Evaluate ACF
    for grb in grb_list[:N_lim]:
        
        if mode=='scipy':
            counts = np.array(grb.model)
            # Compute the ACF using the `scipy.signal.correlate` function
            acf   = signal.correlate(in1=counts, in2=counts, method='auto')
            acf   = acf / np.max(acf)  # np.max(acf) is equal to np.sum(counts**2)
            lags  = signal.correlation_lags(in1_len=len(counts), in2_len=len(counts))
            idx_i = np.where(lags*bin_time==t_min)[0][ 0] # select the index corresponding to t =   0 s
            idx_f = np.where(lags*bin_time<=t_max)[0][-1] # select the index corresponding to t = 150 s
            assert lags[idx_i]==t_min, "ERROR: The left limit of the autocorrelation is not computed correctly..."
            assert np.isclose(lags[idx_f]*bin_time, t_max, atol=1e-1), "ERROR: The right limit of the autocorrelation is not computed correctly..."         
            # Select only the autocorrelation up to a shift of t_max = 150 s
            acf = acf[idx_i:idx_f]
        elif mode=='link93':
            counts = np.array(grb.counts)
            errs   = np.array(grb.errs)
            # errs = 0
            # Compute the ACF using the `Link93` formula
            acf = [np.sum((np.roll(counts, u) * counts)[u:]) / np.sum(counts**2 - errs**2) for u in range(steps)]
            acf = np.array(acf)
        acf_sum += acf
        if compute_rms:
            acf_sum_square += acf**2

    # Compute the average ACF
    acf = acf_sum/N_lim
    if compute_rms:
        acf_square = acf_sum_square/N_lim
        acf_rms    = np.sqrt(acf_square - acf**2)

    if mode=='scipy':
        steps  = lags[idx_i:idx_f]
    elif mode=='link93':
        acf[0] = 1
        steps  = np.arange(steps) 

    if compute_rms:
        return steps, acf, acf_rms
    else:
        return steps, acf



def compute_autocorrelation_fast(grb_list, N_lim, t_min=0, t_max=150, 
                            bin_time=0.064, mode='scipy', compute_rms=False):
    """
    Compute the autocorrelation (ACF) of the GRBs. The ACF is computed up to
    a shift of the light curve of t_max = 150 seconds. 
    The correct way to compute the ACF is by using:
    - for the REAL data, the Link93 formula (using BOTH counts and errs, i.e.,
    col 2 (`grb.counts`) and 3 (`grb.errs`) of data files, respectively). 
    - for the SIMULATED data, the `scipy.signal.correlate` function on the _clean_ 
    model curve (i.e., `grb.model`, with no bkg and no poisson added). 
    Inputs:
    - grb_list: list of GRB objects;
    - N_lim: max number of GRBs with which we compute the average ACF;
    - t_min: min time lag for the autocorrelation [s], set by default to zero;
    - t_max: max time lag for the autocorrelation [s];
    - bin_time: temporal bin size of the instrument [s];
    - mode: choose the method to compute the ACF between:
            'scipy': use the scipy.signal.correlate() function. method='auto' 
            automatically chooses direct or Fourier method based on an estimate
            of which is faster.
            'link93': use the method described in Link et al., 1993;
    - compute_rms: if True, the function computes and returns also the rms of 
                   the autocorrelation;
    Outputs:
    - steps: time lags of the autocorrelation;
    - acf: autocorrelation;
    - acf_rms: autocorrelation rms.
    """

    steps   = int((t_max-t_min)/bin_time) # number of steps for ACF
    acf_sum = np.zeros(steps)
    if compute_rms:
        acf_sum_square = np.zeros(steps)

    # Evaluate ACF
    for grb in grb_list[:N_lim]:
        if mode=='scipy':
            counts = np.array(grb.model)
            N = len(counts)
            
            acf = signal.correlate(in1=counts, in2=counts, method='auto')
            
            zero_lag_idx = N - 1
            acf = acf / acf[zero_lag_idx]  
            
            lag_i_bins = int(np.round(t_min / bin_time))
            
            idx_i = zero_lag_idx + lag_i_bins
            idx_f = idx_i + steps # <--- FIX: Guarantee the length is exactly 'steps'
            
            acf = acf[idx_i:idx_f]

        elif mode=='link93':
            counts = np.array(grb.counts)
            errs   = np.array(grb.errs)
            # Compute the ACF using the `Link93` formula
            acf = [np.sum((np.roll(counts, u) * counts)[u:]) / np.sum(counts**2 - errs**2) for u in range(steps)]
            acf = np.array(acf)
            
        acf_sum += acf
        if compute_rms:
            acf_sum_square += acf**2

    # Compute the average ACF
    acf = acf_sum/N_lim
    if compute_rms:
        acf_square = acf_sum_square/N_lim
        acf_rms    = np.sqrt(acf_square - acf**2)

    if mode=='scipy':
        # Recreate the lags integer array exactly as the original `lags[idx_i:idx_f]` did
        lag_i_bins = int(np.round(t_min / bin_time))
        lag_f_bins = int(np.round(t_max / bin_time))
        steps = np.arange(lag_i_bins, lag_f_bins)
        
    elif mode=='link93':
        acf[0] = 1
        steps  = np.arange(steps) 

    if compute_rms:
        return steps, acf, acf_rms
    else:
        return steps, acf
################################################################################

def compute_kde_log_duration(duration_list, x_left=-2, x_right=5, h_opt=0.09):
    """
    Compute the kernel density estimate of the distribution of the (log10) of
    the duration of the selected GRBs;
    Input:
    - duration_list: list containing all the T20% durations of the selected GRBs
                     obtained as output of the function evaluateDuration20();
    - x_left:   left endpoint for the array onto which we compute the sum of gaussians;
    - x_right: right endpoint for the array onto which we compute the sum of gaussians;
    - h_opt: optimal sigma of the gaussian; this value has been obtained with
             GridSearch optimization (see the notebook in DEBUG section);
    Output:
    - dur_distr: kernel density estimate of the log of the duration of the selected GRBs;
    """ 
    duration_list = np.log10(duration_list)
    # Apply kernel density estimation to distribution of durations:
    x_grid     = np.linspace(x_left, x_right, 1000)
    dur_distr  = stats.norm.pdf(x_grid, duration_list[:, None], h_opt) # (x=, loc=, scale=)
    dur_distr /= len(duration_list)
    dur_distr  = dur_distr.sum(0)
    return dur_distr

def compute_kde_fast(duration_list, x_left=-2, x_right=5, h_opt=0.09):
    # 1. Pre-convert to numpy array and log space
    data = np.log10(duration_list)[:, np.newaxis] # Shape (N, 1)
    x_grid = np.linspace(x_left, x_right, 1000)   # Shape (1000,)
    
    # 2. Use the explicit Gaussian formula
    # Gaussian = (1 / (sigma * sqrt(2 * pi))) * exp(-0.5 * ((x - mu) / sigma)^2)
    inv_sigma = 1.0 / h_opt
    norm_const = inv_sigma / np.sqrt(2 * np.pi)
    
    # Broad-casting math: (1000,) - (N, 1) -> (N, 1000)
    diff = (x_grid - data) * inv_sigma
    np.square(diff, out=diff)
    diff *= -0.5
    np.exp(diff, out=diff)
    
    # Sum and normalize
    return (diff.sum(axis=0) * norm_const) / len(duration_list)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def compute_all_metrics(grb_list_sim,t_f=150,bin_time=0.064,test_sn_distr=True,t90_frac=15):
#--------------------------------------------------------------------------#
# COMPUTE AVERAGE QUANTITIES OF SIMULATED DATA
#--------------------------------------------------------------------------#
### TEST 1&2: Average Peak-Aligned Profiles
    avgd_profile_sim, avgd_profile_square_sim,avgd_profile_cube_sim = compute_avgd_profile(grb_list_sim,npts_before=2343,npts_after=2343)

    ### TEST 3: Autocorrelation
    # For the REAL LCs we use the Link+93 formula to compute the autocorrelation,
    # whereas for the simulated LCs instead we use the scipy.signal.correlate
    # function on the model curve, i.e., the one before adding the Poisson noise.
    steps_sim, acf_sim = compute_autocorrelation_fast(grb_list=grb_list_sim,
                                                    N_lim=len(grb_list_sim),
                                                    t_max=t_f,
                                                    bin_time=bin_time,
                                                    mode='scipy',
                                                    compute_rms=False)
    ### TEST 4: Duration
    duration_sim       = np.array( [ grb.t20 for grb in grb_list_sim ] )
    duration_distr_sim = compute_kde_fast(duration_list=duration_sim)
    ### TEST 5: S2N distribution
    if test_sn_distr:
        sn_distr_sim = np.array( [ grb.s2n for grb in grb_list_sim ] )
    else:
        sn_distr_sim = []
    return avgd_profile_sim,avgd_profile_square_sim,avgd_profile_cube_sim,steps_sim,acf_sim,duration_distr_sim,sn_distr_sim


################################################################################
def two_pop_test(distr_1, distr_2, mode='AD'):
    """ 
    Perform a 2-population statistical compatibility test (AD or KS).
    Returns the p-value of the test.
    """
    if mode=='AD':
        res_ad = anderson_ksamp([distr_1,distr_2])
        pvalue = res_ad.significance_level
        #print('AD (p-value): ', pvalue)
    elif mode=='KS':
        res_ks = ks_2samp(distr_1,distr_2)
        pvalue = res_ks.pvalue
        #print('KS (p-value): ', pvalue)
    return pvalue     
################################################################################

def loss_compatibility_test(p, alpha=0.05, rescale_factor=1):
    """
    Calculate the loss for a compatibility test based on a p-value.
    
    Parameters:
    p (float): The p-value of the compatibility test.
    rescale_factor (float, optional): A rescaling factor for the loss. Default is 1.
    alpha (float, optional): The significance level for the compatibility test. Default is 0.05.
    
    Returns:
    The calculated loss.
    """
    if p >= alpha:
        loss = 0
    else:
        loss = 1 - np.log10(p)

    return loss / rescale_factor

def compute_loss(x_bins, y_bins, edges_left,edges_right,
                x_bins_cube,y_bins_cube,edges_left_cube,edges_right_cube,
                avgd_profile_sim,      avgd_profile_square_sim, avgd_profile_cube_sim,                
                 acf,                  acf_sim,
                 duration,             duration_sim,
                 n_of_pulses,          n_of_pulses_sim,
                 sn_distrib_real=[],   sn_distrib_sim=[],  
                 log_fluence_sim=[],test_fluence =True,
                 test_sn_distr=False,  test_pulse_distr=False,compute_before_peak=False,
                 return_individual_loss=False, log=False, verbose=False):
    """
    Compute the loss to be used for the optimization in the Genetic Algorithm.
    Input:
    -
    Output:
    - l2_loss: L2 loss;
    """
    #print('Check that the real bins are normalized','Avgd:',np.max(y_bins),'Cube:',np.max(y_bins_cube))
    if log:
        
        
        averaged_fluxes[averaged_fluxes<=0] = eps_log
        averaged_fluxes_sim[averaged_fluxes_sim<=0] = eps_log
        averaged_fluxes_cube[averaged_fluxes_cube<=0] = eps_log
        averaged_fluxes_cube_sim[averaged_fluxes_cube_sim<=0] = eps_log
        
        
        averaged_fluxes          = np.log10(averaged_fluxes)
        averaged_fluxes_sim      = np.log10(averaged_fluxes_sim)
        averaged_fluxes_cube     = np.log10(averaged_fluxes_cube)
        averaged_fluxes_cube_sim = np.log10(averaged_fluxes_cube_sim)

        if compute_before_peak:
            
            averaged_fluxes_before_peak[averaged_fluxes_before_peak<=0] = eps_log
            averaged_fluxes_before_peak_sim[averaged_fluxes_before_peak_sim<=0] = eps_log
            averaged_fluxes_cube_before_peak[averaged_fluxes_cube_before_peak<=0] = eps_log
            averaged_fluxes_cube_before_peak_sim[averaged_fluxes_cube_before_peak_sim<=0] = eps_log
            
            averaged_fluxes_before_peak          = np.log10(averaged_fluxes_before_peak)
            averaged_fluxes_before_peak_sim      = np.log10(averaged_fluxes_before_peak_sim)
            averaged_fluxes_cube_before_peak     = np.log10(averaged_fluxes_cube_before_peak)
            averaged_fluxes_cube_before_peak_sim = np.log10(averaged_fluxes_cube_before_peak_sim)
            
        
        acf                      = np.log10(acf)
        
        acf_sim[acf_sim<=0] = eps_log
        acf_sim                  = np.log10(acf_sim)
        
    
    w1 = 1.
    w2 = 1.
    w3 = 1.
    w4 = 1.
    w5 = 1.

    if test_fluence == True:
        w6 = 1
    else:
        w6  = 0

    w_tot = w1 + w2 + w3 + w4 + w5 + w6

    ## real
    
    epsylon=1e-10


    x_adp_sim,y_adp_sim,npts_sim = rebin_with_given_edges_fast(times_tot,avgd_profile_sim,edges_left,edges_right,stat='mean')
    x_adp_sim_cube,y_adp_sim_cube,npts_sim_cube = rebin_with_given_edges_fast(times_tot,avgd_profile_cube_sim,edges_left_cube,edges_right_cube,stat='mean')

    y_bins_safe = np.clip(y_bins, eps_log, np.inf)
    y_adp_sim_safe = np.clip(y_adp_sim, eps_log, np.inf)
    
    y_bins_cube_safe = np.clip(y_bins_cube, eps_log, np.inf)
    y_adp_sim_cube_safe = np.clip(y_adp_sim_cube, eps_log, np.inf)


    l2_loss_fluxes = np.nanmean(np.sqrt(np.abs(np.log10(y_bins_safe+epsylon)-np.log10(y_adp_sim_safe+epsylon))))
    l2_loss_fluxes_cube = np.nanmean(np.sqrt(np.abs(np.log10(y_bins_cube_safe+epsylon)-np.log10(y_adp_sim_cube_safe+epsylon))))


    l2_loss_acf         = np.sqrt( np.nansum(np.power((acf-acf_sim),2)) )
    l2_loss_duration    = np.sqrt( np.sum(np.power((duration-duration_sim),2)) )
    l_sn_distr          = 0.
    l_pulse_distr       = 0.
    
    ### Compute the loss associated to the difference in S/N distributions (real vs sim)
    if test_sn_distr:
        p_sn_distr = two_pop_test(distr_1=sn_distrib_real, 
                                  distr_2=sn_distrib_sim,
                                  mode='KS')
        l_sn_distr = loss_compatibility_test(p=p_sn_distr)

    if test_fluence:
        p_fluence = two_pop_test(distr_1=log_fluence_sim, 
                                  distr_2=log_fluence_obs_list,
                                  mode='KS')
        l_fluence = loss_compatibility_test(p=p_fluence)
        if np.inf(l_fluence) or l_fluence > 1e6:
            l_fluence = 100 

    else:
        l_fluence = 0

   
    ### Total loss
    l2_loss = w1 * l2_loss_fluxes      + \
              w2 * l2_loss_fluxes_cube + \
              w3 * l2_loss_acf         + \
              w4 * l2_loss_duration    + \
              w5 * l_sn_distr          + \
              w6 * l_fluence
              
    # Divide to obtain the _average_ value of the loss
    l2_loss /= w_tot

    print('L2 loss fluxes:', "%1.3f"%l2_loss_fluxes,
        ' L2 loss fluxes cube:', "%1.3f"%l2_loss_fluxes_cube,
          ' L2 loss ACF:', "%1.3f"%l2_loss_acf,' L2 loss duration:', "%1.3f"%l2_loss_duration,
          ' Loss S/N distr:', "%1.3f"%l_sn_distr,'L Fluence',"%1.3f"%l_fluence, 'L_tot',"%1.3f"%l2_loss)


    if verbose:
        # WE SHOULD CHECK WHAT IS THE ORDER OF MAGNITUDE OF EACH LOSS, SO THAT
        # WE KNOW HOW MUCH THEY CONTRIBUTE TO THE TOTAL!
        # Incidentally, there is one combination of log/no-log that makes the 
        # loss functions all in the range ~ np.abs( [0,1] ), which is the one
        # obtained by NOT choosing the log on averaged_fluxes, averaged_fluxes_square,
        # and acf, while choosing the log for duration (which is automatically
        # in log scale, since it is the output of the function compute_kde_log_duration())
        pass
                          
    if return_individual_loss:
        if test_sn_distr and test_pulse_distr:
            return l2_loss, l2_loss_fluxes, l2_loss_fluxes_cube, l2_loss_acf, l2_loss_duration, l_sn_distr, l_pulse_distr
        elif test_sn_distr:
            return l2_loss, l2_loss_fluxes, l2_loss_fluxes_cube, l2_loss_acf, l2_loss_duration, l_sn_distr
        elif test_pulse_distr:
            return l2_loss, l2_loss_fluxes, l2_loss_fluxes_cube, l2_loss_acf, l2_loss_duration, l_pulse_distr
        #elif compute_before_peak:
        #   return l2_loss, l2_loss_fluxes, l2_loss_fluxes_square, l2_loss_fluxes_before_peak, l2_loss_fluxes_cube_before_peak, l2_loss_acf, l2_loss_duration
        else:
            return l2_loss, l2_loss_fluxes, l2_loss_fluxes_cube, l2_loss_acf, l2_loss_duration
    else:
        return l2_loss



def generate_GRBs_lum_func(N_grb,                                                              # number of simulated GRBs to produce
                  tau_i,tau_se,gamma,xi,alpha1,Lb,alpha2,                                                # 7 parameters
                  instrument, bin_time, eff_area, bg_level,
                  e_1,e_2,sn_threshold,t_f,                                                  # constraint parameters 
                  t90_threshold, t90_frac=15, filter=True,                            # constraint parameters
                  export_files=False, export_path='None',                             # other parameters
                  with_bg=False, seed=None,                               # other parameters
                  remove_instrument_path=False, test_pulse_distr=False, N_accepted=15):               # other parameters         
    """
    This function generates a list of GRBs using the stochastic differential equation. As input it takes the X parameters needed for the SDE, and the parameters of the instrument considered. As output it returns 
    a list, where each element of the list is an GRB object, which has successfully 
    passed the constraints selection (see "apply_constraints()" function). 

    Input:
    - N_grb: total number of simulated GRBs to produce in output;
    ### X parameters
    - tau_i:
    - tau_d:
    - alpha:
    - tau_se:
    - gamma:
    - delta:
    - x_0:
    - alpha:
    ### instrument parameters
    - instrument:
    - res:
    - eff_area:
    - bg_level;
    ### constraint parameters
    - sn_threshold: 
    - t_f:
    - t90_threshold:
    - t90_frac:
    - filter:
    ### other parameters
    - export_files: if True, every GRB that passed the constraint selection is
                    exorted into an external file. The file contains 8 columns.
    - export_path
    - with_bg:
    - seed:
    - test_pulse_distr: if True, it appends to each GRB object also the info
                        on the number of significative pulses inside that GRB,
                        and we also compute the time distances between all the
                        pulses in that GRB.
    - remove_instrument_path: if True, remove the instrument name from the name of the path;
    Output:
    -grb_list_sim: list containing N_grb GRB objects, where each light-curve
                   satisfies the imposed constraints.
    """


    cnt=0
    grb_list_sim         = []
    Liso_list_sim         = []
    n_trial_grb=0
    while (cnt<N_grb):
        #############################
        n_trial_grb+=1
        #print(n_trial_grb)
        
        if n_trial_grb == 100:
            if cnt < 1:
                raise Exception("Too many trials to generate GRBs")
            
        if n_trial_grb == 1000:
            if cnt < 2:
                raise Exception("Too many trials to generate GRBs")
            
        if n_trial_grb > N_accepted*N_grb:
                        raise Exception("Too many trials to generate GRBs")
        #####################################
        lc = LC(### 7 parameters defining the SDE
                tau_i=tau_i,
                tau_se=tau_se,
                gamma=gamma,
                xi=xi,
                alpha1=alpha1,
                Lb=Lb,
                alpha2=alpha2,
                ### instrument parameters:
                res=bin_time,
                eff_area=eff_area,
                bg_level=bg_level,
                ### other parameters:
                instrument=instrument,
                with_bg=with_bg,
                e_1=e_1,
                e_2=e_2
                )
        #lc.generate_avalanche(seed=None)
        lc.generate_LC_from_sde()
        if lc.check==0:
            # check that we have generated a lc with non-zero values; otherwise,
            # skip it and continue in the generation process
            del(lc)
            continue

   
        n_of_sig_pulses, \
        n_of_total_pulses, \
        sig_pulses                   = None, None, None
        lc._minimum_peak_rate_list   = None
        lc._peak_rate_list           = None
        lc._current_delay_list       = None
        lc._minimum_pulse_delay_list = None

        # initialize T20% to None
        t20_in=None
        s2n_in = None

        # convert the lc generated from the avalance into a GRB object
        grb = GRB(grb_name='lc_candidate.txt',
                  times=lc._times, 
                  counts=lc._plot_lc,    # these are COUNTS (not count rates!). See `avalanche.py`
                  model=lc._model,       # model COUNTS
                  modelbkg=lc._modelbkg, # model COUNTS + constant bgk
                  bg=lc._bg*lc._res,     # COUNTS of bkg
                  errs=lc._err_lc,       # errors on the COUNTS
                  t90=lc._t90,
                  t90_start=lc._t90_i,
                  t90_stop=lc._t90_f,
                  t20=t20_in,
                  s2n=s2n_in,
                  log_E_iso=lc.log_E_iso,
                  log_L_iso=lc.log_L_iso,
                  z=lc._z,
                  k_corr=lc.k_corr,
                  k_conv =lc.k_conv,
                  alpha_band = lc.alpha_band,
                  beta_band = lc.beta_band,
                  E_pi = lc.E_pi,
                  log_peak_ph_flux=lc.log_peak_ph_flux,
                  log_fluence=lc.log_fluence,
                  num_of_sig_pulses=n_of_sig_pulses,
                  grb_data_file_path=export_path+instrument+'/'+'lc_candidate.txt',
                  minimum_peak_rate_list=lc._minimum_peak_rate_list,
                  peak_rate_list=lc._peak_rate_list,
                  current_delay_list=lc._current_delay_list,
                  minimum_pulse_delay_list=lc._minimum_pulse_delay_list
                  ) # total number of pulses composing the GRB
                  
        # we use a temporary list that contains only _one_ lc, then we
        # check if that GRB satisfies the constraints imposed, ad if that is
        # the case, we append it to the final list of GRBs
        if grb.t90<t90_threshold: 
            # preliminary check to ensure that the savgol will not fail due
            # to short GRBs, for which often this filter fails. The reason
            # is that the `window_length` of savgol filter must be greater than 
            # `polyorder`, but for short GRBs the computed `window_length` is
            # very small.
            del(lc)
            continue
        
        
        grb_list_sim_temp = [ grb ]

        Liso_list_sim.append(grb.log_L_iso)

        # save the GRB into the final list _only_ if it passed the constraints selection
        if (len(grb_list_sim_temp)==1):
     
            if export_files:
                export_grb(grb=grb, 
                           idx=cnt, 
                           instrument=instrument,
                           remove_instrument_path=remove_instrument_path,
                           path=export_path)
               
                grb.name = 'lc'+str(cnt)+'.txt'
            grb_list_sim.append(grb)
            cnt+=1
        del(lc)
    #print('Total Generated: ', generated)
    print('Number of trials to generate {0} GRBs: {1}'.format(N_grb, n_trial_grb))
    return grb_list_sim, Liso_list_sim

################################################################################


def generate_GRBs(N_grb,# number of simulated GRBs to produce
                  tau_i,tau_se,gamma,xi,alpha1,Lb,alpha2,# 7 parameters
                  instrument, bin_time, eff_area, bg_level,
                  e_1,e_2,sn_threshold,t_f,                                              
                  t90_threshold, t90_frac=15, filter=True,                            
                  export_files=False, export_path='None',                             
                  with_bg=False, seed=None,                               
                  remove_instrument_path=False, test_pulse_distr=False, N_accepted=15):                      
    """
    This function generates a list of GRBs using the stochastic differential equation. As input it takes the X parameters needed for the SDE, and the parameters of the instrument considered. As output it returns 
    a list, where each element of the list is an GRB object, which has successfully 
    passed the constraints selection (see "apply_constraints()" function). 
    Input:
    - N_grb: total number of simulated GRBs to produce in output;
    ### X parameters
    - tau_i:
    - alpha:
    - tau_se:
    - gamma:
    - x_0:
    - alpha:
    ### instrument parameters
    - instrument:
    - res:
    - eff_area:
    - bg_level;
    ### constraint parameters
    - sn_threshold: 
    - t_f:
    - t90_threshold:
    - t90_frac:
    - filter:
    ### other parameters
    - export_files: if True, every GRB that passed the constraint selection is
                    exorted into an external file. The file contains 8 columns.
    - export_path
    - with_bg:
    - seed:
    - test_pulse_distr: if True, it appends to each GRB object also the info
                        on the number of significative pulses inside that GRB,
                        and we also compute the time distances between all the
                        pulses in that GRB.
    - remove_instrument_path: if True, remove the instrument name from the name of the path;
    Output:
    -grb_list_sim: list containing N_grb GRB objects, where each light-curve
                   satisfies the imposed constraints.
    """
    cnt=0
    grb_list_sim         = []
    n_trial_grb=0
    while (cnt<N_grb):
        #############################
        n_trial_grb+=1
        if n_trial_grb == 100:
            if cnt < 1:
                #print('Failed after having tried 100 times')
                raise Exception("Too many trials to generate GRBs")
        
        if n_trial_grb == 1000:
            if cnt < 2:
                raise Exception("Too many trials to generate GRBs")
                
        if n_trial_grb > N_accepted*N_grb:
            raise Exception("Too many trials to generate GRBs")
        #raise an error if too many trials are made to generate grbs
        #####################################
        lc = LC(### 7 parameters defining the SDE
                tau_i=tau_i,
                tau_se=tau_se,
                gamma=gamma,
                xi=xi,
                alpha1=alpha1,
                Lb=Lb,
                alpha2=alpha2,
                ### instrument parameters:
                res=bin_time,
                eff_area=eff_area,
                bg_level=bg_level,
                ### other parameters:
                instrument=instrument,
                with_bg=with_bg,
                e_1=e_1,
                e_2=e_2
                )

        lc.generate_LC_from_sde()
        if lc.check==0:
            # check that we have generated a lc with non-zero values; otherwise,
            # skip it and continue in the generation process
            del(lc)
            continue

        n_of_sig_pulses, \
        n_of_total_pulses, \
        sig_pulses                   = None, None, None
        lc._minimum_peak_rate_list   = None
        lc._peak_rate_list           = None
        lc._current_delay_list       = None
        lc._minimum_pulse_delay_list = None
        # initialize T20% to None
        t20_in=None
        s2n_in = None
        # convert the lc generated from the avalance into a GRB object
        grb = GRB(grb_name='lc_candidate.txt',
                  times=lc._times, 
                  counts=lc._plot_lc,    # these are COUNTS (not count rates!). See `avalanche.py`
                  model=lc._model,       # model COUNTS
                  modelbkg=lc._modelbkg, # model COUNTS + constant bgk
                  bg=lc._bg*lc._res,     # COUNTS of bkg
                  errs=lc._err_lc,       # errors on the COUNTS
                  t90=lc._t90,
                  t90_start=lc._t90_i,
                  t90_stop=lc._t90_f,
                  t20=t20_in,
                  s2n=s2n_in,
                  log_E_iso=lc.log_E_iso,
                  log_L_iso=lc.log_L_iso,
                  z=lc._z,
                  k_corr=lc.k_corr,
                  k_conv =lc.k_conv,
                  alpha_band = lc.alpha_band,
                  beta_band = lc.beta_band,
                  E_pi = lc.E_pi,
                  log_peak_ph_flux=lc.log_peak_ph_flux,
                  log_fluence=lc.log_fluence,
                  num_of_sig_pulses=n_of_sig_pulses,
                  grb_data_file_path=export_path+instrument+'/'+'lc_candidate.txt',
                  minimum_peak_rate_list=lc._minimum_peak_rate_list,
                  peak_rate_list=lc._peak_rate_list,
                  current_delay_list=lc._current_delay_list,
                  minimum_pulse_delay_list=lc._minimum_pulse_delay_list
                  ) # total number of pulses composing the GRB
                  
        # we use a temporary list that contains only _one_ lc, then we
        # check if that GRB satisfies the constraints imposed, ad if that is
        # the case, we append it to the final list of GRBs
        if grb.t90<t90_threshold: 
            # preliminary check to ensure that the savgol will not fail due
            # to short GRBs, for which often this filter fails. The reason
            # is that the `window_length` of savgol filter must be greater than 
            # `polyorder`, but for short GRBs the computed `window_length` is
            # very small.
            del(lc)
            continue
        
        
        grb_list_sim_temp = [ grb ]
        
       
        grb_list_sim_temp = apply_constraints(grb_list=grb_list_sim_temp, 
                                              bin_time=bin_time, 
                                              t90_threshold=t90_threshold, 
                                              t90_frac=t90_frac,
                                              sn_threshold=sn_threshold, 
                                              t_f=t_f,
                                              filter=filter,
                                              verbose=False)
        # save the GRB into the final list _only_ if it passed the constraints selection
        if (len(grb_list_sim_temp)==1):
            if export_files:
                export_grb(grb=grb, 
                           idx=cnt, 
                           instrument=instrument,
                           remove_instrument_path=remove_instrument_path,
                           path=export_path)
                
                grb.name = 'lc'+str(cnt)+'.txt'
            grb_list_sim.append(grb)
            cnt+=1
        del(lc)
    print('Number of trials to generate {0} GRBs: {1}'.format(N_grb, n_trial_grb))
    return grb_list_sim

################################################################################

# """
def export_grb(grb, idx, instrument, remove_instrument_path=False, path='../simulations/'):
    """
    Export the simulated grb in a file with these columns: 
        times, counts, err_counts, T90, num of sig. pulses.
    Input:
    - grb: object that contains the GRB;
    - idx: number of the light curve;
    - instrument: string with the name of the instrument;
    - remove_instrument_path: if True, remove the instrument name from the name of the path;
    - path: path where to store the results of the simulations;
    """
    if remove_instrument_path:
        outfile = path+'/'+'lc'+str(idx)+'.txt'
    else:
        outfile = path+instrument+'/'+'lc'+str(idx)+'.txt'
        outfile2 =  path+instrument+'/'+'lc'+str(idx)+'energetics.txt'
        outfile3 =  path+instrument+'/'+'spectral'+str(idx)+'.txt'

    times        = grb.times
    lc           = grb.counts # this are COUNTS, not count rates!
    err_lc       = grb.errs
    np.savetxt(outfile, np.column_stack((times, lc, err_lc)),fmt="%1.4f")
    
    log_E_iso = grb.log_E_iso
    log_L_iso = grb.log_L_iso
    z         = grb.z
    log_peak_ph_flux = grb.log_peak_ph_flux
    log_fluence = grb.log_fluence

 
    np.savetxt(outfile2,[log_E_iso,log_L_iso,z,log_peak_ph_flux,log_fluence],fmt='%1.4f')

    alpha_band = grb.alpha_band
    beta_band = grb.beta_band
    E_pi = grb.E_pi

    np.savetxt(outfile3,[alpha_band,beta_band,E_pi],fmt='%1.4f')




def read_values(filename):
    """Reads variables and their values from a text file, skipping comments 
    and empty lines. Defines a dictionary with all the stored variables. Cast
    all the numerical/string variables with the proper type."""
    variables = {}
    with open(filename, 'r') as file:
        for line in file:
            # Skip comment lines and empty lines
            if not line.strip().startswith('#') and not line.strip()=='': 
                # Split the line into variable name and value, stripping whitespace
                name, value = line.strip().split('=', 1)
                name  = name.strip()
                value = value.strip()
                # If the value is a number, then automatically convert the type
                # to 'float'; if the conversion fails, keep the value as 'string'
                try:
                    value = float(value)
                except ValueError:
                    pass
                variables[name] = value
        # Check that the variables that must be numbers/str, are so
        assert isinstance(variables['instrument'], (str))
        assert isinstance(variables['N_grb'],      (float))
        assert isinstance(variables['mu'],         (float))
        assert isinstance(variables['mu0'],        (float))
        assert isinstance(variables['alpha'],      (float))
        assert isinstance(variables['delta1'],     (float))
        assert isinstance(variables['delta2'],     (float))
        assert isinstance(variables['tau_min'],    (float))
        assert isinstance(variables['tau_max'],    (float))
        # Cast 'N_grb' as 'int'
        variables['N_grb'] = int(variables['N_grb'])
    # Return the 'dict' containing all the parameters needed for the simulation
    return variables

###############################################################################

def create_dir(variables, path='../', folder='simulated_LCs'):
    """Create the directory where to store the simulated GRB LCs, using the
    date and time of code execution."""
    base_dir = Path(path)
    # Create a directory to store all the simulations (if it does not already exist)
    sim_dir = Path(base_dir/folder)
    sim_dir.mkdir(parents=True, exist_ok=True)
    # Generate timestamp for the execution (Year-Month-Day_Hour_Minute_Second)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Create a directory to store the LCs
    lcs_dir = Path(sim_dir/f"{timestamp}") # Path(sim_dir/f"{variables['instrument]+'_'+timestamp}")
    lcs_dir.mkdir(parents=True, exist_ok=True)
    return sim_dir, lcs_dir, timestamp

###############################################################################

def save_config(variables, file_name=None):
    """Print and export the 'config'."""
    for name, value in variables.items():
        print(f"{name} = {value}")
        #print(f"{name} (type={type(value)}) = {value}")
    if file_name is not None:
        with open(file_name, 'w') as f:
            for name, value in variables.items():
                print(f"{name} = {value}", file=f)

###############################################################################

def rebin_histogram(bin_edges, data, n_min=20):
    """
    Rebins the histogram so that each bin contains at least 20 data points.
    We start with a predefined set of bins (e.g., we decide the have nbins=15
    bins, so that bin_edges will be: 
        bin_edges = np.linspace(min(data), max(data), nbins+1)
    Every time in a predefined bin there are less than n_min=20 counts, we 
    extend the right endpoint of the current bin towards the next bin. We do
    this iteratively until the current (enlarged) bin has >= n_min=20 points.
    The last points on the right that do not form a complete bin of n_min is
    included in the bin immediately before (which then becomes the last one).
    Args:
        - bin_edges: A list of the bin endpoints;
        - data: The data to be rebinned;
    Returns:
        - new_bin_edges: A list of the new bin edges;
        - new_bin_counts: A list of the new bin counts;
    """
  
    data      = np.array(data)
    bin_edges = np.array(bin_edges)
    new_bin_counts = []
    new_bin_edges  = []
    new_bin_edges.append(bin_edges[0])
    for i in range(len(bin_edges)-1):
        bin_count = len( data[(new_bin_edges[-1]<=data) & (data<bin_edges[i+1])] )
        if (bin_count >= n_min):
            new_bin_edges.append(bin_edges[i+1])
            new_bin_counts.append(bin_count)

    if (np.sum(new_bin_counts)!=len(data)):
        # if the last data did not form a bin with len>20, 
        # then we incorporate in the last complete (n>20) bin
        new_bin_counts[-1] += len(data[data>=new_bin_edges[-1]])
        new_bin_edges[-1]   = bin_edges[-1]
    new_bin_counts = np.array(new_bin_counts).astype('int')
 
    assert np.sum(new_bin_counts) == len(data), "The number of counts in the rebinned histogram is different from the initial one!" 
    
    return new_bin_edges, new_bin_counts
###############################################################################
