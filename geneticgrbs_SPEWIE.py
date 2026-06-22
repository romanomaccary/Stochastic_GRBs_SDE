# -*- coding: utf-8 -*-
################################################################################
# # IMPORT LiBRARIES
################################################################################
from ast import If
from genericpath import exists
import sys
import time
from tkinter import N
from matplotlib.pylab import f
import pygad
import pickle
import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import uuid
import socket
# Get machine name (hostname)
hostname = socket.gethostname()
print(f"Hostname: {hostname}")


print(1)
### Increase the recursion limit to avoid: "RecursionError: maximum recursion depth exceeded in comparison"
rec_lim=50000
if sys.getrecursionlimit()<rec_lim:
    sys.setrecursionlimit(rec_lim)

### Suppress some warnings_
# import warnings
# warnings.filterwarnings("ignore", message="p-value capped")
# warnings.filterwarnings("ignore", message="p-value floored")


save_plot=0

save_folder='/astrodata/romain/sde_GA/geneticgrbs_v2/genetic_algorithm/RESULT/result_SPEWie4.3_fermi_rest_frame_lomax_renorm_poly_bpl_paper/norm_exp_relax_fast_v1/'
export_path='/astrodata/romain/GA_SIMULATIONS/geneticgrbs_simulations_SPEWie4.3_fermi_rest_frame_lomax_renorm_poly_bpl_paper/norm_exp_relax_fast_v1/'


os.makedirs(save_folder,exist_ok=True)
os.makedirs(export_path,exist_ok=True)
################################################################################

random_seed=677762

print(random_seed)

def fix_all_seeds(seed):
    #Fix randomness. Usage: fix_all_seeds(42)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark     = True

fix_all_seeds(random_seed)
print_time=True

################################################################################
# SET PATHS
################################################################################

### Set the username for the path of the files
user = 'romano'
if user == 'romano':
    # library paths
    sys.path.append('/astrodata/romain/sde_GA/geneticgrbs_v2/SPEWie4.3_bat_rest_frame_lomax_norm_poly_bpl_paper')
    # real data
    batse_path = '/astrodata/guidorzi/CGRO_BATSE/'
    swift_path = '/astrodata/guidorzi/Swift_BAT/'
    sax_path   = '/astrodata/guidorzi/BeppoSAX_GRBM/'
    fermi_path = '/astrodata/romain/GBM_LC_repository/data/' 
else:
    raise ValueError('Assign to the variable "user" a correct username!')

from statistical_test_SPEWIE import *
from sde_SPEWIE4 import LC

################################################################################
# SET PARAMETERS
################################################################################

### Choose the instrument


instrument = 'fermi' # choose between: 'batse', 'swift', and 'fermi'

if instrument=='swift':

    t_i           = 0                            # [s]
    t_f           = 150                          # [s]
    eff_area      = instr_swift['eff_area']      # 1400 # effective area of instrument [cm2]
    bg_level      = instr_swift['bg_level']      # (10000/eff_area) # background level [cnt/cm2/s]
    t90_threshold = instr_swift['t90_threshold'] # 2 # [s] --> used to select only _long_ GfRBs
    t90_frac      = 15
    sn_threshold  = instr_swift['sn_threshold']  # 10 # signal-to-noise ratio
    bin_time      = instr_swift['res']           # 0.064 # [s] temporal bins for Swift (time resolution)
    e_1           = instr_swift['e_1']
    e_2           = instr_swift['e_2']
    N_accepted     = instr_swift['N_accepted']
    test_times    = np.linspace(t_i, t_f, int((t_f-t_i)/bin_time))
    swift_data = '/astrodata/romain/sde_GA/BAT_real_data/'
    avgd_profile = np.loadtxt(swift_data+'avgd_profile_BAT.txt')
    x_bins,y_bins,edges_left,edges_right = np.loadtxt(swift_data+'binned_profile_BAT.txt',unpack=True)
    x_bins_cube,y_bins_cube,edges_left_cube,edges_right_cube = np.loadtxt(swift_data+'binned_profile_cube_BAT.txt',unpack=True)
    acf_real = np.loadtxt('/astrodata/romain/sde_GA/BAT_real_data/acf_swift.txt')

elif instrument=='batse':
    t_i           = 0                            # [s]
    t_f           = 150                          # [s]
    eff_area      = instr_batse['eff_area']      # 2025  # effective area of instrument [cm2]
    bg_level      = instr_batse['bg_level']      # 2.8   # background level [cnt/cm2/s]
    t90_threshold = instr_batse['t90_threshold'] # 2     # [s] --> used to select only _long_ GRBs
    t90_frac      = 15
    sn_threshold  = instr_batse['sn_threshold']  # 70    # signal-to-noise ratio
    e_1           = instr_batse['e_1']
    e_2           = instr_batse['e_2']
    bin_time      = instr_batse['res']           # 0.064 # [s] temporal bins for BATSE (time resolution)
    N_accepted     = instr_batse['N_accepted']       # 25    # number of accepted trials before rejecting parameters
    test_times    = np.linspace(t_i, t_f, int((t_f-t_i)/bin_time))
    batse_data_path = '/astrodata/romain/sde_GA/BATSE_real_data/'
    avgd_profile = np.loadtxt(batse_data_path+'avgd_flux_BATSE.txt')
    x_bins,y_bins,edges_left,edges_right =  np.loadtxt(batse_data_path+'binned_profile_BATSE.txt',unpack=True)
    avgd_profile_cube =  np.loadtxt(batse_data_path+'avgd_flux_cube_BATSE.txt')
    x_bins_cube,y_bins_cube,edges_left_cube,edges_right_cube = np.loadtxt(batse_data_path+'binned_profile_cube_BATSE.txt',unpack=True)
    acf_real = np.loadtxt(batse_data_path+'acf_BATSE.txt')


elif instrument=='fermi':
    t_i           = 0                            # [s]
    t_f           = 150                          # [s]
    eff_area      = instr_fermi['eff_area']      # 1400 # effective area of instrument [cm2]
    bg_level      = instr_fermi['bg_level']      # (10000/eff_area) # background level [cnt/cm2/s]
    t90_threshold = instr_fermi['t90_threshold'] # 2 # [s] --> used to select only _long_ GRBs
    t90_frac      = 15
    sn_threshold  = instr_fermi['sn_threshold']  # 15 # signal-to-noise ratio
    bin_time      = instr_fermi['res']           # 0.064 # [s] temporal bins for Swift (time resolution)
    e_1          = instr_fermi['e_1']              # 8.0   # lower energy bound of the energy band considered for Fermi GRBs (in keV)
    e_2          = instr_fermi['e_2']              # 1000.0# upper energy bound of the energy band considered for Fermi GRBs (in keV)
    N_accepted     = instr_fermi['N_accepted']       # 25    # number of accepted trials before rejecting parameters
    bin_time = 0.064 # [s]
    test_times    = np.linspace(t_i, t_f, int((t_f-t_i)/bin_time))
    fermi_path = '/astrodata/romain/GBM_LC_repository/data/' 
    fermi_data_path = '/astrodata/romain/sde_GA/fermi_real_data/'
    
    avgd_profile = np.loadtxt(fermi_data_path+'avgd_profile_fermi.txt')
    avgd_profile_cube = np.loadtxt(fermi_data_path+'avgd_profile_cube_fermi.txt')
    print('avgd_profile_cube:', avgd_profile_cube)
    x_bins,y_bins,edges_left,edges_right = np.loadtxt(fermi_data_path+'binned_profile_fermi.txt',unpack=True)
    x_bins_cube,y_bins_cube,edges_left_cube,edges_right_cube = np.loadtxt(fermi_data_path+'binned_profile_cube_fermi.txt',unpack=True)
    acf_real = np.loadtxt(fermi_data_path+'acf_fermi.txt')

    # Load data and apply constraints
    saved = False

else:


    raise NameError('Variable "instrument" not defined properly; choose between: "batse", "swift", "sax", "sax_lr", and "fermi".')


#------------------------------------------------------------------------------#

# Genetic Algorithm parameters
parent_selection_type = "tournament" 
crossover_probability = 1                      # 'None' means couples parent k with parent k+1, otherwise it selects from the parents candidate list each one of them with probability 'crossover_probability', and then it takes two of them at random
initial_population    = None                   # if 'None', the initial population is randomly chosen using the 'sol_per_pop; and 'num_genes' parameters
mutation_type         = "random"
crossover_type        = "scattered"
num_generations       = 10                     # Number of generations.
sol_per_pop           = 100 # 1000             # Number of solutions in the population (i.e., number of different sets per generation).
print('Number of solutions per population',sol_per_pop)
num_parents_mating    = int(0.15*sol_per_pop)  # Number of solutions to be selected as parents in the mating pool.
keep_parents          = 0                      # if 0, keep NO parents (the ones selected for mating in the current population) in the next population
keep_elitism          = 0                      # keep in the next generation the best N solution of the current generation
mutation_probability  = 0.06 # 0.08                   # by default is 'None', otherwise it selects a value randomly from the current gene's space (each gene is changed with probability 'mutation_probability')



# Other parameters
N_grb            = 30   # 2000 number of simulated GRBs to produce per set of parameters
test_sn_distr    = True   # add a fifth metric regarding  the S/N distribution (set True by default)
test_pulse_distr = False  # add a sixth metric regarding the distribution of number of pulses per GRB (set False by default)
test_fluence      = False # add a seventh metric regarding the luminosity distribution
print('test_fluence:',test_fluence)
compute_before_peak = False # add the pre-peak profile

print("Number of accepted trials before rejecting parameters: ", N_accepted)
zero_padding = True # if True, we apply zero padding to the simulated LCs, i.e., we set to zero the counts before the first and after the last detected pulse. This is done to be consistent with the real data, where we apply zero padding to the LCs before computing the average peak-aligned profile. We set it to False because it is computationally faster, and because in this way we can use the same code for both real and simulated LCs without having to apply zero padding to the simulated LCs at each iteration of the GA.
print('zero_padding:',zero_padding)
# Options for parallelization
if user=='pleiadi':
    n_processes = int(os.environ['OMP_NUM_THREADS'])
else:
    n_processes = 20
    
parallel_processing  = ["process", n_processes]  # USE THIS ONE!  
#parallel_processing = ["thread", n_processes]   # this is slower
parallel_processing = None                      # single thread

# Name of the pkl file where to save the GA instance at the end of the run
filename_model = save_folder+'geneticGRB_sde'

epsilon = 1e-6

##########################################################################
### swift ################################################################
range_tau_i      = {"low": np.log10(0.2),            "high": np.log10(2.5)} # log scale
range_tau_se     = {"low": np.log10(0.5),             "high": np.log10(10)}
range_gamma     = {"low":0.05,                             "high":0.55}
range_xi     = {"low":20,                       "high":50}
range_alpha1 = {"low":0.75,                         "high":1.75}
range_Lb   = {"low":np.log10(5e51),                         "high":np.log10(1e53)}
range_alpha2 = {"low":1.5,                         "high":2.5}
###########################################################################
###########################################################################

print('range_tau_i',range_tau_i)
print('range_tau_se',range_tau_se)
print('range_gamma',range_gamma)
print('range_xi',range_xi)
print('range_alpha1',range_alpha1)
print('range_Lb',range_Lb)
print('range_alpha2',range_alpha2)

range_constraints = [range_tau_i,range_tau_se,range_gamma,range_xi,range_alpha1,range_Lb,range_alpha2]
num_genes = len(range_constraints)

nparams=len(range_constraints) 
print('nparams=',nparams)
save_model = True

print('\n\n')
print('################################################################################')
print('START')
print('################################################################################')
print('\n\n')


################################################################################
# LOAD REAL DATA
################################################################################

init_load_time = time.perf_counter()

### Load the Swift GRBs
if instrument=='swift': 
    # load all data
    grb_list_real = load_lc_swift(path=swift_path)
    # apply constraints
    grb_list_real = apply_constraints(grb_list=grb_list_real, 
                                      bin_time=bin_time, 
                                      t90_threshold=t90_threshold,
                                      t90_frac=t90_frac, 
                                      sn_threshold=sn_threshold, 
                                      t_f=t_f,
                                      zero_padding=zero_padding)
    n_of_pulses_real = None
    
    grbs_oks = np.loadtxt('/astrodata/romain/sde_GA/grbs_swift_no_zero_padding.txt',dtype=str)
    grb_list_real =  [grb for grb in grb_list_real if grb.name in grbs_oks]
    print('Number of grbs without zero padding',len(grb_list_real))



elif instrument=='batse':
    # load all data
    grb_list_real = load_lc_batse(path=batse_path)
    # apply constraints
    grb_list_real = apply_constraints(grb_list=grb_list_real,
                                      bin_time=bin_time,
                                      t90_threshold=t90_threshold,
                                      t90_frac=t90_frac,
                                      sn_threshold=sn_threshold,
                                      #sn_threshold_sup=sn_threshold_sup,
                                      t_f=t_f,
                                      zero_padding=zero_padding)
    n_of_pulses_real = None

elif instrument=='fermi':


    grb_list_real = load_lc_fermi(fermi_path)
    all_grb_list_real = [grb.name for grb in grb_list_real]
    grb_list_real = apply_constraints(grb_list_real,t90_threshold,sn_threshold,bin_time,150,zero_padding=zero_padding)        
    n_of_pulses_real = None


else:
    raise NameError('Variable "instrument" not defined properly; choose between: "batse", "swift", "sax", "fermi".')

end_load_time = time.perf_counter()
print('\n')
print('--------------------------------------------------------------------------------')
print('* {} data loaded in {:0.0f} sec'.format(instrument,(end_load_time-init_load_time)))
print('--------------------------------------------------------------------------------')



################################################################################
# COMPUTE AVERAGE QUANTITIES OF REAL DATA
################################################################################


### TEST 4: Duration
duration_real = [grb.t20 for grb in grb_list_real ]
duration_distr_real = compute_kde_fast(duration_list=duration_real)


### TEST 5: S2N distribution
if test_sn_distr:
    sn_distr_real = np.array( [ grb.s2n for grb in grb_list_real ] )
else:
    sn_distr_real = []

################################################################################
# DEFINE FITNESS FUNCTION OF THE GENETIC ALGORITHM
################################################################################


def fitness_func(ga_instance, solution, solution_idx=None):

    try:
        grb_list_sim = generate_GRBs(# number of simulated GRBs to produce:
                    N_grb=N_grb,
                    # 5 parameters:
                    tau_i=10**solution[0],
                    tau_se=10**solution[1],
                    gamma=solution[2],
                    xi = solution[3],
                    alpha1 = solution[4],
                    Lb = 10**solution[5],
                    alpha2 = solution[6],
                    # instrument parameters:
                    instrument=instrument,
                    bin_time=bin_time,
                    eff_area=eff_area,
                    bg_level=bg_level,
                    e_1=e_1,
                    e_2=e_2,
                    # constraint parameters:
                    sn_threshold=sn_threshold,
                    t90_threshold=t90_threshold,
                    t90_frac=t90_frac,
                    t_f=t_f,
                    filter=True,
                    # other parameters:
                    export_files=False,
                    with_bg=False,
                    test_pulse_distr=test_pulse_distr,
                    N_accepted=N_accepted,
                    #zero_padding=True
                    )
    except Exception:
        return 1e-9


    if test_pulse_distr:
        n_of_pulses_sim = [ grb.num_of_sig_pulses for grb in grb_list_sim ]
    else:
        n_of_pulses_sim = None

    if test_fluence:
        log_fluence_sim = np.array([grb.log_fluence for grb in grb_list_sim])
    else:
        log_fluence_sim = []

    #--------------------------------------------------------------------------#
    # COMPUTE AVERAGE QUANTITIES OF SIMULATED DATA
    #--------------------------------------------------------------------------#
    avgd_profile_sim, \
    avgd_profile_square_sim, \
    avgd_profile_cube_sim, \
    steps_sim, \
    acf_sim, \
    duration_distr_sim, \
    sn_distr_sim = compute_all_metrics(grb_list_sim=grb_list_sim)

    #--------------------------------------------------------------------------#
    # COMPUTE LOSS
    #--------------------------------------------------------------------------#
    l2_loss = compute_loss(x_bins=x_bins,
                           y_bins=y_bins,
                           edges_left=edges_left,
                           edges_right=edges_right,
                           x_bins_cube=x_bins_cube,
                           y_bins_cube=y_bins_cube,
                           edges_left_cube=edges_left_cube,
                           edges_right_cube=edges_right_cube,
                           avgd_profile_sim=avgd_profile_sim,
                           avgd_profile_square_sim=avgd_profile_square_sim,
                           avgd_profile_cube_sim=avgd_profile_cube_sim,
                           acf=acf_real,
                           acf_sim=acf_sim,
                           duration=duration_distr_real,
                           duration_sim=duration_distr_sim,
                           n_of_pulses=n_of_pulses_real,
                           n_of_pulses_sim=n_of_pulses_sim,
                           sn_distrib_real=sn_distr_real,
                           sn_distrib_sim=sn_distr_sim,
                           log_fluence_sim=log_fluence_sim,
                           test_fluence=test_fluence,
                           test_pulse_distr=test_pulse_distr,
                           test_sn_distr=test_sn_distr,
                           compute_before_peak=compute_before_peak,
                           log=False
                           )
    fitness = 1.0 / (l2_loss + 1.e-9)
    print('fitness=',"%1.3f"%fitness)
    return fitness

################################################################################
# DEFINE AUXILIARY FUNCTION
################################################################################

def write_best_par_per_epoch(solution,loss, generation, filename=save_folder+'best_par_per_epoch.txt'):
    """
    Function to write the best parameters of each generation in a file. The file
    is opened in append mode, so that we can append the results of eacch generation
    at the end of the file at each epoch.
    Parameters:
    - solution: array containing the best solution (7 params) of a generation.
    - filename: The name of the file to open in append mode. Default is 'output.txt'.
    """
    with open(filename, 'a') as file:
        file.write("Generation = {generation}".format(generation=generation)+'\n')
        file.write("tau_i        = {solution}".format(solution="%1.3f"%(10**solution[0]))+'\n')
        file.write("tau_se    = {solution}".format(solution="%1.3f"%(10**solution[1]))+'\n')
        file.write("gamma   = {solution}".format(solution="%1.3f"%(solution[2]))+'\n')
        file.write("xi   = {solution}".format(solution="%1.3f"%(solution[3]))+'\n')
        file.write("alpha1    = {solution}".format(solution="%1.3f"%(solution[4]))+'\n')
        file.write("Lb    = {solution}".format(solution="%1.3e"%(10**solution[5]))+'\n')
        file.write("alpha2    = {solution}".format(solution="%1.3f"%(solution[6]))+'\n')
        file.write(loss+'\n')
        file.write('\n')
        

def write_median_best_par_per_epoch(population, generation,loss,filename=save_folder+'median_best_par_per_epoch.txt'):
    
    median_params = np.median(population, axis=0)
    # Extract each median parameter explicitly
    tau_i     = 10**median_params[0]
    tau_se    = 10**median_params[1]
    gamma     = median_params[2]
    xi        = median_params[3]
    alpha1    = median_params[4]
    Lb        = 10**median_params[5]
    alpha2    = median_params[6]
    # write to file with the same format as in the print above
    with open(filename, 'a') as file:
        file.write("###")
        file.write('Generation = {generation}'.format(generation=generation)+'\n')
        file.write("tau_i        = {solution}".format(solution="%1.3f"%(tau_i)+'\n'))
        file.write("tau_se    = {solution}".format(solution="%1.3f"%tau_se+'\n'))
        file.write("gamma   = {solution}".format(solution="%1.3f"%gamma+'\n'))
        file.write("xi   = {solution}".format(solution="%1.3f"%xi+'\n'))
        file.write("alpha1    = {solution}".format(solution="%1.3f"%alpha1+'\n'))
        file.write("Lb    = {solution}".format(solution="%1.3e"%(Lb)+'\n'))
        file.write("alpha2    = {solution}".format(solution="%1.3f"%alpha2+'\n'))
        file.write("Averaged Loss  = {avgd_loss}".format(avgd_loss="%1.3f"%loss+'\n'))
        file.write('###')
        file.write('\n')
    
last_fitness, last_loss, current_fitness, current_loss = 0, 0, 0, 0
def on_generation(ga_instance):
    """
    This function is executed after each generation. It prints useful 
    information of the current epoch, in particular, the details of the best
    solution in the current generation.
    """
    global last_fitness, last_loss, current_fitness, current_loss
    print('--------------------------------------------------------------------------------')
    print("Generation     = {generation}".format(generation=ga_instance.generations_completed))
    current_fitness       = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
    current_loss          = current_fitness**(-1)                
    print("Best Loss      = {solution_loss}".format(solution_loss="%1.3f"%current_loss))
    avgd_loss           = np.median(ga_instance.last_generation_fitness**(-1))
    print("Averaged Loss  = {avgd_loss}".format(avgd_loss="%1.3f"%avgd_loss))
    print("Best Fitness   = {fitness}".format(fitness="%1.3f"%current_fitness))
    print("Fitness Change = {change}".format(change=current_fitness-last_fitness))
    last_fitness          = current_fitness
    last_loss             = current_loss

    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    # Print the best solution of the current generation on TERMINAL
    print("Parameters of the best solution in the current generation:")
    print("    - tau_i        = {solution}".format(solution=10**solution[0]))
    print("    - tau_se       = {solution}".format(solution=10**solution[1]))
    print("    - gamma       = {solution}".format(solution=solution[2]))
    print("    - xi          = {solution}".format(solution=solution[3]))
    print("    - alpha1     = {solution}".format(solution=solution[4]))
    print("    - Lb         = {solution}".format(solution=10**solution[5]))
    print("    - alpha2     = {solution}".format(solution=solution[6]))

    population = ga_instance.population  # Shape: (num_solutions, 6)
    median_params = np.median(population, axis=0)
    generation = ga_instance.generations_completed

    # # Extract each median parameter explicitly
    tau_i     = 10**median_params[0]
    tau_se    = 10**median_params[1]
    gamma     = median_params[2]
    xi        = median_params[3]
    alpha1   = median_params[4]
    Lb      = 10**median_params[5]
    alpha2   = median_params[6]


    # Print each parameter explicitly    
    print(f"\nGeneration {generation} median parameters:")
    print(f"  tau_i    : {tau_i:.3f}")
    print(f"  tau_se   : {tau_se:.3f}")
    print(f"  gamma    : {gamma:.3f}")
    print(f"  xi       : {xi:.3f}")
    print(f"  alpha1   : {alpha1:.3f}")
    print(f"  Lb       : {Lb:.3e}")
    print(f"  alpha2   : {alpha2:.3f}")

    fitness_values = ga_instance.last_generation_fitness
    loss_values = fitness_values**(-1)
    print('loss values below threshold',len( np.where(loss_values >1e8)[0] ))
    num_below_threshold = len( np.where(loss_values>1e8)[0] )/len(loss_values)
    print("Percentage of discarded solutions: ","%1.3f"%(100*num_below_threshold),"%")
    # Print the best solution of the current generation on FILE
    write_best_par_per_epoch(solution,"Best Loss      = {solution_loss}".format(solution_loss=current_loss),generation)
    write_median_best_par_per_epoch(population, generation,avgd_loss,filename=save_folder+'median_best_par_per_epoch.txt')

def on_start(ga_instance):
    print("ga started")

def on_fitness(ga_instance, population_fitness):
    print("computing fitness...")

def on_parents(ga_instance, selected_parents):
    print("selecting parents...")

def on_crossover(ga_instance, offspring_crossover):
    print("do crossover ...")

def on_mutation(ga_instance, offspring_mutation):
    print("do mutation ...")


def on_stop(ga_instance, last_population_fitness):
    print("on_stop()")


################################################################################
# INSTANTIATE THE 'GENETIC ALGORITHM' CLASS
################################################################################


def load_ga(filename="saved_ga.pkl"):
    with open(filename, "rb") as f:
        ga_GRB = pickle.load(f)
        
    ga_GRB.on_start = on_start
    ga_GRB.on_fitness = on_fitness
    ga_GRB.on_parents = on_parents
    ga_GRB.on_crossover = on_crossover
    ga_GRB.on_mutation = on_mutation
    ga_GRB.on_generation = on_generation
    ga_GRB.on_stop = on_stop
    
    return ga_GRB

if __name__ == '__main__':

    MODE ='first'
    print("MODE=",MODE)
    print('seed=',random_seed)
    if MODE=='first':
        ga_GRB = pygad.GA(num_generations=num_generations,
                          num_parents_mating=num_parents_mating,
                          sol_per_pop=sol_per_pop,
                          num_genes=num_genes,
                          gene_type=float,
                          initial_population=initial_population,
                          on_start=on_start,
                          on_fitness=on_fitness,
                          on_parents=on_parents,
                          on_crossover=on_crossover,
                          on_mutation=on_mutation,
                          on_generation=on_generation,
                          on_stop=on_stop,
                          ### fitness function:
                          fitness_func=fitness_func,
                          ### parent selection:
                          parent_selection_type=parent_selection_type,
                          keep_parents=keep_parents,           
                          keep_elitism=keep_elitism,           
                          ### crossover:
                          crossover_probability=crossover_probability,
                          crossover_type=crossover_type,
                          ### mutation:
                          mutation_type=mutation_type,
                          mutation_probability=mutation_probability,     
                          ### set range of parameters:
                          gene_space=range_constraints,
                          ### other stuff:
                          save_best_solutions=True,
                          save_solutions=True,
                          parallel_processing=parallel_processing,
                          random_seed=random_seed)
    



    ga_GRB.summary() 

    ############################################################################
    # RUN THE GENETIC ALGORITHM
    ############################################################################

    init_run_time = time.perf_counter()
    print('\nStarting the GA...\n')
    # write the time in day, month, year, hour, minute, second
    print('On',hostname,'with',n_processes,'processes',' at time',time.strftime("%d/%m/%Y %H:%M:%S"),'with random seed',random_seed)
    #measure the execution time of ga_GRB.run()

    
    print(f"Total generations requested: {ga_GRB.num_generations}")
    print(f"Generations completed: {ga_GRB.generations_completed}")
    print(f"Remaining generations: {ga_GRB.num_generations - ga_GRB.generations_completed}")

    ga_GRB.run()
    #ga_GRB.plot_fitness()
    end_run_time = time.perf_counter()
    
    run_time = end_run_time - init_run_time
    # write the time in day, month, year, hour, minute, second
    print('Finished the GA at time',time.strftime("%d/%m/%Y %H:%M:%S"))
    #print('Execution time: {0} seconds'.format(run_time))
    # convert in hours
    print('Execution time: {0} hours'.format(run_time/3600))
    #
    

    print('\n')
    print('--------------------------------------------------------------------------------')
    print('* Model run in {:0.0f} sec'.format((end_run_time-init_run_time)))
    print('--------------------------------------------------------------------------------')


    ############################################################################
    # SAVE THE MODEL
    ############################################################################

    ### Save the GA instance
    if save_model:
        ga_GRB.on_start = None  # Remove before saving to avoid pickling error
        ga_GRB.on_fitness = None
        ga_GRB.on_crossover = None
        ga_GRB.on_mutation = None
        ga_GRB.on_generation = None
        ga_GRB.on_stop = None
        ga_GRB.on_parents = None
        ga_GRB.save(filename=filename_model)

    ############################################################################
    # PRINT FINAL RESULTS
    ############################################################################

    #--------------------------------------------------------------------------#
    # Print on terminal
    #--------------------------------------------------------------------------#
    solution, solution_fitness, solution_idx = ga_GRB.best_solution(ga_GRB.last_generation_fitness)
    print('\n################################################################################')
    print('################################################################################')
    print("* Parameters of the BEST solution:")
    print("    - tau_i        = {solution:.3f}".format(solution=10**solution[0]))
    print("    - tau_se       = {solution:.3f}".format(solution=10**solution[1]))
    print("    - gamma       = {solution:.3f}".format(solution=solution[2]))
    print("    - xi          = {solution:.3f}".format(solution=solution[3]))
    print("    - alpha1    = {solution:.3f}".format(solution=solution[4]))
    print("    - Lb        = {solution:.3e}".format(solution=10**solution[5]))
    print("    - alpha2    = {solution:.3f}".format(solution=solution[6]))

    print("* Loss value of the best solution    : {solution_loss}".format(solution_loss=solution_fitness**(-1)))
    print("* Fitness value of the best solution : {solution_fitness}".format(solution_fitness=solution_fitness))
    #print("Index of the best solution          : {solution_idx}".format(solution_idx=solution_idx))
    if ga_GRB.best_solution_generation != -1:
        print("* Best fitness value reached after N={best_solution_generation} generations.".format(best_solution_generation=ga_GRB.best_solution_generation))
    print('################################################################################')
    print('################################################################################')
    #--------------------------------------------------------------------------#
    # Print on file
    #--------------------------------------------------------------------------#
    if MODE == 'first':
        file = open(save_folder+"simulation_info.txt", "w")
    elif MODE == 'resume':
        file = open(save_folder+"simulation_info.txt", "a")   
    #file = open(save_folder+"simulation_info.txt", "w")
    file.write('################################################################################')
    file.write('\n')
    file.write("INPUT")
    file.write('\n')
    file.write('################################################################################')
    file.write('\n')
    file.write('\n')
    file.write('N_GRBs_per_set       = {}'.format(N_grb))
    file.write('\n')
    file.write('num_generations      = {}'.format(num_generations))
    file.write('\n')
    file.write('sol_per_pop          = {}'.format(sol_per_pop))
    file.write('\n')
    file.write('num_parents_mating   = {}'.format(num_parents_mating))
    file.write('\n')
    file.write('keep_parents         = {}'.format(keep_parents))
    file.write('\n')
    file.write('keep_elitism         = {}'.format(keep_elitism))
    file.write('\n')
    file.write('mutation_probability = {}'.format(mutation_probability))
    file.write('\n')
    file.write('Number of accepted trials before rejecting parameters = {}'.format(N_accepted))
    file.write('\n')
    file.write('\n')
    file.write('range_tau_i             = {}'.format(range_tau_i))
    file.write('\n')
    file.write('range_tau_se         = {}'.format(range_tau_se))
    file.write('\n')
    file.write('range_gamma         = {}'.format(range_gamma))
    file.write('\n')
    file.write('range_xi         = {}'.format(range_xi))
    file.write('\n')
    file.write('range_alpha1         = {}'.format(range_alpha1))
    file.write('\n')
    file.write('range_Lb         = {}'.format(range_Lb))
    file.write('\n')
    file.write('range_alpha2         = {}'.format(range_alpha2))
    file.write('\n')
    file.write('################################################################################')
    file.write('\n')
    file.write("OUTPUT")
    file.write('\n')
    file.write('################################################################################')
    file.write('\n')
    file.write('\n')
    file.write("* Parameters of the BEST solution:")
    file.write('\n')
    file.write("    - tau_i      = {solution}".format(solution=10**solution[0]))
    file.write('\n')
    file.write("    - tau_se  = {solution}".format(solution=10**solution[1]))
    file.write('\n')
    file.write("    - gamma  = {solution}".format(solution=solution[2]))
    file.write('\n')
    file.write("    - xi  = {solution}".format(solution=solution[3]))
    file.write('\n')
    file.write("    - alpha1  = {solution}".format(solution=solution[4]))
    file.write('\n')
    file.write("    - Lb  = {solution}".format(solution=10**solution[5]))
    file.write('\n')
    file.write("    - alpha2  = {solution}".format(solution=solution[6]))
    file.write('\n')
    file.write("* Loss value of the best solution    : {solution_loss}".format(solution_loss=solution_fitness**(-1)))
    file.write('\n')
    file.write("* Fitness value of the best solution : {solution_fitness}".format(solution_fitness=solution_fitness))
    file.write('\n')
    #print("Index of the best solution          : {solution_idx}".format(solution_idx=solution_idx))
    if ga_GRB.best_solution_generation != -1:
        file.write("* Best fitness value reached after N = {best_solution_generation} generations.".format(best_solution_generation=ga_GRB.best_solution_generation))
    file.write('\n')
    file.write('\n')
    file.write('################################################################################')
    file.write('\n')
    file.write('################################################################################')
    file.close()
    #--------------------------------------------------------------------------#


    ############################################################################
    # EXPORT DATA FOR THE PLOT 1
    ############################################################################

    if MODE=='first':
        best_loss = np.array(ga_GRB.best_solutions_fitness)**(-1)
        loss_list = np.array(ga_GRB.solutions_fitness)**(-1)
        loss_list = loss_list[np.where(loss_list<1e7)[0]]
        avg_loss  = np.zeros(len(best_loss))
        std_loss  = np.zeros(len(best_loss))
        for i in range(len(best_loss)):
            avg_loss[i] = np.nanmedian( loss_list[i*sol_per_pop:(i+1)*sol_per_pop] )
            std_loss[i] = np.nanstd(  loss_list[i*sol_per_pop:(i+1)*sol_per_pop] )
        
        datafile = save_folder+'datafile.txt'
        file = open(datafile, 'w')
        file.write('# generation\t best_loss\t avg_loss\t std_loss\t std_loss/sqrt(sol_per_pop)\n')
        for i in range(len(best_loss)):
            file.write('{0} {1} {2} {3} {4}\n'.format(i, best_loss[i], avg_loss[i], std_loss[i], std_loss[i]/np.sqrt(sol_per_pop)))
        file.close()


    ############################################################################
    # PLOT THE RESULTS
    ############################################################################

    if save_plot:
        if MODE=='first':
            plt.plot(best_loss, ls='-', lw=2, c='b')
            #plt.yscale('log')
            plt.xlabel(r'Generation')
            plt.ylabel(r'Best Loss')
            plt.savefig(save_folder+'fig01.pdf')
            plt.clf()

            plt.errorbar(np.arange(len(best_loss)), avg_loss, yerr=std_loss/np.sqrt(sol_per_pop), ls='-', lw=2, c='b')
            #plt.yscale('log')
            plt.xlabel(r'Generation')
            plt.ylabel(r'Average Loss')
            plt.savefig(save_folder+'fig02.pdf')
            plt.clf()

            plt.plot(std_loss, ls='-', lw=2, c='b')
            plt.xlabel(r'Generation')
            plt.ylabel(r'Standard Deviation of the loss')
            plt.savefig(save_folder+'fig03.pdf')
            plt.clf()


    ############################################################################
    # EXPORT DATA FOR THE PLOT 2
    ############################################################################
    # Here we save the parameters of ALL the individuals in ALL generations,
    # along with their associated fitness.

    # all fitness values in the ALL epochs:
    all_gen_fitness = np.array(ga_GRB.solutions_fitness[:])

    # all solutions in the ALL epochs:
    all_gen_sol       = np.array(ga_GRB.solutions[:])
    all_gen_tau_i        = 10**np.array(all_gen_sol[:,0])       # array with all the mu      of the ALL generations 
    all_gen_tau_se    = 10**np.array(all_gen_sol[:,1])       # array with all the tau_se  of the ALL generations
    all_gen_gamma    = np.array(all_gen_sol[:,2])       # array with all the gamma  of the ALL generations
    all_gen_xi    = np.array(all_gen_sol[:,3])       # array with all the xi  of the ALL generations
    all_gen_alpha1 = np.array(all_gen_sol[:,4])       # array with all the alpha1  of the ALL generations
    all_gen_Lb = 10**np.array(all_gen_sol[:,5])       # array with all the Lb  of the ALL generations
    all_gen_alpha2 = np.array(all_gen_sol[:,6])       # array with all the alpha2  of the ALL generations

    data_all_gen = {
        'tau_i':        all_gen_tau_i,
        'tau_se':    all_gen_tau_se,
        'gamma':    all_gen_gamma,
        'xi':    all_gen_xi,
        'alpha1': all_gen_alpha1,
        'Lb': all_gen_Lb,
        'alpha2': all_gen_alpha2,
        'fitness':   all_gen_fitness
    }
    df_all_gen = pd.DataFrame(data_all_gen)
    df_all_gen.to_csv(save_folder+'df_all_gen.csv', index=False)    
    
    ############################################################################
    ############################################################################

    print('\n')
    print('################################################################################')
    print('END')
    print('################################################################################')


    ## save light curves produce with the median parameters of the last generation
    last_gen_tau_i = all_gen_tau_i[-sol_per_pop:]
    last_gen_tau_se = all_gen_tau_se[-sol_per_pop:]
    last_gen_gamma = all_gen_gamma[-sol_per_pop:]
    last_gen_xi = all_gen_xi[-sol_per_pop:]
    last_gen_alpha1 = all_gen_alpha1[-sol_per_pop:]
    last_gen_Lb = all_gen_Lb[-sol_per_pop:]
    last_gen_alpha2 = all_gen_alpha2[-sol_per_pop:]

    median_tau_i     = np.median(last_gen_tau_i)
    median_tau_se   = np.median(last_gen_tau_se)
    median_gamma   = np.median(last_gen_gamma)
    median_xi  = np.median(last_gen_xi)
    median_alpha1   = np.median(last_gen_alpha1)
    median_Lb   = np.median(last_gen_Lb)
    median_alpha2   = np.median(last_gen_alpha2)

    print('Out of GA: Median parameters of the last generation:')
    print("    - tau_i      = {solution}".format(solution=median_tau_i))
    print("    - tau_se  = {solution}".format(solution=median_tau_se))
    print("    - gamma  = {solution}".format(solution=median_gamma))
    print("    - xi  = {solution}".format(solution=median_xi))
    print("    - alpha1  = {solution}".format(solution=median_alpha1))
    print("    - Lb  = {solution}".format(solution=median_Lb))
    print("    - alpha2  = {solution}".format(solution=median_alpha2))

    grb_list_sim = generate_GRBs(# number of simulated GRBs to produce:
                                N_grb=N_grb,
                                # 5 parameters:
                                tau_i=median_tau_i,
                                tau_se=median_tau_se,
                                gamma=median_gamma,
                                xi = median_xi,
                                alpha1 = median_alpha1,
                                Lb = median_Lb,
                                alpha2 = median_alpha2,
                                # instrument parameters:
                                instrument=instrument,
                                bin_time=bin_time,
                                eff_area=eff_area,
                                bg_level=bg_level,
                                e_1=e_1,
                                e_2=e_2,
                                # constraint parameters:
                                sn_threshold=sn_threshold,
                                t90_threshold=t90_threshold,
                                t90_frac=t90_frac,
                                t_f=t_f,
                                filter=True,
                                # other parameters:
                                export_files=True,
                                export_path = export_path,
                                with_bg=False,
                                test_pulse_distr=test_pulse_distr,
                                N_accepted=N_accepted
                                )