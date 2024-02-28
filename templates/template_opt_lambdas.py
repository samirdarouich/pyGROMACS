#!/usr/bin/env python3.10
#PBS -q long
#PBS -l nodes=1:ppn=1
#PBS -l walltime=96:00:00
#PBS -j oe
#PBS -N opt_intermediate_python
#PBS -o {{log_path}}
#PBS -l mem=1000mb

import os
import yaml
import logging

from pyGROMACS.utils_automated import ( get_mbar, convergence, get_unified_lambdas, get_inital_intermediates,
                                        change_number_of_states, get_gridpoint_function, adapted_distribution,
                                        restart_configuration, prepare_free_energy, submit_free_energy )

## Define general paths and files ##

simulation_folder = "{{paths.simulation_folder}}"

# Define parameter files
params_setup       = "{{paths.parameter.setup}}"
params_default     = "{{paths.parameter.default}}"
params_ensemble    = "{{paths.parameter.ensemble}}"
params_free_energy = "{{paths.parameter.free_energy}}"

# Define intial coordinate and checkpoint file to start
initial_coord      = "{{paths.initial_coord}}"
initial_cpt        = "{{paths.initial_cpt}}"

# Define logger 
logger_file = f'{simulation_folder}/opt_intermediates.log'

# Delete old logger file if exists
if os.path.exists( logger_file ): os.remove( logger_file )

# Create a logger
logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)

# Create a file handler and set the level to debug
file_handler = logging.FileHandler( logger_file )
file_handler.setLevel(logging.INFO)

# Set the format for log messages
file_handler.setFormatter( logging.Formatter('%(asctime)s -  %(message)s') )

# Add the file handler to the logger
logger.addHandler(file_handler)



#### Simulations settings ####

# Read in simulation settings
with open( params_setup ) as file: simulation_setup = yaml.safe_load(file)

with open( params_default ) as file: simulation_default = yaml.safe_load(file)

with open( params_ensemble ) as file: simulation_ensemble = yaml.safe_load(file)

with open( params_free_energy ) as file: simulation_free_energy = yaml.safe_load(file)

# Define temperature pressure and compressibility
temperature, pressure, compressibility = {{temperature}}, {{pressure}}, {{compressibility}}

# Define simulation time per iteration (nanoseconds)
t_iteration = {{iteration_time}}

# Initial Vdw and Coulomb intermediates
precision  = {{precision}}

combined_lambdas = get_inital_intermediates( no_vdw = 7, no_coul = 4, precision = precision )

logger.info( f"Initial vdW intermediates: { ' '.join( [ f'{max(l-1,0.0):.{precision}f}' for l in combined_lambdas] ) }\n" )
logger.info( f"Initial coul intermediates: { ' '.join( [ f'{min(l,1.0):.{precision}f}' for l in combined_lambdas] ) }\n" )

# Define iteration to start and tolerance (of relative rmsd) when to stop
iteration   = 0
tolerance   = {{tolerance}}
min_overlap = {{min_overlap}}
max_overlap = {{max_overlap}}

# Define iteration ensembles and simulation time
ensembles        = [ "prod" ]
simulation_times = [ t_iteration ]

# Submit intial simulations
job_files = prepare_free_energy( destination_folder = f'{simulation_folder}/iteration{iteration}', combined_lambdas = combined_lambdas, 
                                 ensembles = ensembles, simulation_times = simulation_times, temperature = temperature, 
                                 pressure = pressure, compressibility = compressibility, simulation_free_energy = simulation_free_energy,
                                 simulation_default = simulation_default, simulation_setup = simulation_setup, simulation_ensemble = simulation_ensemble,
                                 initial_coord = [initial_coord]*len(combined_lambdas), initial_cpt = [initial_cpt]*len(combined_lambdas), precision = precision )

submit_free_energy( job_files )

while iteration <= {{max_iterations}}:

    logger.info( f"Iteration n°{iteration}\n" )

    # Initialize the MBAR object
    mbar = get_mbar( path = f"{simulation_folder}/iteration_{iteration}", production = "prod", temperature = temperature )

    logger.info( f"Free energy difference: {mbar.delta_f_.iloc[-1,0] * 8.314 * temperature / 1000 :.3f} ± {mbar.d_delta_f_.iloc[-1,0] * 8.314 * temperature / 1000 :.3f} kJ/mol\n" )

    # Check convergence, if achieved stop.
    rmsd_rel = convergence( mbar )

    logger.info( f"Relative devation to equal partial free energy uncertainties: {rmsd_rel*100 :.2f}%\n" )

    # End the loop if relative rmsd is below tolerance
    if rmsd_rel < tolerance :
        logger.info( "Tolerance is achieved. Conduct final simulations with optimal lambdas!\n" )
        combined_lambdas = get_unified_lambdas( mbar )
        logger.info( f"Best combined intermediates : { ' '.join( [ f'{l:.{precision}f}' for l in combined_lambdas] )  }\n")
        logger.info( f"Best vdW intermediates: { ' '.join( [ f'{max(l-1,0.0):.{precision}f}' for l in combined_lambdas] ) }\n" )
        logger.info( f"Best coul intermediates: { ' '.join( [ f'{min(l,1.0):.{precision}f}' for l in combined_lambdas] ) }\n" )
        break

    # Get the lambdas of previous iteration
    unified_lambdas     = get_unified_lambdas( mbar )

    logger.info( f"Previous vdW intermediates: { ' '.join( [ f'{max(l-1,0.0):.{precision}f}' for l in unified_lambdas] ) }\n" )
    logger.info( f"Previous coul intermediates: { ' '.join( [ f'{min(l,1.0):.{precision}f}' for l in unified_lambdas] ) }\n" )

    # Check overlapp matrix to define if a intermediate need to be added / removed or kept constant
    delta_intermediates = change_number_of_states( mbar, min_overlap = min_overlap, max_overlap = max_overlap)

    logger.info( f"The number of intermediates {'remains the same' if delta_intermediates == 0 else 'is increased by one' if delta_intermediates == 1 else 'is decreased by one' }.\n" )

    # Get gridpoint function of free energy uncertanties
    gridpoint_function  = get_gridpoint_function( mbar )

    # Define new lambda states, the new number of intermediates depends on the overlapp matrix
    combined_lambdas    = adapted_distribution( unified_lambdas, gridpoint_function, len(unified_lambdas)+delta_intermediates)

    logger.info( f"New vdW intermediates: { ' '.join( [ f'{max(l-1,0.0):.{precision}f}' for l in combined_lambdas] ) }\n" )
    logger.info( f"New coul intermediates: { ' '.join( [ f'{min(l,1.0):.{precision}f}' for l in combined_lambdas] ) }\n" )

    # Search the indices of the intermediate states that fit the most to the new adapted lambdas
    restart_indices = restart_configuration( unified_lambdas, combined_lambdas) 

    # Write job files for each lambda
    cord_files = []
    cpt_files  = []
    for i,(_,j) in enumerate( zip( combined_lambdas, restart_indices ) ):
        # coordinate file pointing on the outputfile of the previous iteration that matches the current lambda i the best (relative path)
        cord_files.append( f"../../../iteration_{iteration}/lambda_{j}/00_prod/prod{j}.tpr" )
        # checkpoint file pointing on the previous simulation that iteration the current lambda i the best (relative path)
        cpt_files.append( f"../../../iteration_{iteration}/lambda_{j}/00_prod/prod{j}.cpt" )

    job_files = prepare_free_energy( destination_folder = f'{simulation_folder}/iteration{iteration}', combined_lambdas = combined_lambdas, 
                                     ensembles = ensembles, simulation_times = simulation_times, temperature = temperature, 
                                     pressure = pressure, compressibility = compressibility, simulation_free_energy = simulation_free_energy,
                                     simulation_default = simulation_default, simulation_setup = simulation_setup, simulation_ensemble = simulation_ensemble,
                                     initial_coord = cord_files, initial_cpt = cpt_files, precision = precision )

    submit_free_energy( job_files )
    
    logger.info("\nJobs are finished! Continue with postprocessing\n")

    iteration += 1