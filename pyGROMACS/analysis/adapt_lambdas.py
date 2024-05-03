import re
import os
import glob
import logging
import subprocess
import numpy as np
import pandas as pd

from alchemlyb.estimators import MBAR
from typing import List, Tuple, Dict, Any
from scipy.special import roots_legendre

logger = logging.getLogger("my_logger")

## Special function to submit free energy in optimization
from ..tools.utils import generate_mdp_files, generate_job_file

def prepare_free_energy( destination_folder: str, combined_lambdas: List[float], ensembles: List[str], simulation_times: List[float], temperature: float, 
                         pressure: float, compressibility: float, simulation_free_energy: Dict[str, str|int], simulation_default: Dict[str, Any|Dict[str, str|float]],
                         simulation_setup: Dict[str, Any|Dict[str, str|float]], simulation_ensemble: Dict[str, Any|Dict[str, str|float]], 
                         initial_coord: List[str], initial_cpt: List[str], initial_topo: str, precision: int=3):
        
    # Overwrite provided lambdas in free energy settings
    simulation_free_energy.update( { "init_lambda_states": "".join([f"{x:.0f}" + " "*(precision+2) if x < 10 else f"{x:.0f}" + " "*(precision+1) for x,_ in enumerate(combined_lambdas)]), 
                                     "vdw_lambdas": " ".join( [ f"{max(l-1,0.0):.{precision}f}" for l in combined_lambdas] ), 
                                     "coul_lambdas": " ".join( [ f"{min(l,1.0):.{precision}f}" for l in combined_lambdas] ) } )
        
    # Add free energy settings to overall simulation input
    simulation_default["free_energy"] = simulation_free_energy

    job_files = []

    # Define for each lambda an own folder
    for i,_ in enumerate(combined_lambdas):
        simulation_default["free_energy"]["init_lambda_state"] = i

        lambda_folder = f"{destination_folder}/lambda_{i}"

        # Produce mdp files (for each ensemble an own folder 0x_ensemble)
        mdp_files = generate_mdp_files( destination_folder = lambda_folder, mdp_template = simulation_setup["system"]["paths"]["template"]["mdp_file"],
                                        ensembles = ensembles, temperature = temperature, pressure = pressure, 
                                        compressibility = compressibility, simulation_times = simulation_times,
                                        dt = simulation_default["system"]["dt"], kwargs = simulation_default, 
                                        ensemble_definition = simulation_ensemble )
    
        # Create job file
        job_files.append( generate_job_file( destination_folder = lambda_folder, job_template = simulation_setup["system"]["paths"]["template"]["job_file"], 
                                             mdp_files = mdp_files, intial_coord = initial_coord[i], initial_topo = initial_topo,
                                             job_name = f'lambda_{i}', job_out = f"job_lambda_{i}.sh", initial_cpt = initial_cpt[i], 
                                             ensembles = ensembles ) )

    return job_files


def get_gauss_legendre_points_intermediates(a: float, b: float, no_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the Gauss-Legendre points and weights for numerical integration.

    Parameters:
        a (float): The lower bound of the integration interval.
        b (float): The upper bound of the integration interval.
        no_points (int): The number of points to generate.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the scaled points and weights.
    """
    # get gauss-legendre weights and points
    points, weights = roots_legendre(no_points)

    # Scale points and weights to the interval [a, b]
    points_scaled = 0.5 * (b - a) * points + 0.5 * (a + b)  

    # Scale weights to account for the interval change
    weights_scaled = 0.5 * (b - a) * weights

    return points_scaled, weights_scaled

def get_inital_intermediates( no_vdw: int, no_coul: int, precision: int=3 ):

    points, weights = get_gauss_legendre_points_intermediates( 0.0, 1.0, no_vdw )

    lambdas_vdw  = np.round(points, precision)

    points, weights = get_gauss_legendre_points_intermediates( 0.0, 1.0, no_coul )

    lambdas_coul = np.round(points, precision)

    combined_lambdas = np.concatenate( (lambdas_coul,lambdas_vdw+1) )

    return combined_lambdas.tolist()


def get_unified_lambdas( mbar: MBAR ) -> List[float]:
    """
    Return a list of unified lambdas.

    Parameters:
        mbar (MBAR): The MBAR object containing the states.

    Returns:
        List[float]: A list of unified lambdas, where lambdas coul range from 0 to 1 and lambdas vdw range from 1 to 2.
    """
    # lambdas coul between 0 and 1, lambdas vdw between 1 and 2
    lambdas_unified = [ sum(l) for l in mbar.states_ ]
	
    return lambdas_unified

def get_gridpoint_function( mbar: MBAR, precision: int=3 ) -> List[float]: 
    """
    Calculate the gridpoint function for a given MBAR estimator.

    Parameters:
        mbar (MBAR): The MBAR estimator object.
        precision (int, optional): The number of decimal places to round the gridpoint function values to. Default is 3.

    Returns:
        List[float]: The gridpoint function values.

    """
    gridpoint_function = []	

    gridpoint_function.append(0.0)

    for i in range( len(mbar.d_delta_f_) - 1 ):
        gridpoint_function.append( round( mbar.d_delta_f_.iloc[i,i+1] + gridpoint_function[i], precision ) )
    
    return gridpoint_function    


def change_number_of_states( mbar: MBAR, min_overlap: float=0.20, max_overlap: float=0.40 ) -> int: 
    """
    Determines the change in the number of lambda-states by analyzing the overlap matrix.

    Parameters:
        mbar (MBAR): The MBAR object containing the overlap matrix.
        min_overlap (float,optional): Define miminum overlap, after which one state is added. Defaults to 0.2.
        max_overlap (float,optional): Define maximum overlap, after which one state is removed. Defaults to 0.4.

    Returns:
        int: The change in the number of lambda-states. Possible values are -1, 0, or 1.

    """

    # Determines the change in the number of lambda-states by analysing the overlap matrix
    xi = 0 
    
    overlap_between_neighbours = []

    for i,row in enumerate(mbar.overlap_matrix[:-1]):
        if i == 0:
            overlap_between_neighbours.append( round(row[i+1],2) )
        else:
            overlap_between_neighbours.append( round(row[i-1],2) )
            overlap_between_neighbours.append( round(row[i+1],2) )

    if min(overlap_between_neighbours) < min_overlap:
        xi = 1
    elif max(overlap_between_neighbours) > max_overlap:
        xi = -1
    else:
        xi = 0

    # Print overlap matrix
    lines = []
    for row in mbar.overlap_matrix:
        lines.append( "  ".join( np.round(row,2).astype("str") ) )
    
    lines = "Overlap matrix:\n" + "\n".join( lines ) + "\n"

    logger.info( lines )
    
    return int(xi)

def adapted_distribution(lambdas: List[float], gridpoint_function: List[float], nstates: int) -> List[float]:
    """
    Calculates the adapted distribution of lambda-states based on a linear interpolation approach for equal partial uncertainties.

    Parameters:
    - lambdas (List[float]): A list of lambda values.
    - gridpoint_function (List[float]): A list of gridpoint function values.
    - nstates (int): The number of states.

    Returns:
    - adapted_distribution (List[float]): A list of adapted distribution values.

    Example:
    lambdas = [0.1, 0.5, 0.9]
    gridpoint_function = [1.0, 2.0, 3.0, 4.0, 5.0]
    nstates = 4

    adapted_distribution(lambdas, gridpoint_function, nstates)
    # Output: [0.1, 0.3, 0.7, 0.9]
    """

    # Defines new lambda-states (ranging from 0 to 2) by linear interpolation approach for equal partial uncertainties
    adapted_distribution = []
    average              = max(gridpoint_function)/(nstates-1)

    for i in range(nstates):		
        adapted_distribution.append( float( round( np.interp(i*average, gridpoint_function, lambdas),3 ) ) )

    return adapted_distribution

def restart_configuration(unified_oldlambdas: List[float], unified_newlambdas: List[float]) -> List[int]:
    """
    Returns the index for the best restart configuration for the new set of lambda-states.

    Parameters:
    - unified_oldlambdas (List[float]): A list of old lambda-states.
    - unified_newlambdas (List[float]): A list of new lambda-states.

    Returns:
    - List[int]: A list of indices representing the best restart configuration for the new set of lambda-states.

    """
     
    # Returns the index for the best restart configuration for the new set of lambda-states
    restart_configur = []  

    for i in range(len(unified_newlambdas)):
        restart_configur.append( int( round( np.interp(unified_newlambdas[i], unified_oldlambdas, range(len(unified_oldlambdas)) ), 0 ) ) )

    return restart_configur 


def convergence( mbar: MBAR ) -> float:
    """
    Returns the relative RMSD needed to determine convergence of the iteration procedure.

    Parameters:
    mbar (MBAR): The MBAR object containing the d_delta_f_ attribute.

    Returns:
    float: The relative RMSD value.

    """

    # Returns the relative RMSD needed to determine convergence of the iteration procedure
    partial_uncertainty = []
    txt                 = "Partial uncertanties:\n"

    for i in range( len(mbar.d_delta_f_) - 1 ):
        partial_uncertainty.append( mbar.d_delta_f_.iloc[i,i+1] )
        txt += f"{i+1} -> {i+2}: {mbar.d_delta_f_.iloc[i,i+1]:.4f}\t"
    
    logger.info(txt+"\n")

    RMSD_rel = np.std(partial_uncertainty, ddof=1) / np.average(partial_uncertainty)

    return RMSD_rel