import re
import os
import sys
import glob
import logging
import subprocess
import numpy as np
import pandas as pd

from alchemlyb.estimators import MBAR
from scipy.special import roots_legendre
from typing import List, Tuple, Dict, Any
from alchemlyb.parsing.gmx import extract_u_nk

logger = logging.getLogger("my_logger")

# General functions for automization
def submit_and_wait( job_files: List[str], submission_command: str="qsub"):
    
    job_list = []
    for job_file in job_files:
        # Submit job file
        exe = subprocess.run([submission_command,job_file],capture_output=True,text=True)
        job_list.append( exe.stdout.split("\n")[0].split()[-1] )

    logger.info("These are the submitted jobs:\n" + " ".join(job_list) + "\nWaiting until they are finished...")

    # Let python wait for the jobs to be finished (check job status every 1 min and if all jobs are done
    trackJobs( job_list, submission_command = submission_command)

    logger.info("\nJobs are finished! Continue with postprocessing\n")

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

def get_mbar( path: str, ensemble: str, temperature: float, pattern: str=r'lambda_(\d+)' ) -> MBAR:
    """
    Calculate the MBAR (Multistate Bennett Acceptance Ratio) estimator for a given set of free energy output files.

    Parameters:
        path (str): The path to the simulation folder.
        ensemble (str): The ensemble from which to extract. Should look like this xx_ensemble
        temperature (float): The temperature at which the simulation was performed.
        pattern (str, optional): The regular expression pattern used to extract the intermediate number from the file names. Defaults to r'lambda_(\d+)'.

    Returns:
        MBAR: The MBAR estimator object.

    Example:
        mbar = get_mbar(path='/path/to/simulation', production='00_prod', temperature=300.0, pattern=r'lambda_(\d+)')
    """

    # Seperatre the ensemble name to determine output files
    ensemble_name = "_".join(ensemble.split("_")[1:])
    
    # Search for all free energy output files within the simulation folder
    filelist = glob.glob(f"{path}/**/{ensemble}/{ensemble_name}.xvg", recursive=True)

    # Sort the paths ascendingly after the intermediate number
    filelist.sort( key=lambda path: int(re.search(pattern, path).group(1)) )

    # Extract the energy differences
    u_nk = pd.concat( [ extract_u_nk(xvg, T=temperature) for xvg in filelist ] )

    # Call the MBAR class
    mbar = MBAR().fit(u_nk)

    return mbar

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


def trackJobs(jobs, waittime=15, submission_command="qsub"):
    while len(jobs) != 0:
        for jobid in jobs:
            # SLURM command to check job status
            if submission_command == "qsub":
                x = subprocess.run(['qstat', jobid],capture_output=True,text=True)
                # Check wether the job is finished but is still shuting down
                try:
                    dummy = " C " in x.stdout.split("\n")[-2]
                except:
                    dummy = False
            # SBATCH command to check job status
            elif submission_command == "sbatch":
                x = subprocess.run(['scontrol', 'show', 'job', jobid], capture_output=True, text=True)
                # Check wether the job is finished but is still shuting down
                try:
                    dummy = "JobState=COMPLETING" in x.stdout.split("\n")[3] or "JobState=CANCELLED" in x.stdout.split("\n")[3] or "JobState=COMPLETED" in x.stdout.split("\n")[3]
                except:
                    dummy = False

            # If it's already done, then an error occur
            if dummy or x.stderr:
                jobs.remove(jobid)
                break
        os.system("sleep " + str(waittime))
    return