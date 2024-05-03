import re
import logging
import numpy as np
import pandas as pd

from typing import List
from scipy.constants import R
from pyLAMMPS.analysis import plot_data
from alchemlyb.estimators import BAR, MBAR, TI
from pyLAMMPS.analysis.solvation_free_energy import TI_spline
from alchemlyb.preprocessing import decorrelate_u_nk, decorrelate_dhdl
from alchemlyb.parsing.gmx import extract_u_nk, extract_dHdl, _get_headers, _extract_state

# Prevent alchemlyb correlation info to be printed to screen
logging.getLogger('alchemlyb').setLevel('WARNING')

def read_out_optimized_lambdas( log_file: str, pattern = "Best combined intermediates:" ):
    """
    This functions reads in the log file from the lambda optimization and returns the combined optimized lambdas

    Args:
        log_file (str): Log file of optimization script.
        pattern (str, optional): String pattern to search optimized lambdas. Defaults to "Best combined intermediates".

    Returns:
        combined_lambdas (List[float]): Optimized combined lambdas
    """
    combined_lambdas = []

    with open(log_file) as f:
        for line in f:
            if "Tolerance is not achieved" in line:
                print("!Warning, tolerance is not achieved, these are the best lambdas found yet!\n\n")
            if pattern in line:
                combined_lambdas = [ float(l) for l in re.search(rf'{pattern} (.*)$', line).group(1).split() ]
                break
    
    if combined_lambdas:
        return combined_lambdas
    else:
        raise KeyError(f"Something went wrong during optimization! No optimized lambdas found!" )

def extract_current_state( file: str ):
    headers = _get_headers(file)
    state, lambdas, statevec = _extract_state(file, headers)
    return state,lambdas,statevec

def extract_combined_states( fep_files: List[str] ):
    """
    Function that takes a list of paths to GROMACS fep output files, sort them after copy and lambda, and then extract from the first copy the combined state vector.

    Parameters:
     - fep_files (List[str]): List of paths to GROMACS fep output files

    Returns:
     - combined_lambdas (List[float]): List with lambda states
    """
    
    copy_pattern = re.compile( r'copy_(\d+)')
    lambda_pattern = re.compile( r'lambda_(\d+)')

    fep_files.sort( key=lambda x: ( int(copy_pattern.search(x).group(1)), 
                                    int(lambda_pattern.search(x).group(1))
                                  )
                   )

    unique_copy = [ file for file in fep_files if "copy_0" in file ]
    combined_states = [ sum( extract_current_state(file)[2] ) for file in unique_copy ] 

    return combined_states

def get_free_energy_difference( fep_files: List[str], T: float, method: str="MBAR", fraction: float=0.0, 
                                decorrelate: bool=True, coupling: bool=True  ):
    """
    Calculate the free energy difference using different methods.

    Parameters:
     - fep_files (List[str]): A list of file paths to the FEP output files.
     - T (float): The temperature in Kelvin.
     - method (str, optional): The method to use for estimating the free energy difference. Defaults to "MBAR".
     - fraction (float, optional): The fraction of data to be discarded from the beginning of the simulation. Defaults to 0.0.
     - decorrelate (bool, optional): Whether to decorrelate the data before estimating the free energy difference. Defaults to True.
     - coupling (bool, optional): If coupling (True) or decoupling (False) is performed. If decoupling, 
                                  multiply free energy results *-1 to get solvation free energy. Defaults to True.
    Returns:
     - df (pd.DataFrame): Pandas dataframe with mean, std and unit of the free energy difference

    Raises:
     - KeyError: If the specified method is not implemented.

    Notes:
     - The function supports the following methods: "MBAR", "BAR", "TI", and "TI_spline".
     - The function uses the 'extract_u_nk' function for "MBAR" and "BAR" methods, and the 'extract_dUdl' function for "TI" and "TI_spline" methods.
     - The function concatenates the data from all FEP output files into a single DataFrame.
     - The function fits the free energy estimator using the combined DataFrame.
     - The function extracts the mean and standard deviation of the free energy difference from the fitted estimator.

    """

    if method in [ "MBAR","BAR" ]:
        # Get combined df for all lambda states
        combined_df = []
        for file in fep_files:
            # Extraxt data
            df = extract_u_nk( file, T = T )

            # Discard everything from start to fraction
            idx = df.index.get_level_values("time").values > df.index.get_level_values("time").values.max()*fraction
            df = df[idx]

            if decorrelate:
                # Decorrelate data
                df = decorrelate_u_nk( df )

            # Append to overall list
            combined_df.append( df )
        
        combined_df = pd.concat( combined_df )
        
        # Get free energy estimator
        FE = MBAR() if method == "MBAR" else BAR()

    elif method in [ "TI", "TI_spline" ]:
        # Get combined df for all lambda states
        combined_df = []
        for file in fep_files:
            # Extraxt data
            df = extract_dHdl( file, T = T )

            # Discard everything from start to fraction
            idx = df.index.get_level_values("time").values > df.index.get_level_values("time").values.max()*fraction
            df = df[idx]

            if decorrelate:
                # Decorrelate data
                df = decorrelate_dhdl( df )

            # Append to overall list
            combined_df.append( df )

        combined_df = pd.concat( combined_df )

        # Get free energy estimator
        FE = TI() if method == "TI" else TI_spline()

    else:
        raise KeyError(f"Specified free energy method '{method}' is not implemented. Available are: 'MBAR', 'BAR', 'TI' or 'TI_spline' ")
    
    # Get free energy difference
    FE.fit( combined_df )
    
    # Extract mean and std
    mean, std = FE.delta_f_.iloc[0,-1], FE.d_delta_f_.iloc[0,-1]

    # In case decoupling is performed, negate the value to get solvation free energy
    if not coupling:
        mean *= -1

    # BAR only provides the standard deviation from adjacent intermediates. Hence, to get the global std propagate the error
    if method == "BAR":
        std = np.sqrt( ( np.array( [ FE.d_delta_f_.iloc[i, i+1] for i in range(FE.d_delta_f_.shape[0]-1) ] )**2 ).sums() )

    # Convert from dimensionless to kJ/mol
    df = pd.DataFrame( { "property": "solvation_free_energy", "mean": mean * R * T / 1000, "std": std * R * T / 1000, "unit": "kJ/mol" }, index = [0] )

    return df

def visualize_dudl( fep_files: List[str], T: float, 
                    fraction: float=0.0, decorrelate: bool=True,
                    save_path: str=""  
                  ):
    
    # Get combined df for all lambda states
    combined_df = []
    for file in fep_files:
        # Extraxt data
        df = extract_dHdl( file, T = T )

        # Discard everything from start to fraction
        idx = df.index.get_level_values("time").values > df.index.get_level_values("time").values.max()*fraction
        df = df[idx]

        if decorrelate:
            # Decorrelate data
            df = decorrelate_dhdl( df )

        # Append to overall list
        combined_df.append( df )

    combined_df = pd.concat( combined_df )
   
    # Extract vdW and Coulomb portion
    vdw_dudl = combined_df.groupby("vdw-lambda")["vdw"].agg(["mean","std"])
    coul_dudl = combined_df.groupby("coul-lambda")["coul"].agg(["mean","std"])
    
    # Plot vdW part
    datas = [ [ vdw_dudl.index.values, vdw_dudl["mean"].values, None, vdw_dudl["std"].values ] ]
    set_kwargs = { "xlabel": "$\lambda_\mathrm{vdW}$",
                   "ylabel": "$ \\langle \\frac{\partial U}{\partial \lambda} \\rangle_{\lambda_{\mathrm{vdW}}} \ / \ (k_\mathrm{B}T)$",
                   "xlim": (0,1)
                 }
    plot_data( datas, save_path = f"{save_path}/dudl_vdw.png", set_kwargs = set_kwargs ) 

    # Plot Coulomb part
    datas = [ [ coul_dudl.index.values, coul_dudl["mean"].values, None, coul_dudl["std"].values ] ]
    set_kwargs = { "xlabel": "$\lambda_\mathrm{coul}$",
                   "ylabel": "$ \\langle \\frac{\partial U}{\partial \lambda} \\rangle_{\lambda_{\mathrm{coul}}} \ / \ (k_\mathrm{B}T)$",
                   "xlim": (0,1)
                 }
    plot_data( datas, save_path = f"{save_path}/dudl_coul.png", set_kwargs = set_kwargs ) 