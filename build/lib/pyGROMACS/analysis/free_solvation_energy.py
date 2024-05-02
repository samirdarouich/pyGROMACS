import glob
from alchemlyb.estimators import MBAR
from typing import List, Tuple, Dict, Any
from alchemlyb.parsing.gmx import extract_u_nk

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