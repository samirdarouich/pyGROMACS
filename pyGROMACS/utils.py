import os
import json
import subprocess
import numpy as np
import pandas as pd
from jinja2 import Template
from typing import List, Dict, Any


def generate_initial_configuration( destination_folder: str, coordinate_paths: List[str], no_molecules: List[int], box_lenghts: List[float], 
                                    build_intial_box: bool=True, initial_system: str="", n_try: int=10000, gmx_version: str="chem/gromacs/2022.4" ):
    """
    Generate initial configuration for molecular dynamics simulation with GRMOACS.

    Parameters:
    - destination_folder (str): The destination folder where the initial configurations will be saved.
    - coordinate_paths (List[str]): List of paths to coordinate files (GRO format) for each ensemble.
    - no_molecules (List[int]): List of number of molecules for each ensemble.
    - box_lengths (List[float]): List of box lengths for each ensemble. Provide [] if build_intial_box is false.
    - build_intial_box (bool, optional): If initial box needs to be builded with box_lenghts, otherwise start with initial_system. Defaulst to True.
    - initial_system (str, optional): Path to initial system, if initial system should be used to add molecules rather than new box. Defaults to "".
    - n_try (int, optional): Number of attempts to insert molecules. Defaults to 10000.
    - gmx_version (str, optional): Gromacs version to load.

    Returns:
    - intial_coord (str): Path of inital configuration

    """

    box_folder = f"{destination_folder}/box"

    # Create and the output folder of the box
    os.makedirs( box_folder, exist_ok = True )

    # Create bash script that builds the box using GROMACS
    bash_txt = f"# Bash script to generate GROMACS box. Automaticaly created by pyGROMACS.\n\nmodule purge\nmodule load {gmx_version}\n\n\ncd {box_folder}\n\n"

    for i,(gro,nmol) in enumerate( zip( coordinate_paths, no_molecules) ):
    
        if i == 0 and build_intial_box:
            bash_txt += " ".join(["gmx", "insert-molecules", "-ci", f"{gro}", "-nmol", f"{nmol}", "-box", *[str(l) for l in box_lenghts], "-o", f"temp{i}.gro"]) + "\n"
        elif i == 0:
            bash_txt += " ".join(["gmx", "insert-molecules", "-ci", f"{gro}", "-nmol", f"{nmol}", "-f", f"{initial_system}", "-try", f"{n_try}", "-o", f"temp{i}.gro"]) + "\n"
        elif i < (len( coordinate_paths ) - 1):
            bash_txt += " ".join(["gmx", "insert-molecules", "-ci", f"{gro}", "-nmol", f"{nmol}", "-f", f"temp{i-1}.gro", "-try", f"{n_try}", "-o", f"temp{i}.gro"]) + "\n"
        else:
            bash_txt += " ".join(["gmx", "insert-molecules", "-ci", f"{gro}", "-nmol", f"{nmol}", "-f", f"temp{i-1}.gro", "-try", f"{n_try}", "-o", "init_conf.gro"]) + "\n"
    
    # make sure that if only one molecule is added, the configuration has the correct name.
    if i == 0:
        bash_txt += f"mv temp{i}.gro init_conf.gro\n"
            
    with open( f"{box_folder}/build_box.sh", "w" ) as f:
        f.write( bash_txt )

    # Call the bash to build the box. Write GROMACS output to file.
    with open(f"{box_folder}/build_output.txt", "w") as f:
        subprocess.run(["bash", f"{box_folder}/build_box.sh"], stdout=f, stderr=f)
    
    intial_coord = f"{box_folder}/init_conf.gro"

    # Check if the system is build 
    if not os.path.isfile( intial_coord ):
        raise FileNotFoundError(f"Something went wrong during the box building! { intial_coord } not found.")

    return intial_coord


def generate_mdp_files( destination_folder: str, mdp_template: str, ensembles: List[str], temperature: float, pressure: float, compressibility: float,
                        ensemble_definition: Dict[str, Any|Dict[str, str|float]], simulation_times: List[float], dt: float, kwargs: Dict[str, Any]={} ):
    """
    Generate MDP files for simulation pipeline.

    Parameters:
    - destination_folder (str): The destination folder where the MDP files will be saved. Will be saved under destination_folder/0x_ensebmle/ensemble.mdp
    - mdp_template (str): The path to the MDP template file.
    - ensembles (List[str]): A list of ensembles to generate MDP files for.
    - temperature (float): The temperature for the simulation.
    - pressure (float): The pressure for the simulation.
    - compressibility (float): The compressibility for the simulation.
    - ensemble_definition (Dict[str, Any|Dict[str, str|float]]): Dictionary containing the ensemble settings for each ensemble.
    - simulation_times (List[float]): A list of simulation times (ns) for each ensemble.
    - dt (float): The time step for the simulation.
    - time_output (Dict[str, int]): A dictionary specifying the time output settings.
    - rcut (float): Cutoff distance.
    - constraints (Dict[str, str|int], optional): A dictionary specifying the constraint settings. Defaults to {}.
    - kwargs (Dict[str, Any], optional): Additional keyword arguments for the mdp. That should contain all default values. Defaults to {}.

    Raises:
    - KeyError: If an invalid ensemble is specified.

    Returns:
    - mdp_files (List[str]): List with paths of the mdp files

    """

    # Produce mdp files for simulation pipeline
    mdp_files = []
    for j,(ensemble,time) in enumerate( zip( ensembles, simulation_times ) ):
        
        try:
            ensemble_settings = ensemble_definition[ensemble]
        except:
            raise KeyError(f"Wrong ensemple specified: {ensemble}. Valid options are: {', '.join(ensemble_definition.keys())} ")
        
        # Add temperature of sim to ensemble settings
        if "t" in ensemble_settings.keys():
            ensemble_settings["t"]["ref_t"] = temperature
        
        # Add pressure and compressibility to ensemble settings
        if "p" in ensemble_settings.keys():
            ensemble_settings["p"].update( { "ref_p": pressure, "compressibility": compressibility } )

        # Simulation time is provided in nano seconds and dt in pico seconds, hence multiply with factor 1000
        kwargs["system"]["nsteps"] = int( 1000 * time / dt ) if not ensemble == "em" else int(time)
        kwargs["system"]["dt"]     = dt
        kwargs["restart"]          = "no" if ensemble == "em" or ensembles[j-1] == "em" else "yes"

        # Overwrite the ensemble settings
        kwargs["ensemble"]        = ensemble_settings

        # Provide a seed for tempearture generating:
        kwargs["seed"] = np.random.randint(0,1e5)
        
        # Open and fill template
        with open( mdp_template ) as f:
            template = Template( f.read() )

        rendered = template.render( ** kwargs ) 

        # Write template
        mdp_out = f"{destination_folder}/{'0' if j < 10 else ''}{j}_{ensemble}/{ensemble}.mdp"

        # Create the destination folder
        os.makedirs( os.path.dirname( mdp_out ), exist_ok = True )

        with open( mdp_out, "w" ) as f:
            f.write( rendered )
            
        mdp_files.append( mdp_out )

    return mdp_files

def generate_job_file( destination_folder: str, job_template: str, mdp_files: List[str], intial_coord: str, initial_topo: str, 
                       job_name: str, job_out: str="job.sh", ensembles: List[str]=[ "em", "nvt", "npt", "prod" ], initial_cpt: str="" ):
    """
    Generate initial job file for a set of simulation ensemble

    Parameters:
        destination_folder (str): Path to the destination folder where the job file will be created.
        job_template (str): Path to the job template file.
        mdp_files (List[List[str]]): List of lists containing the paths to the MDP files for each simulation phase.
        intial_coord (str): Path to the initial coordinate file.
        initial_topo (str): Path to the initial topology file.
        job_name (str): Name of the job.
        job_out (str, optional): Name of the job file. Defaults to "job.sh".
        ensembles (List[str], optional): List of simulation ensembles. Defaults to ["em", "nvt", "npt", "prod"].
        initial_cpt (str, optional): Path to the inital checkpoint file. Defaults to "".

    Returns:
        job_file (str): Path of job file

    Raises:
        FileNotFoundError: If the job template file does not exist.
        FileNotFoundError: If any of the MDP files does not exist.
        FileNotFoundError: If the initial coordinate file does not exist.
        FileNotFoundError: If the initial topology file does not exist.
        FileNotFoundError: If the initial checkpoint file does not exist.
    """

    # Check if job template file exists
    if not os.path.isfile( job_template ):
        raise FileNotFoundError(f"Job template file { job_template } not found.")

    # Check for mdp files
    for file in mdp_files:
        if not os.path.isfile( file ):
            raise FileNotFoundError(f"Mdp file { file  } not found.")
    
    # Check if topology file exists
    if not os.path.isfile( initial_topo ):
        raise FileNotFoundError(f"Topology file { initial_topo } not found.")

    # Check if coordinate file exists
    if not os.path.isfile( intial_coord ):
        raise FileNotFoundError(f"Coordinate file { intial_coord } not found.")
    
    # Check if checkpoint file exists
    if initial_cpt and not os.path.isfile( initial_cpt ):
        raise FileNotFoundError(f"Checkpoint file { initial_cpt } not found.")
    
    with open(job_template) as f:
        template = Template(f.read())

    job_file_settings = { "ensembles": { f"{'0' if j < 10 else ''}{j}_{step}": {} for j,step in enumerate(ensembles)} }
    ensemble_names    = list(job_file_settings["ensembles"].keys())

    # Create the simulation folder
    os.makedirs( destination_folder, exist_ok = True )

    # Relative paths for each mdp file for each simulation phase
    mdp_relative  = [ os.path.relpath( mdp_files[j], f"{destination_folder}/{step}" ) for j,step in enumerate(ensemble_names) ]

    # Relative paths for each coordinate file (for energy minimization use initial coodinates, otherwise use the preceeding output)
    cord_relative = [ f"../{ensemble_names[j-1]}/{ensembles[j-1]}.gro" if j > 0 else os.path.relpath( intial_coord, f"{destination_folder}/{step}" ) for j,step in enumerate(job_file_settings["ensembles"].keys()) ]

    # Relative paths for each checkpoint file 
    cpt_relative  = [ f"../{ensemble_names[j-1]}/{ensembles[j-1]}.cpt" if j > 0 else os.path.relpath( initial_cpt, f"{destination_folder}/{step}" ) if not "em" in step and initial_cpt else "" for j,step in enumerate(ensemble_names) ]

    # Relative paths for topology
    topo_relative = [ os.path.relpath( initial_topo, f"{destination_folder}/{step}" ) for j,step in enumerate(ensemble_names) ]

    # output file 
    out_relative  = [ f"{step}.tpr -maxwarn 10" for step in ensembles]

    for j,step in enumerate(ensemble_names):

        # If first or preceeding step is energy minimization, or if there is no cpt file to read in
        if ensembles[j-1]  == "em" or ensembles[j] == "em" or not cpt_relative[j]:
            job_file_settings["ensembles"][step]["grompp"] = f"-f {mdp_relative[j]} -c {cord_relative[j]} -p {topo_relative[j]} -o {out_relative[j]}"
        else:
            job_file_settings["ensembles"][step]["grompp"] = f"-f {mdp_relative[j]} -c {cord_relative[j]} -p {topo_relative[j]} -t {cpt_relative[j]} -o {out_relative[j]}"

        job_file_settings["ensembles"][step]["mdrun"]  = f"-deffnm {ensembles[j]}" 

    # Define LOG output
    log_path   = f"{destination_folder}/LOG"

    # Add to job file settings
    job_file_settings.update( { "job_name": job_name, "log_path": log_path, "working_path": destination_folder } )

    rendered = template.render( **job_file_settings )

    # Create the job folder
    job_file = f"{destination_folder}/{job_out}"

    os.makedirs( os.path.dirname( job_file ), exist_ok = True )

    # Write new job file
    with open( job_file, "w") as f:
        f.write( rendered )

    return job_file


## General utilities ##

def change_topo( topology_path: str, destination_folder: str, molecules_no_dict: Dict[str, int], file_name: str="topology.top" ):
    """
    Change the number of molecules in a topology file.

    Parameters:
    - topology_path (str): The path to the topology file.
    - molecules_no_dict (str): Dictionary with numbers and names of the molecules. If a molecule has the number None, then it will increase the initial topology number of that molecule by one.

    Returns:
    - topology_file (str): Destination of new topology file

    Description:
    This function reads the content of the topology file specified by 'topology_path' and searches for the section containing the number of molecules. 
    It then finds the line containing the molecule and changes the number to the specified one.

    Example:
    change_topo('topology.txt', {'water':5})
    """
    
    with open(topology_path) as f: 
        lines = [line for line in f]
    
    molecule_str_idx = [ i for i,line in enumerate(lines) if "[ molecules ]" in line and not line.startswith(";")][0]
    molecule_end_idx = [ i for i,line in enumerate(lines) if (line.startswith("[") or i == len(lines)-1) and i > molecule_str_idx ][0]

    for molecule,no_molecule in molecules_no_dict.items():

        for i,line in enumerate(lines[ molecule_str_idx : molecule_end_idx + 1 ]):

            if molecule in line:
                old = line.split()[1]
                new = str( no_molecule ) if no_molecule else str( int(old) + 1 )
                lines[i+molecule_str_idx] = lines[i+molecule_str_idx].replace( old, new )

    # Write new topology file
    os.makedirs( destination_folder, exist_ok = True )

    topology_file = f"{destination_folder}/{file_name}"

    with open(topology_file,"w") as f:
        f.writelines(lines)

    return topology_file

def read_gromacs_xvg(file_path, fraction = 0.7):
    """
    Reads data from a Gromacs XVG file and returns a pandas DataFrame.

    Parameters:
    - file_path (str): The path to the XVG file.
    - fraction (float, optional): The fraction of data to select. Defaults to 0.7.

    Returns:
    - pandas.DataFrame: A DataFrame containing the selected data.

    Description:
    This function reads data from a Gromacs XVG file specified by 'file_path'. It extracts the data columns and their corresponding properties from the file. The data is then filtered based on the 'fraction' parameter, selecting only the data points that are within the specified fraction of the maximum time value. The selected data is returned as a pandas DataFrame, where each column represents a property and each row represents a data point.

    Example:
    read_gromacs_xvg('data.xvg', fraction=0.5)
    """
    data = []
    properties = ["Time"]
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('@') and ("s" in line and "legend" in line):
                properties.append( line.split('"')[1] )
                continue
            elif line.startswith('@') or line.startswith('#'):
                continue  # Skip comments and metadata lines
            parts = line.split()
            data.append([float(part) for part in parts])  
    
    # Create column wise array with data
    data = np.array([np.array(column) for column in zip(*data)])

    # Only select data that is within (fraction,1)*t_max
    idx = data[0] > fraction * data[0][-1]

    return pd.DataFrame(dict(zip(properties, data[:,idx])))

def merge_nested_dicts(existing_dict, new_dict):
    """
    Function that merges nested dictionaries

    Args:
        existing_dict (Dict): Existing dictionary that will be merged with the new dictionary
        new_dict (Dict): New dictionary
    """
    for key, value in new_dict.items():
        if key in existing_dict and isinstance(existing_dict[key], dict) and isinstance(value, dict):
            # If both the existing and new values are dictionaries, merge them recursively
            merge_nested_dicts(existing_dict[key], value)
        else:
            # If the key doesn't exist in the existing dictionary or the values are not dictionaries, update the value
            existing_dict[key] = value

def work_json(file_path: str, data: Dict={}, to_do: str="read", indent: int=2):
    """
    Function to work with json files

    Args:
        file_path (string): Path to json file
        data (dict): If write is choosen, provide input dictionary
        to_do (string): Action to do, chose between "read", "write" and "append". Defaults to "read".

    Returns:
        data (dict): If read is choosen, returns dictionary
    """
    
    if to_do=="read":
        with open(file_path) as f:
            data = json.load(f)
        return data
    
    elif to_do=="write":
        with open(file_path,"w") as f:
            json.dump(data,f,indent=indent)

    elif to_do=="append":
        if not os.path.exists(file_path):
            with open(file_path,"w") as f:
                json.dump(data,f,indent=indent)
        else:
            with open(file_path) as f:
                current_data = json.load(f)
            merge_nested_dicts(current_data,data)
            with open(file_path,"w") as f:
                json.dump(current_data,f,indent=indent)
        
    else:
        raise KeyError("Wrong task defined: %s"%to_do)