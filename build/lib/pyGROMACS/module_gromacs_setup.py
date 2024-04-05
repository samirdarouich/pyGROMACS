import os
import re
import yaml
import glob
import subprocess
import numpy as np
import pandas as pd
import multiprocessing
from jinja2 import Template
from typing import List, Dict, Any

from .utils import ( generate_initial_configuration, generate_mdp_files, generate_job_file, 
                     change_topo, read_gromacs_xvg, work_json, merge_nested_dicts )
from .utils_automated import get_mbar, submit_and_wait

def extract_function(file):
    subprocess.run(["bash",file], capture_output=True)

class GROMACS_setup:

    def __init__( self, system_setup: str, simulation_default: str, simulation_ensemble: str, submission_command: str="qsub" ) -> None:
        """
        Initialize a new instance of the gromacs class.

        Parameters:
         - system_setup (str): Path to the system setup YAML file. Containing all system settings.
         - simulation_default (str): Path to the simulation default YAML file. Containing all default GROMACS settings.
         - simulation_ensemble (str): Path to the simulation ensemble YAML file. Containing all GROMACS ensemble settings (except temperature, pressure, and compressibility).
         - submission_command (str, optional): Command to submit jobs to cluster. Defaults to "qsub".
        
        Returns:
            None
        """
        # Save paths of the yaml files.
        self.system_setup_path        = os.path.abspath( system_setup )
        self.simulation_default_path  = os.path.abspath( simulation_default )
        self.simulation_ensemble_path = os.path.abspath( simulation_ensemble )
        self.submission_command       = submission_command

        with open( system_setup ) as file: 
            self.system_setup    = yaml.safe_load(file)
        
        with open( simulation_default ) as file:
            self.simulation_default  = yaml.safe_load(file)

        with open( simulation_ensemble ) as file:
            self.simulation_ensemble = yaml.safe_load(file)

        # Create an analysis dictionary containing all files
        self.analysis_dictionary = {}

    def prepare_simulation(self, ensembles: List[str], simulation_times: List[float], initial_systems: List[str]=[], 
                           copies: int=0, folder_name: str="md", mdp_kwargs: Dict[str, Any]={}, 
                           on_cluster: bool=False, off_set: int=0):
        """
        Prepares the simulation by generating job files for each temperature, pressure, and compressibility combination specified in the simulation setup.
        The method checks if an initial configuration file is provided. If not, it generates the initial configuration based on the provided molecule numbers and coordinate files. 
        It then generates MDP files for each ensemble in a separate folder and creates a job file for each copy of the simulation.

        Parameters:
         - ensembles (List[str]): A list of ensembles to generate MDP files for. Definitions of each ensemble is provided in self.simulation_ensemble.
         - simulation_times (List[float]): A list of simulation times (ns) for each ensemble.
         - initial_systems (List[str]): A list of initial system .gro files to be used for each temperature and pressure state.
         - copies (int, optional): Number of copies for the specified system. Defaults to 0.
         - folder_name (str, optional): Name of the subfolder where to perform the simulations. Defaults to "md".
                                        Path structure is as follows: system.folder/system.name/folder_name
         - mdp_kwargs (Dict[str, Any], optional): Further kwargs that are parsed to the mdp template. Defaults to "{}".
         - on_cluster (bool, optional): If the GROMACS build should be submited to the cluster. Defaults to "False".
         - off_set (int, optional): First ensemble starts with 0{off_set}_ensemble. Defaulst to 0.

        Returns:
            None
        """
        self.job_files = []

        # Define simulation folder
        sim_folder = f'{self.system_setup["folder"]}/{self.system_setup["name"]}/{folder_name}'


        # Create initial configuration in (sim_folder/box)
        if not initial_systems:
            print("\nBuilding system based on provided molecule numbers and coordinate files!\n" )
            initial_coord = generate_initial_configuration( build_template = self.system_setup["paths"]["template"]["build_system_file"],
                                                            destination_folder = sim_folder, coordinate_paths = self.system_setup["paths"]["gro"], 
                                                            molecules_no_dict = self.system_setup["molecules"], box_lenghts = self.system_setup["box"],
                                                            submission_command = self.submission_command, on_cluster = on_cluster )
            initial_systems = [initial_coord]*len(self.system_setup["temperature"])
            flag_cpt = False
        else:
            print(f"\nSystems already build and initial configurations are provided at:\n\n" + "\n".join(initial_systems) + "\n" )
            flag_cpt = all( os.path.exists( initial_system.replace( initial_system.split(".")[-1], "cpt") ) for initial_system in initial_systems )
            if flag_cpt:
                print(f"Checkpoint files (.cpt) are provided in the same folder.\n")

        # Copy tolopoly to the box folder and change it according to system specification
        initial_topo = change_topo( topology_path = self.system_setup["paths"]["topol"], destination_folder = f"{sim_folder}/box", 
                                    molecules_no_dict = self.system_setup["molecules"],
                                    system_name = self.system_setup["name"],
                                    file_name = f'topology_{self.system_setup["name"]}.top' )

        for initial_system, temperature, pressure, compressibility in zip( initial_systems, 
                                                                           self.system_setup["temperature"], 
                                                                           self.system_setup["pressure"], 
                                                                           self.system_setup["compressibility"]  ):
            
            job_files = []

            # If system is already build assume the cpt file is in the same folder as the coordinates.
            initial_cpt = initial_system.replace( initial_system.split(".")[-1], "cpt") if flag_cpt else ""
            
            # Define folder for specific temp and pressure state, as well as for each copy
            for copy in range( copies + 1 ):
                copy_folder = f"{sim_folder}/temp_{temperature:.0f}_pres_{pressure:.0f}/copy_{copy}"

                # Produce mdp files (for each ensemble an own folder 0x_ensemble)
                mdp_files = generate_mdp_files( destination_folder = copy_folder, mdp_template = self.system_setup["paths"]["template"]["mdp_file"],
                                                ensembles = ensembles, temperature = temperature, pressure = pressure, 
                                                compressibility = compressibility, simulation_times = simulation_times,
                                                dt = self.simulation_default["system"]["dt"], kwargs = { **mdp_kwargs, **self.simulation_default }, 
                                                ensemble_definition = self.simulation_ensemble, off_set = off_set )
                
                # Create job file
                job_files.append( generate_job_file( destination_folder = copy_folder, job_template = self.system_setup["paths"]["template"]["job_file"], mdp_files = mdp_files, 
                                                     intial_coord = initial_system, initial_topo = initial_topo, job_name = f'{self.system_setup["name"]}_{temperature:.0f}',
                                                     job_out = f"job_{temperature:.0f}.sh", ensembles = ensembles, initial_cpt = initial_cpt, off_set = off_set ) )
            self.job_files.append( job_files )

    def add_guest_molecule_and_prepare_equilibrate(self, solute: str, solute_coordinate: str, initial_systems: List[str], ensembles: List[str], 
                                                   simulation_times: List[float], copies: int=0, folder_name: str="free_energy",
                                                   on_cluster: bool=False):
        """
        Function that adds a guest molecule to the system and prepares it for equilibration simulations.

        Parameters:
         - solute (str): The name of the guest molecule.
         - solute_coordinate (str): The path to the coordinate file of the guest molecule.
         - initial_systems (List[str]): A list of initial system .gro files to be used to insert the guest molecule for each temperature and pressure state.
         - ensembles (List[str]): A list of ensembles to generate MDP files for. Definitions of each ensemble is provided in self.simulation_ensemble.
         - simulation_times (List[float]): A list of simulation times (ns) for each ensemble.
         - copies (int, optional): Number of copies for the specified system. Defaults to 0.
         - folder_name (str, optional): The name of the folder where the simulations will be performed. Subfolder call "equilibration" will be created there. Defaults to "free_energy".
         - on_cluster (bool, optional): If the GROMACS build should be submited to the cluster. Defaults to "False".

        Returns:
            None

        The method creates a new folder for the simulation of the guest molecule in the specified subfolder. It copies the initial topology file to the new folder
        and modifies it by increasing the number of the guest molecule by one.
        """
        self.job_files = []

        # Define simulation folder
        sim_folder = f'{self.system_setup["folder"]}/{self.system_setup["name"]}/{folder_name}/{solute}'

        # Change initial topology by increasing the number of the solute by one.
        initial_topo = change_topo( topology_path = self.system_setup["paths"]["topol"], destination_folder = f"{sim_folder}/box", 
                                    molecules_no_dict = { **self.system_setup["molecules"], solute: -1},
                                    system_name = self.system_setup["name"],
                                    file_name = f'topology_{self.system_setup["name"]}.top' )
        
        # If no initial systems are provided, use a empty string for each temparature & pressure state
        if not initial_systems:
            initial_systems = [ "" for _ in self.system_setup["temperature"] ]

        # Check if checkpoint files are provided with intial systems
        flag_cpt = all( os.path.exists( initial_system.replace( initial_system.split(".")[-1], "cpt") ) for initial_system in initial_systems )
        if flag_cpt:
            print(f"Checkpoint files (.cpt) are provided in the same folder as initial coordinates.\n")
        
        # Prepare equilibration of new system for each temperature/pressure state
        for initial_system, temperature, pressure, compressibility in zip( initial_systems, 
                                                                           self.system_setup["temperature"], 
                                                                           self.system_setup["pressure"], 
                                                                           self.system_setup["compressibility"]  ):
            job_files = []

            # Check if inital system is provided, if thats not the case, build new system with one solute more
            molecules_no_dict = { solute: 1 } if initial_system else  { **self.system_setup["molecules"], solute: 1}
            coordinate_paths  = [ solute_coordinate ] if initial_system else [*self.system_setup["paths"]["gro"], solute_coordinate ]

            # Genereate initial box with solute ( a equilibrated structure is provided for every temperature & pressure state )
            box_folder = f"{sim_folder}/equilibration/temp_{temperature:.0f}_pres_{pressure:.0f}"
            initial_coord = generate_initial_configuration( build_template = self.system_setup["paths"]["template"]["build_system_file"],
                                                            destination_folder = box_folder, coordinate_paths = coordinate_paths, 
                                                            molecules_no_dict = molecules_no_dict, box_lenghts = self.system_setup["box"], 
                                                            initial_system = initial_system, on_cluster = on_cluster,
                                                            submission_command = self.submission_command )
        
            # Assume that the cpt file is in the same folder as the coordinates.
            initial_cpt = initial_system.replace( initial_system.split(".")[-1], "cpt") if flag_cpt else ""

            # Define folder for specific temp and pressure state, as well as for each copy
            for copy in range( copies + 1 ):
                copy_folder = f"{box_folder}/copy_{copy}"

                # Produce mdp files (for each ensemble an own folder 0x_ensemble)
                mdp_files = generate_mdp_files( destination_folder = copy_folder, mdp_template = self.system_setup["paths"]["template"]["mdp_file"],
                                                ensembles =ensembles, temperature = temperature, pressure = pressure, 
                                                compressibility = compressibility, simulation_times = simulation_times,
                                                dt = self.simulation_default["system"]["dt"], kwargs = self.simulation_default, 
                                                ensemble_definition = self.simulation_ensemble )
                
                # Create job file
                job_files.append( generate_job_file( destination_folder = copy_folder, job_template = self.system_setup["paths"]["template"]["job_file"], mdp_files = mdp_files, 
                                                    intial_coord = initial_coord, initial_topo = initial_topo, job_name = f'{self.system_setup["name"]}_{temperature:.0f}_{solute}_eq',
                                                    job_out = f"job_{temperature:.0f}.sh", ensembles = ensembles, initial_cpt = initial_cpt ) )
                
            self.job_files.append( job_files )

    
    def prepare_free_energy_simulation( self, simulation_free_energy: str, solute: str, combined_lambdas: List[float], initial_systems: List[str], ensembles: List[str], 
                                        simulation_times: List[float], copies: int=0, folder_name: str="free_energy", precision: int=3, flag_cpt: bool=True  ):
        """
        Function that prepares free energy simulations for several temperatures and pressure states.

        Parameters:
         - simulation_free_energy (str): Path to the simulation free energy YAML file. Containing all free energy settings.
         - combined_lambdas (List[float]): List of combined lambdas for the free energy simulations.
         - initial_systems (List[str]): A list of initial system .gro files to be used for each temperature and pressure state.
         - ensembles (List[str]): A list of ensembles to generate MDP files for. Definitions of each ensemble is provided in self.simulation_ensemble.
         - simulation_times (List[float]): A list of simulation times (ns) for each ensemble.
         - copies (int, optional): Number of copies for the specified system. Defaults to 0.
         - folder_name (str, optional): Name of the subfolder where to perform the simulations. Defaults to "free_energy".
                                        Path structure is as follows: system.folder/system.name/folder_name
         - precision (int, optional): The number of decimals of the lambdas. Defaults to 3.
         - flag_cpt (bool, optional): If checkpoint files are provided in the same folder as the inital systems. Otherwise dont use checkpoint files.
        
        Returns:
            None

        The method loads the free energy settings from the provided YAML file and overwrites the provided lambdas with the combined lambdas.
        For each temperature, pressure, and compressibility combination specified in the simulation setup, the method prepares a separate folder for each copy. 
        Within each copy folder, it creates a folder for each lambda value. It then generates MDP files for each ensemble in the lambda folder and creates a job file for each copy of the simulation.

        """
        self.job_files = []

        # Define simulation folder
        sim_folder = f'{self.system_setup["folder"]}/{self.system_setup["name"]}/{folder_name}/{solute}'

        # Load free energy settings
        with open( simulation_free_energy ) as file: 
            simulation_free_energy = yaml.safe_load(file)

        # Overwrite provided lambdas in free energy settings
        simulation_free_energy.update( { "init_lambda_states": "".join([f"{x:.0f}" + " "*(precision+2) if x < 10 else f"{x:.0f}" + " "*(precision+1) for x,_ in enumerate(combined_lambdas)]), 
                                         "vdw_lambdas": " ".join( [ f"{max(l-1,0.0):.{precision}f}" for l in combined_lambdas] ), 
                                         "coul_lambdas": " ".join( [ f"{min(l,1.0):.{precision}f}" for l in combined_lambdas] ),
                                         "couple_moltype": solute } )
        
        # Add free energy settings to overall simulation input
        self.simulation_default["free_energy"] = simulation_free_energy

        # Change initial topology by increasing the number of the solute by one.
        initial_topo = change_topo( topology_path = self.system_setup["paths"]["topol"], destination_folder = f"{sim_folder}/box", 
                                    molecules_no_dict = { **self.system_setup["molecules"], solute: -1},
                                    system_name = self.system_setup["name"],
                                    file_name = f'topology_{self.system_setup["name"]}.top' )

        # Prepare free energy simulations for several temperatures & pressure states
        for initial_system, temperature, pressure, compressibility in zip( initial_systems, 
                                                                           self.system_setup["temperature"], 
                                                                           self.system_setup["pressure"], 
                                                                           self.system_setup["compressibility"]  ):
            
            job_files = []

            # Assume that the cpt file is in the same folder as the coordinates.
            initial_cpt = initial_system.replace( initial_system.split(".")[-1], "cpt") if flag_cpt else ""

            # Define folder for specific temp and pressure state, as well as for each copy
            for copy in range( copies + 1 ):
                copy_folder = f"{sim_folder}/temp_{temperature:.0f}_pres_{pressure:.0f}/copy_{copy}"

                # Define for each lambda an own folder
                for i,_ in enumerate(combined_lambdas):
                    self.simulation_default["free_energy"]["init_lambda_state"] = i

                    lambda_folder = f"{copy_folder}/lambda_{i}"

                    # Produce mdp files (for each ensemble an own folder 0x_ensemble)
                    mdp_files = generate_mdp_files( destination_folder = lambda_folder, mdp_template = self.system_setup["paths"]["template"]["mdp_file"],
                                                    ensembles = ensembles, temperature = temperature, pressure = pressure, 
                                                    compressibility = compressibility, simulation_times = simulation_times,
                                                    dt = self.simulation_default["system"]["dt"], kwargs = self.simulation_default, 
                                                    ensemble_definition = self.simulation_ensemble )
                
                    # Create job file
                    job_files.append( generate_job_file( destination_folder = lambda_folder, job_template = self.system_setup["paths"]["template"]["job_file"], 
                                                         mdp_files = mdp_files, intial_coord = initial_system, initial_topo = initial_topo,
                                                         job_name = f'{self.system_setup["name"]}_{temperature:.0f}_lambda_{i}', job_out = f"job_{temperature:.0f}_lambda_{i}.sh",
                                                         ensembles = ensembles, initial_cpt = initial_cpt ) )
            
            self.job_files.append( job_files )

    def optimize_intermediates( self, simulation_free_energy: str, solute: str, initial_coord: str, initial_cpt: str, 
                                initial_topo: str, iteration_time: float, temperature: float, pressure: float, compressibility: float,
                                precision: int=3, tolerance: float=0.05, min_overlap: float=0.15, max_overlap: float=0.25, 
                                max_iterations: int=20, folder_name: str="free_energy"):
        """
        Function for optimizing solvation free energy intermediates using the decoupling approach. This wiöl

        Parameters:
            simulation_free_energy (str): Path to the file containing simulation free energy data.
            initial_coord (str): Path to the initial coordinates file.
            initial_cpt (str): Path to the initial checkpoint file.
            initial_topo (str): Path to the intial topology file.
            iteration_time (float): Time (ns) allocated for each optimization iteration.
            temperature (float): System temperature with which the optimization is performed.
            pressure (float): System pressure with which the optimization is performed.
            compressibility (float): System compressibility with which the optimization is performed.
            precision (int, optional): Number of decimal places for precision. Default is 3.
            tolerance (float, optional): Tolerance level for optimization convergence. Default is 0.05.
            min_overlap (float, optional): Minimum overlap value for optimization. Default is 0.15.
            max_overlap (float, optional): Maximum overlap value for optimization. Default is 0.25.
            max_iterations (int, optional): Maximum number of optimization iterations. Default is 20.
            folder_name (str, optional): Name of the folder for optimization results. Simulations will be performed in a subfolder "optimization". Default is "free_energy".

        Returns:
            None
        """

        if not os.path.exists(self.system_setup["paths"]["template"]["optimize_lambda_file"]):
            raise FileExistsError(f'Provided extract template does not exists: {self.system_setup["paths"]["template"]["optimize_lambda_file"]}')
        else:
            # Open template
            with open(self.system_setup["paths"]["template"]["optimize_lambda_file"]) as f:
                template = Template( f.read() )

        # Define simulation folder
        sim_folder = f'{self.system_setup["folder"]}/{self.system_setup["name"]}/{folder_name}/{solute}/optimization'
        
        paths = { "simulation_folder": sim_folder, "initial_coord": initial_coord, "initial_cpt": initial_cpt, "initial_topo": initial_topo,
                  "parameter": { "setup": self.system_setup_path, "default": self.simulation_default_path, 
                                 "ensemble": (self.simulation_ensemble_path), "free_energy": os.path.abspath(simulation_free_energy) } 
                }
        
        settings_dict = { "paths" : paths, "iteration_time": iteration_time, "precision": precision, "tolerance": tolerance, "min_overlap": min_overlap,
                          "max_overlap": max_overlap, "max_iterations": max_iterations, "log_path":f"{sim_folder}/LOG_optimize_intermediates",
                          "temperature": temperature, "pressure": pressure, "compressibility": compressibility, "solute": solute  }
        
        rendered = template.render( **settings_dict )

        job_file = f"{sim_folder}/optimize_intermediates.py"
        os.makedirs( os.path.dirname( job_file ), exist_ok = True )
        with open( job_file, "w" ) as f:
            f.write( rendered )

        print(f"Submitting automized intermediate optimization job: {job_file}")
        subprocess.run( [self.submission_command, job_file] )
        print("\n")

    def submit_simulation(self):
        """
        Function that submits predefined jobs to the cluster.
        
        Parameters:
            None

        Returns:
            None
        """
        for temperature, pressure, job_files in zip( self.system_setup["temperature"], self.system_setup["pressure"], self.job_files ):
            print(f"\nSubmitting simulations at Temperature = {temperature:.0f} K, Pressure = {pressure:.0f} bar\n")

            for job_file in job_files:
                print(f"Submitting job: {job_file}")
                subprocess.run( [self.submission_command, job_file] )
                print("\n")

    def analysis_extract_properties( self, analysis_folder: str, ensemble: str, extracted_properties: List[str], command: str="energy", fraction: float=0.0, 
                                     args: List[str]=[""], output_name: str="properties", on_cluster: bool=False, extract: bool= True ):
        """
        Extracts properties from GROMACS output files for a specific ensemble.

        Parameters:
            analysis_folder (str): The name of the folder where the analysis will be performed.
            ensemble (str): The name of the ensemble for which properties will be extracted. Should be xx_ensemble.
            extracted_properties (List[str]): A list of properties to be extracted from the GROMACS output files.
            command (str, optional): The GROMACS command to be used for property extraction. Defaults to "energy".
            fraction (float, optional): The fraction of data to be discarded from the beginning of the simulation. Defaults to 0.0.
            args (List[str], optional): A list of strings that will be added to the gmx command after "-f ... -s ..." and before "-o ...". Defaulst to [""]
            output_name ( str, optional): Name of the resulting xvg output file. Defaults to "properties".
            on_cluster (bool, optional): If the GROMACS analysis should be submited to the cluster. Defaults to "False".
            extract (bool, optional): If the values are already extracted from GROMACS and only the xvg files should be read out, 
                                      this should be set to False. Defaults to "True".

        Returns:
            None

        The method searches for output files in the specified analysis folder that match the given ensemble.
        For each group of files with the same temperature and pressure, the properties are extracted using the specified GROMACS command.
        The extracted properties are then averaged over all copies and the mean and standard deviation are calculated.
        The averaged values and the extracted data for each copy are saved as a JSON file in the destination folder.
        The extracted values are also added to the class's analysis dictionary.
        """
        
        if not os.path.exists(self.system_setup["paths"]["template"]["extract_property_file"]):
            raise FileExistsError(f'Provided extract template does not exists: {self.system_setup["paths"]["template"]["extract_property_file"]}')
        else:
            with open(self.system_setup["paths"]["template"]["extract_property_file"]) as f:
                template = Template(f.read())
                
        # Define folder for analysis
        sim_folder = f'{self.system_setup["folder"]}/{self.system_setup["name"]}/{analysis_folder}'

        # Seperatre the ensemble name to determine output files
        ensemble_name = "_".join(ensemble.split("_")[1:])

        if extract:
            bash_files = []

            # Search output files and sort them after temperature / pressure and then copy
            for temperature, pressure in zip( self.system_setup["temperature"], 
                                              self.system_setup["pressure"]
                                            ):
                
                # Define folder for specific temp and pressure state
                state_folder = f"{sim_folder}/temp_{temperature:.0f}_pres_{pressure:.0f}"

                # Search for available copies
                files = glob.glob( f"{state_folder}/copy_*/{ensemble}/{ensemble_name}.edr" )
                files.sort( key=lambda x: int(re.search(r'copy_(\d+)', x).group(1)) ) 

                if len(files) == 0:
                    raise KeyError(f"No files found machting the ensemble: {ensemble} in folder\n:   {sim_folder}")
            
                # Iterate through the files and write extraction bash file
                for path in files:
                    
                    rendered = template.render( folder = os.path.dirname( path ),
                                                extracted_properties = extracted_properties,
                                                gmx_command = f"{command} -f {ensemble_name} -s {ensemble_name} {' '.join(args)} -o {output_name}" )

                    bash_file = f"{os.path.dirname( path )}/extract_properties.sh"

                    with open( bash_file, "w") as f: 
                        f.write( rendered )

                    bash_files.append( bash_file )
        
        if on_cluster and extract:
            print( "Submit extraction to cluster:\n" )
            print( '\n'.join(bash_files), "\n" )
            print( "Wait until extraction is done..." )
            submit_and_wait( bash_files, submission_command = self.submission_command )
        elif not on_cluster and extract:
            print("Extract locally\n")
            print( '\n'.join(bash_files), "\n" )
            print( "Wait until extraction is done..." )
            num_processes = multiprocessing.cpu_count()
            # In case there are multiple CPU's, leave one without task
            if num_processes > 1:
                num_processes -= 1

            # Create a pool of processes
            pool = multiprocessing.Pool( processes = multiprocessing.cpu_count()  )

            # Execute the tasks in parallel
            data_list = pool.map(extract_function, bash_files)

            # Close the pool to free up resources
            pool.close()
            pool.join()

        if extract:
            print( f"Extraction finished!\n\n" )

        for temperature, pressure in zip( self.system_setup["temperature"], 
                                          self.system_setup["pressure"]
                                        ):
            print(f"Temperature: {temperature}, Pressure: {pressure}\n   ")
            
            # Define folder for specific temp and pressure state
            state_folder = f"{sim_folder}/temp_{temperature:.0f}_pres_{pressure:.0f}"

            files = glob.glob( f"{state_folder}/copy_*/{ensemble}/{output_name}.xvg" )
            files.sort( key=lambda x: int(re.search(r'copy_(\d+)', x).group(1)) ) 

            if len(files) == 0:
                raise FileExistsError("Extracting the properties did not work for:\n   " + "\n   ".join(files) + "\n")

            data_list = [ read_gromacs_xvg( file_path = file, fraction = fraction) for file in files ]

            # Mean the values for each copy and exctract mean and standard deviation
            mean_std_list  = [df.iloc[:, 1:].agg(['mean', 'std']).T.reset_index().rename(columns={'index': 'property'}) for df in data_list]

            # Extract units from the property column and remove it from the label and make an own unit column
            for df in mean_std_list:
                df['unit']     = df['property'].str.extract(r'\((.*?)\)')
                df['property'] = [ p.split('(')[0].replace(" ","") for p in df['property'] ]

            final_df           = pd.concat(mean_std_list,axis=0).groupby("property", sort=False)["mean"].agg(["mean","std"]).reset_index()
            final_df["unit"]   = df["unit"]

            # in case only system is there, the std gonna be 0, replace it with the std of the one system
            if len(mean_std_list) == 1:
                final_df["std"] = df["std"]
            
            print("\nAveraged values over all copies:\n\n",final_df,"\n")

            # Save as json
            json_data = { f"copy_{i}": { d["property"]: {key: value for key,value in d.items() if not key == "property"} for d in df.to_dict(orient="records") } for i,df in enumerate(mean_std_list) }
            json_data["average"] = { d["property"]: {key: value for key,value in d.items() if not key == "property"} for d in final_df.to_dict(orient="records") }

            # Either append the new data to exising file or create new json
            json_path = f"{state_folder}/results.json"
            
            work_json( json_path, { "temperature": temperature, "pressure": temperature,
                                    ensemble: { "data": json_data, "paths": files, "fraction_discarded": fraction } }, "append" )
        
            # Add the extracted values for the command, analysis_folder and ensemble to the class
            merge_nested_dicts( self.analysis_dictionary, { (temperature, temperature): { analysis_folder: { ensemble: final_df } } } )
        
    def analysis_free_energy( self, analysis_folder: str, solute: str, ensemble: str, method: str="MBAR"):
        """
        Extracts free energy difference for a specified folder and solute and ensemble.

        Parameters:
         - analysis_folder (str): The name of the folder where the analysis will be performed.
         - solute (str): Solute under investigation
         - ensemble (str): The name of the ensemble for which properties will be extracted. Should be xx_ensemble.
         - method (str, optional): The free energy method that should be used. Defaults to "MBAR" 

        Returns:
            None

        The method searches for output files in the specified analysis folder that match the given ensemble.
        For each group of files with the same temperature and pressure, the properties are extracted using alchempy.
        The extracted properties are then averaged over all copies and the mean and standard deviation are calculated.
        The averaged values and the extracted data for each copy are saved as a JSON file in the destination folder.
        The extracted values are also added to the class's analysis dictionary.
        """
        # Define folder for analysis
        sim_folder = f'{self.system_setup["folder"]}/{self.system_setup["name"]}/{analysis_folder}/{solute}'

        print(f"\nExtract solvation free energy resulst for solute: {solute}\n")

        # Loop over each temperature & pressure state
        for temp, press in zip( self.system_setup["temperature"], self.system_setup["pressure"] ):

            analysis_folder = f"{sim_folder}/temp_{temp:.0f}_pres_{press:.0f}"

            copies = [ copy for copy in os.listdir(analysis_folder) if "copy_" in copy ]
            copies.sort(key = lambda x: int(x.split("_")[1]))

            data = { "data": {}, "paths": [ f"{analysis_folder}/{copy}" for copy in copies ], "fraction_discarded": 0.0  }

            print(f"Temperature: {temp}, Pressure: {press}\n   "+"\n   ".join(data["paths"]) + "\n")

            for copy in copies:
                copy_folder = f"{analysis_folder}/{copy}"
                MBAR = get_mbar( path = copy_folder, ensemble = ensemble, temperature = temp )
                # Negate the value, as decoupling is done, and solvation free energy described the coupling into a solution
                data["data"][copy] = { "property": ["solvation_free_energy"], "mean": [ -MBAR.delta_f_.iloc[-1,0] * 8.314 * temp / 1000 ], "std": [ MBAR.d_delta_f_.iloc[-1,0] * 8.314 * temp / 1000 ], "units": [ "kJ / mol" ] }

            data["data"]["average"] = { "property": ["solvation_free_energy"], "mean":  [ np.mean([ item["mean"][0] for key, item in data["data"].items() ]) ], 
                                        "std": [ np.std( [ item["mean"][0] for key, item in data["data"].items() ], ddof=1 ) ] if len(data["data"].items()) > 1 else  np.mean( [ item["std"][0] for key, item in data["data"].items() ]), "units": [ "kJ / mol" ] }

            print("\nAveraged values over all copies:\n\n",pd.DataFrame(data["data"]["average"]),"\n")

            # Either append the new data to exising file or create new json
            json_path = f"{analysis_folder}/results.json"

            work_json( json_path, data, "append" )

            # Add the extracted values for the command, analysis_folder and ensemble to the class
            merge_nested_dicts( self.analysis_dictionary, { (temp, press): { "solvation_free_energy": { analysis_folder: { ensemble: pd.DataFrame(data["data"]["average"]) } } } } )
