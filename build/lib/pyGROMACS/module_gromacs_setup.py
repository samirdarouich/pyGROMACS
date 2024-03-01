import os
import re
import yaml
import glob
import subprocess
import pandas as pd
from jinja2 import Template
from itertools import groupby
from typing import List, Dict, Any

from .utils import ( generate_initial_configuration, generate_mdp_files, generate_job_file, 
                     change_topo, read_gromacs_xvg, work_json, merge_nested_dicts )

class GROMACS_setup:

    def __init__( self, simulation_setup: str, simulation_default: str, simulation_ensemble: str ) -> None:
        """
        Initialize a new instance of the gromacs class.

        Parameters:
            simulation_setup (str): Path to the simulation setup YAML file. Containing all pipeline settings.
            simulation_default (str): Path to the simulation default YAML file. Containing all default GROMACS settings.
            simulation_ensemble (str): Path to the simulation ensemble YAML file. Containing all GROMACS ensemble settings (except temperature, pressure, and compressibility).

        Returns:
            None
        """
        # Save paths of the yaml files.
        self.simulation_setup_path    = os.path.abspath( simulation_setup )
        self.simulation_default_path  = os.path.abspath( simulation_default )
        self.simulation_ensemble_path = os.path.abspath( simulation_ensemble )

        with open( simulation_setup ) as file: 
            self.simulation_setup    = yaml.safe_load(file)
        
        with open( simulation_default ) as file:
            self.simulation_default  = yaml.safe_load(file)

        with open( simulation_ensemble ) as file:
            self.simulation_ensemble = yaml.safe_load(file)

        # Create an analysis dictionary containing all files
        self.analysis_dictionary = {}

    def prepare_simulation(self, ensembles: List[str], simulation_times: List[float], initial_systems: List[str]=[], folder_name: str="md", mdp_kwargs: Dict[str, Any]={}):
        """
        Prepares the simulation by generating job files for each temperature, pressure, and compressibility combination specified in the simulation setup.
        The method checks if an initial configuration file is provided. If not, it generates the initial configuration based on the provided molecule numbers and coordinate files. 
        It then generates MDP files for each ensemble in a separate folder and creates a job file for each copy of the simulation.

        Parameters:
         - ensembles (List[str]): A list of ensembles to generate MDP files for. Definitions of each ensemble is provided in self.simulation_ensemble.
         - simulation_times (List[float]): A list of simulation times (ns) for each ensemble.
         - initial_systems (List[str]): A list of initial system .gro files to be used for each temperature and pressure state.
         - folder_name (str, optional): Name of the subfolder where to perform the simulations. Defaults to "md".
                                        Path structure is as follows: system.folder/system.name/folder_name
         - mdp_kwargs (Dict[str, Any], optional): Further kwargs that are parsed to the mdp template. Defaults to "{}".
        
        Returns:
            None
        """
        self.job_files = []

        # Define simulation folder
        sim_folder = f'{self.simulation_setup["system"]["folder"]}/{self.simulation_setup["system"]["name"]}/{folder_name}'


        # Create initial configuration in (sim_folder/box)
        if not initial_systems:
            print("\nBuilding system based on provided molecule numbers and coordinate files!\n" )
            initial_coord = generate_initial_configuration( destination_folder = sim_folder, coordinate_paths = self.simulation_setup["system"]["paths"]["gro"], 
                                                            no_molecules = list(self.simulation_setup["system"]["molecules"].values()), box_lenghts = self.simulation_setup["system"]["box"] )
            initial_systems = [initial_coord]*len(self.simulation_setup["system"]["temperature"])
            flag_cpt = False
        else:
            print(f"\nSystems already build and initial configurations are provided at:\n\n" + "\n".join(initial_systems) + "\n" )
            print(f"Assuming that .cpt file is in the same folder!\n")
            flag_cpt = True

        # Copy tolopoly to the box folder and change it according to system specification
        initial_topo = change_topo( topology_path = self.simulation_setup["system"]["paths"]["topol"], destination_folder = f"{sim_folder}/box", 
                                    molecules_no_dict = self.simulation_setup['system']["molecules"], 
                                    file_name = f'topology_{self.simulation_setup["system"]["name"]}.top' )

        for initial_system, temperature, pressure, compressibility in zip( initial_systems, 
                                                                           self.simulation_setup["system"]["temperature"], 
                                                                           self.simulation_setup["system"]["pressure"], 
                                                                           self.simulation_setup["system"]["compressibility"]  ):
            
            job_files = []

            # If system is already build assume the cpt file is in the same folder as the coordinates.
            initial_cpt = initial_system.replace( initial_system.split(".")[-1], "cpt") if flag_cpt else ""
            
            # Define folder for specific temp and pressure state, as well as for each copy
            for copy in range( self.simulation_setup['system']["copies"] + 1 ):
                copy_folder = f"{sim_folder}/temp_{temperature:.0f}_pres_{pressure:.0f}/copy_{copy}"

                # Produce mdp files (for each ensemble an own folder 0x_ensemble)
                mdp_files = generate_mdp_files( destination_folder = copy_folder, mdp_template = self.simulation_setup["system"]["paths"]["template"]["mdp_file"],
                                                ensembles = ensembles, temperature = temperature, pressure = pressure, 
                                                compressibility = compressibility, simulation_times = simulation_times,
                                                dt = self.simulation_default["system"]["dt"], kwargs = { **mdp_kwargs, **self.simulation_default }, 
                                                ensemble_definition = self.simulation_ensemble )
                
                # Create job file
                job_files.append( generate_job_file( destination_folder = copy_folder, job_template = self.simulation_setup["system"]["paths"]["template"]["job_file"], mdp_files = mdp_files, 
                                                     intial_coord = initial_system, initial_topo = initial_topo, job_name = f'{self.simulation_setup["system"]["name"]}_{temperature:.0f}',
                                                     job_out = f"job_{temperature:.0f}.sh", ensembles = ensembles, initial_cpt = initial_cpt ) )
            self.job_files.append( job_files )

    def add_guest_molecule_and_prepare_equilibrate(self, solute: str, solute_coordinate: str, initial_systems: List[str], ensembles: List[str], 
                                                   simulation_times: List[float], folder_name: str="free_energy", flag_cpt: bool=True ):
        """
        Function that adds a guest molecule to the system and prepares it for equilibration simulations.

        Parameters:
         - solute (str): The name of the guest molecule.
         - solute_coordinate (str): The path to the coordinate file of the guest molecule.
         - initial_systems (List[str]): A list of initial system .gro files to be used to insert the guest molecule for each temperature and pressure state.
         - ensembles (List[str]): A list of ensembles to generate MDP files for. Definitions of each ensemble is provided in self.simulation_ensemble.
         - simulation_times (List[float]): A list of simulation times (ns) for each ensemble.
         - folder_name (str, optional): The name of the folder where the simulations will be performed. Subfolder call "equilibration" will be created there. Defaults to "free_energy".
         - flag_cpt (bool, optional): If checkpoint files are provided in the same folder as the inital systems. Otherwise dont use checkpoint files.
        
        Returns:
            None

        The method creates a new folder for the simulation of the guest molecule in the specified subfolder. It copies the initial topology file to the new folder
        and modifies it by increasing the number of the guest molecule by one.
        """
        self.job_files = []

        # Define simulation folder
        sim_folder = f'{self.simulation_setup["system"]["folder"]}/{self.simulation_setup["system"]["name"]}/{folder_name}/{solute}'

        # Change initial topology by increasing the number of the solute by one.
        initial_topo = change_topo( topology_path = self.simulation_setup["system"]["paths"]["topol"], destination_folder = f"{sim_folder}/box", 
                                    molecules_no_dict = { **self.simulation_setup['system']["molecules"], solute: None}, 
                                    file_name = f'topology_{self.simulation_setup["system"]["name"]}.top' )
                
        # Prepare equilibration of new system for each temperature/pressure state
        for initial_system, temperature, pressure, compressibility in zip( initial_systems, 
                                                                           self.simulation_setup["system"]["temperature"], 
                                                                           self.simulation_setup["system"]["pressure"], 
                                                                           self.simulation_setup["system"]["compressibility"]  ):
            job_files = []

            
            # Genereate initial box with solute ( a equilibrated structure is provided for every temperature & pressure state )
            initial_coord = generate_initial_configuration( destination_folder = sim_folder, coordinate_paths = [solute_coordinate], no_molecules = [1],
                                                            box_lenghts = [], build_intial_box = False, initial_system = initial_system )
        
            # Assume that the cpt file is in the same folder as the coordinates.
            initial_cpt = initial_system.replace( initial_system.split(".")[-1], "cpt") if flag_cpt else ""

            # Define folder for specific temp and pressure state, as well as for each copy
            for copy in range( self.simulation_setup['system']["copies"] + 1 ):
                copy_folder = f"{sim_folder}/equilibration/temp_{temperature:.0f}_pres_{pressure:.0f}/copy_{copy}"

                # Produce mdp files (for each ensemble an own folder 0x_ensemble)
                mdp_files = generate_mdp_files( destination_folder = copy_folder, mdp_template = self.simulation_setup["system"]["paths"]["template"]["mdp_file"],
                                                ensembles =ensembles, temperature = temperature, pressure = pressure, 
                                                compressibility = compressibility, simulation_times = simulation_times,
                                                dt = self.simulation_default["system"]["dt"], kwargs = self.simulation_default, 
                                                ensemble_definition = self.simulation_ensemble )
                
                # Create job file
                job_files.append( generate_job_file( destination_folder = copy_folder, job_template = self.simulation_setup["system"]["paths"]["template"]["job_file"], mdp_files = mdp_files, 
                                                    intial_coord = initial_coord, initial_topo = initial_topo, job_name = f'{self.simulation_setup["system"]["name"]}_{temperature:.0f}_{solute}_eq',
                                                    job_out = f"job_{temperature:.0f}.sh", ensembles = ensembles, initial_cpt = initial_cpt ) )
                
            self.job_files.append( job_files )

    
    def prepare_free_energy_simulation( self, simulation_free_energy: str, solute: str, combined_lambdas: List[float], initial_systems: List[str], ensembles: List[str], 
                                        simulation_times: List[float], folder_name: str="free_energy", precision: int=3, flag_cpt: bool=True  ):
        """
        Function that prepares free energy simulations for several temperatures and pressure states.

        Parameters:
         - simulation_free_energy (str): Path to the simulation free energy YAML file. Containing all free energy settings.
         - combined_lambdas (List[float]): List of combined lambdas for the free energy simulations.
         - initial_systems (List[str]): A list of initial system .gro files to be used for each temperature and pressure state.
         - ensembles (List[str]): A list of ensembles to generate MDP files for. Definitions of each ensemble is provided in self.simulation_ensemble.
         - simulation_times (List[float]): A list of simulation times (ns) for each ensemble.
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
        sim_folder = f'{self.simulation_setup["system"]["folder"]}/{self.simulation_setup["system"]["name"]}/{folder_name}/{solute}'

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
        initial_topo = change_topo( topology_path = self.simulation_setup["system"]["paths"]["topol"], destination_folder = f"{sim_folder}/box", 
                                    molecules_no_dict = { **self.simulation_setup['system']["molecules"], solute: None}, 
                                    file_name = f'topology_{self.simulation_setup["system"]["name"]}.top' )

        # Prepare free energy simulations for several temperatures & pressure states
        for initial_system, temperature, pressure, compressibility in zip( initial_systems, 
                                                                           self.simulation_setup["system"]["temperature"], 
                                                                           self.simulation_setup["system"]["pressure"], 
                                                                           self.simulation_setup["system"]["compressibility"]  ):
            
            job_files = []

            # Assume that the cpt file is in the same folder as the coordinates.
            initial_cpt = initial_system.replace( initial_system.split(".")[-1], "cpt") if flag_cpt else ""

            # Define folder for specific temp and pressure state, as well as for each copy
            for copy in range( self.simulation_setup['system']["copies"] + 1 ):
                copy_folder = f"{sim_folder}/temp_{temperature:.0f}_pres_{pressure:.0f}/copy_{copy}"

                # Define for each lambda an own folder
                for i,_ in enumerate(combined_lambdas):
                    self.simulation_default["free_energy"]["init_lambda_state"] = i

                    lambda_folder = f"{copy_folder}/lambda_{i}"

                    # Produce mdp files (for each ensemble an own folder 0x_ensemble)
                    mdp_files = generate_mdp_files( destination_folder = lambda_folder, mdp_template = self.simulation_setup["system"]["paths"]["template"]["mdp_file"],
                                                    ensembles = ensembles, temperature = temperature, pressure = pressure, 
                                                    compressibility = compressibility, simulation_times = simulation_times,
                                                    dt = self.simulation_default["system"]["dt"], kwargs = self.simulation_default, 
                                                    ensemble_definition = self.simulation_ensemble )
                
                    # Create job file
                    job_files.append( generate_job_file( destination_folder = lambda_folder, job_template = self.simulation_setup["system"]["paths"]["template"]["job_file"], 
                                                         mdp_files = mdp_files, intial_coord = initial_system, initial_topo = initial_topo,
                                                         job_name = f'{self.simulation_setup["system"]["name"]}_{temperature:.0f}_lambda_{i}', job_out = f"job_{temperature:.0f}_lambda_{i}.sh",
                                                         ensembles = ensembles, initial_cpt = initial_cpt ) )
            
            self.job_files.append( job_files )

    def optimize_intermediates( self, simulation_free_energy: str, optimize_template: str,  solute: str, initial_coord: str, initial_cpt: str, 
                                initial_topo: str, iteration_time: float, temperature: float, pressure: float, compressibility: float,
                                precision: int=3, tolerance: float=0.05, min_overlap: float=0.15, max_overlap: float=0.25, 
                                max_iterations: int=20, folder_name: str="free_energy", submission_command: str="qsub"):
        """
        Function for optimizing solvation free energy intermediates using the decoupling approach. This wiöl

        Parameters:
            simulation_free_energy (str): Path to the file containing simulation free energy data.
            optimize_template (str): Path to the template file used for optimization.
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
            submission_command (str, optional): Submission command for the cluster. Default is "qsub".

        Returns:
            None
        """

        # Open template
        with open(optimize_template) as f:
            template = Template( f.read() )

        # Define simulation folder
        sim_folder = f'{self.simulation_setup["system"]["folder"]}/{self.simulation_setup["system"]["name"]}/{folder_name}/{solute}/optimization'
        
        paths = { "simulation_folder": sim_folder, "initial_coord": initial_coord, "initial_cpt": initial_cpt, "initial_topo": initial_topo,
                  "parameter": { "setup": self.simulation_setup_path, "default": self.simulation_default_path, 
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
        subprocess.run( [submission_command, job_file] )
        print("\n")

    def submit_simulation(self, submission_command: str="qsub"):
        """
        Function that submits predefined jobs to the cluster.
        
        Parameters:
            submission_command (str, optional): Submission command for the cluster

        Returns:
            None
        """
        for temperature, pressure, job_files in zip( self.simulation_setup["system"]["temperature"], self.simulation_setup["system"]["pressure"], self.job_files ):
            print(f"\nSubmitting simulations at Temperature = {temperature:.0f} K, Pressure = {pressure:.0f} bar\n")

            for job_file in job_files:
                print(f"Submitting job: {job_file}")
                subprocess.run( [submission_command, job_file] )
                print("\n")

    def analysis_extract_properties(self, analysis_folder: str, ensemble: str, extracted_properties: List[str], command: str="gmx energy", gmx_version: str="chem/gromacs/2022.4", fraction: float=0.7 ):
        """
        Extracts properties from GROMACS output files for a specific ensemble.

        Parameters:
            analysis_folder (str): The name of the folder where the analysis will be performed.
            ensemble (str): The name of the ensemble for which properties will be extracted. Should be xx_ensemble.
            extracted_properties (List[str]): A list of properties to be extracted from the GROMACS output files.
            command (str, optional): The GROMACS command to be used for property extraction. Defaults to "gmx energy".
            gmx_version (str, optional): The version of GROMACS to be used. Defaults to "chem/gromacs/2022.4".
            fraction (float, optional): The fraction of data to be discarded from the beginning of the simulation. Defaults to 0.7.

        Returns:
            None

        The method searches for output files in the specified analysis folder that match the given ensemble.
        For each group of files with the same temperature and pressure, the properties are extracted using the specified GROMACS command.
        The extracted properties are then averaged over all copies and the mean and standard deviation are calculated.
        The averaged values and the extracted data for each copy are saved as a JSON file in the destination folder.
        The extracted values are also added to the class's analysis dictionary.
        """
        # Define folder for analysis
        sim_folder = f'{self.simulation_setup["system"]["folder"]}/{self.simulation_setup["system"]["name"]}/{analysis_folder}'

        # Seperatre the ensemble name to determine output files
        ensemble_name = ensemble.split("_")[1]

        # Search output files and sort them after temperature / pressure and then copy
        files = glob.glob( f"{sim_folder}/**/{ensemble}/{ensemble_name}.edr", recursive = True )
        files.sort( key=lambda x: (int(re.search(r'temp_(\d+)', x).group(1)),
                                   int(re.search(r'pres_(\d+)', x).group(1)),
                                   int(re.search(r'copy_(\d+)', x).group(1))) )

        if len(files)>0:
            pass
        else:
            raise KeyError(f"No files found machting the ensemble: {ensemble} in folder\n:   {sim_folder}")
        

        # Group paths by temperature and pressure states
        grouped_paths = {}
        for (temp, pres), paths_group in groupby(files, key=lambda x: (int(re.search(r'temp_(\d+)', x).group(1)),
                                                                       int(re.search(r'pres_(\d+)', x).group(1)))):
            grouped_paths[(temp, pres)] = list(paths_group)

        # Iterate through the files and extract properties from gromacs
        print( f"These output files are found:\n")
        for (temp, pres), paths_group in grouped_paths.items():
            print(f"Temperature: {temp}, Pressure: {pres}\n   "+"\n   ".join(paths_group) + "\n")

            data_list = []

            for path in paths_group:
                txt  = f"# Bash script to extract GROMACS properties. Automaticaly created by pyGROMACS.\n\nmodule purge\nmodule load {gmx_version}\n\n"
                txt += f"cd {os.path.dirname( path )}\n"
                txt += "echo -e '"+ r"\n".join(extracted_properties) + r"\n" + f"' | {command} -f {ensemble_name} -s {ensemble_name} -o properties\n"
                txt += f"# Delete old .xvg files instead of backuping them\nrm -f \#properties.xvg.*#\n"
 
                bash_file = f"{os.path.dirname( path )}/extract_properties.sh"
                with open( bash_file, "w") as f: 
                    f.write( txt )

                subprocess.run(["bash",bash_file], capture_output=True)

                # Check if output is correctly genereated
                if not os.path.exists( f"{os.path.dirname( path )}/properties.xvg" ):
                    raise FileExistsError(f"Extracting the properties did not work for:\n   {path}")


                # Analysis data
                data_list.append( read_gromacs_xvg( f"{os.path.dirname( path )}/properties.xvg", fraction = fraction) )

            # Mean the values for each copy and exctract mean and standard deviation
            mean_std_list  = [df.drop(columns=['Time']).agg(['mean', 'std']).T.reset_index().rename(columns={'index': 'property'}) for df in data_list]
            final_df       = pd.concat(mean_std_list,axis=0).groupby("property")["mean"].agg(["mean","std"]).reset_index()

            print("\nAveraged values over all copies:\n\n",final_df,"\n")

            # Save as json 
            json_data = { f"copy_{i}": df.to_dict(orient="list") for i,df in enumerate(mean_std_list) }
            json_data.update( {"average" : final_df.to_dict(orient="list") } )

            # Extract main folder for the state:
            destination_folder = re.search(r'(/.*?/temp_\d+_pres_\d+/)', path).group(1)

            # Either append the new data to exising file or create new json
            json_path = f"{destination_folder}/results.json"
            
            work_json( json_path, {command.split()[1]: { ensemble: { "data": json_data, "paths": paths_group, "fraction_discarded": fraction } } }, "append" )
        
            # Add the extracted values for the command, analysis_folder and ensemble to the class
            merge_nested_dicts( self.analysis_dictionary, { (temp, pres): { command.split()[1]: { analysis_folder: { ensemble: final_df } } } } )
        
        