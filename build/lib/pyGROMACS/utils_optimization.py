## Special function to submit free energy in optimization
from .utils import generate_mdp_files, generate_job_file

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