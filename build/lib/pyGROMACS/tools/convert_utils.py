

import os
import yaml, json, toml

from typing import List
from pyLAMMPS.tools.general_utils import flatten_list, find_key_by_value
from moleculegraph.molecule_utils import sort_force_fields, sort_graph_key

# Set precision for parameters
PRECISION = 4

def extract_force_field_gromacs( itp_files: List[str], top_file: str, output_path: str ):
    
    print("Starting to extract GROMACS force field...\n")

    # Define force field dictionary
    force_field = { "format": "GROMACS", "atoms": {}, "bonds": {}, "angles": {}, "torsions": {} }

    # Read in topology
    default_section, atom_section, bond_section, angle_section, dihedral_section = extract_top_file( top_file )

    # Read in itp_files
    itp_atoms = []
    itp_bonds = []
    itp_angles = []
    itp_dihedrals = []

    for itp_file in itp_files:
        _, itp_atom, itp_bond, itp_angle, itp_dihedral = extract_itp_file( itp_file )
        itp_atoms.extend( itp_atom )
        itp_bonds.extend( itp_bond )
        itp_angles.extend( itp_angle )
        itp_dihedrals.extend( itp_dihedral )

    # Define mapping from atom no to force field key.
    atom_map = { atom[0]: [] for atom in atom_section }
    for atom in itp_atoms:
        atom_map[ atom[1] ].append( atom[0] )

    # Define charge map, in case itp and topology has different charges
    # First use topology charges and then overwrite with itp charges
    charge_map = { atom[0]: float(atom[3]) for atom in atom_section } 
    charge_map.update( { atom[1]: float(atom[6]) for atom in itp_atoms } )

    # If topology has no informaton about bonds, angles, dihedrals, but itp does, add the information.
    if any( itp_bonds ) and not bond_section:  
        bond_section = []
        for bond in itp_bonds:
            name1, name2, style, *p = bond
            name1 = find_key_by_value( atom_map, name1 )
            name2 = find_key_by_value( atom_map, name2 )

            bond_section.append( [ name1, name2, style, *p ] )
    
    if any( itp_angles ) and not angle_section:
        angle_section = []
        for angle in itp_angles:
            name1, name2, name3, style, *p = angle
            name1 = find_key_by_value( atom_map, name1 )
            name2 = find_key_by_value( atom_map, name2 )
            name3 = find_key_by_value( atom_map, name3 )
            
            angle_section.append( [ name1, name2, name3, style, *p ] )

    if any( itp_dihedrals ) and not dihedral_section:
        dihedral_section = []
        for dihedral in itp_dihedrals:
            name1, name2, name3, name4, style, *p = dihedral
            name1 = find_key_by_value( atom_map, name1 )
            name2 = find_key_by_value( atom_map, name2 )
            name3 = find_key_by_value( atom_map, name3 )
            name4 = find_key_by_value( atom_map, name4 )

            dihedral_section.append( [ name1, name2, name3, name4, style, *p ] )


    # Write information

    for atom in atom_section:
        name, atom_no, mass, _, _, sigma, epsilon = atom
        charge = charge_map[name]
        force_field["atoms"][name] = { "name": name,
                                        "mass": round(float(mass),PRECISION),
                                        "charge": float(charge),
                                        "sigma": round(float(sigma),PRECISION),
                                        "epsilon": round(float(epsilon),PRECISION),
                                        "atom_no": int(atom_no)
                                    }

    for bond in bond_section:
        name1, name2, style, *p = bond

        graph_key = sort_graph_key( "[" + "][".join([name1,name2]) + "]" )
        name_list = sort_force_fields( [name1,name2] ).tolist()

        if "#" in p:
            p = p[:p.index("#")]
        
        force_field["bonds"][graph_key] = { "list": name_list,
                            "p": [round(float(pp),PRECISION) for pp in p],
                            "style": style
                        }
        
    for angle in angle_section:
        name1, name2, name3, style, *p = angle

        graph_key = sort_graph_key( "[" + "][".join([name1,name2,name3]) + "]" )
        name_list = sort_force_fields( [name1,name2,name3] ).tolist()

        if "#" in p:
            p = p[:p.index("#")]
        
        force_field["angles"][graph_key] = { "list": name_list,
                            "p": [round(float(pp),PRECISION) for pp in p],
                            "style": style
                            }

    for dihedral in dihedral_section:
        name1, name2, name3, name4, style, *p = dihedral

        graph_key = sort_graph_key( "[" + "][".join([name1,name2,name3,name4]) + "]" )
        name_list = sort_force_fields( [name1,name2,name3,name4] ).tolist()

        if "#" in p:
            p = p[:p.index("#")]
        
        force_field["torsions"][graph_key] = { "list": name_list,
                                "p": [round(float(pp),PRECISION) for pp in p],
                                "style": style
                                }

    # In case bonds, anlges, or dihedrals are not provided, add dummy entries
    if not force_field["bonds"]:
        force_field["bonds"]["dummy"] = { "list": [ "dummy", "dummy" ],
                                        "p": [],
                                        "style": 0 
                                        }
    if not force_field["angles"]:
        force_field["angles"]["dummy"] = { "list": [ "dummy", "dummy", "dummy" ],
                                        "p": [],
                                        "style": 0 
                                        }
    if not force_field["torsions"]:
        force_field["torsions"]["dummy"] = { "list": [ "dummy", "dummy", "dummy", "dummy" ],
                                        "p": [],
                                        "style": 0 
                                        }
    
    output_path = os.path.abspath( output_path )

    os.makedirs( os.path.dirname( output_path ), exist_ok = True )

    if ".yaml" in output_path:
        yaml.dump( force_field, open(output_path,"w") )
    elif ".json" in output_path:
        json.dump( force_field, open(output_path,"w"), indent=2 )
    elif ".toml" in output_path:
        toml.dump( force_field, open(output_path,"w") ) 

    print("Success!")



def extract_itp_file( itp_file: str ):

    with open(itp_file) as f:
        lines = f.readlines()

    in_molecule_section = False
    in_atom_section = False
    in_bond_section = False
    in_angle_section = False
    in_dihedral_section = False

    molecule_section = []
    atom_section = []
    bond_section = []
    angle_section = []
    dihedral_section = []

    for line in lines:
        if in_molecule_section:
            if line.startswith("[ atoms ]"):
                in_molecule_section = False
                in_atom_section = True
            elif line.strip() and not line.startswith(";"):
                molecule_section.append(line.split())
        elif in_atom_section:
            if line.startswith("[ bonds ]"):
                in_atom_section = False
                in_bond_section = True
            elif line.startswith("[") or line.startswith("#"):
                in_atom_section = False
            elif line.strip() and not line.startswith(";"):
                atom_section.append(line.split())
        elif in_bond_section:
            if line.startswith("[ angles ]"):
                in_bond_section = False
                in_angle_section = True
            elif line.startswith("[") or line.startswith("#"):
                in_bond_section = False
            elif line.strip() and not line.startswith(";"):
                bond_section.append(line.split())
        elif in_angle_section:
            if line.startswith("[ dihedrals ]"):
                in_angle_section = False
                in_dihedral_section = True
            elif line.startswith("[") or line.startswith("#"):
                in_angle_section = False
            elif line.strip() and not line.startswith(";"):
                angle_section.append(line.split())
        elif in_dihedral_section:
            if line.startswith("[") or line.startswith("#"):
                in_dihedral_section = False
            elif line.strip() and not line.startswith(";"):
                dihedral_section.append(line.split())
        elif line.startswith("[ moleculetype ]"):
            in_molecule_section = True
        elif line.startswith("[ bonds ]"):
            in_bond_section = True
    
    return molecule_section, atom_section, bond_section, angle_section, dihedral_section

def extract_top_file( top_file: str ):

    with open(top_file) as f:
        lines = f.readlines()

    in_default_section = False
    in_atom_section = False
    in_bond_section = False
    in_angle_section = False
    in_dihedral_section = False

    default_section = []
    atom_section = []
    bond_section = []
    angle_section = []
    dihedral_section = []

    for line in lines:
        if in_default_section:
            if line.startswith("[ atomtypes ]"):
                in_default_section = False
                in_atom_section = True
            elif line.strip() and not line.startswith(";"):
                default_section.append(line.split())
        elif in_atom_section:
            if line.startswith("[ bondtypes ]"):
                in_atom_section = False
                in_bond_section = True
            elif line.startswith("[") or line.startswith("#"):
                in_atom_section = False
            elif line.strip() and not line.startswith(";"):
                atom_section.append(line.split())
        elif in_bond_section:
            if line.startswith("[ angletypes ]"):
                in_bond_section = False
                in_angle_section = True
            elif line.startswith("[") or line.startswith("#"):
                in_bond_section = False
            elif line.strip() and not line.startswith(";"):
                bond_section.append(line.split())
        elif in_angle_section:
            if line.startswith("[ dihedraltypes ]"):
                in_angle_section = False
                in_dihedral_section = True
            elif line.startswith("[") or line.startswith("#"):
                in_angle_section = False
            elif line.strip() and not line.startswith(";"):
                angle_section.append(line.split())
        elif in_dihedral_section:
            if line.startswith("[") or line.startswith("#"):
                in_dihedral_section = False
            elif line.strip() and not line.startswith(";"):
                dihedral_section.append(line.split())
        elif line.startswith("[ defaults ]"):
            in_default_section = True
        elif line.startswith("[ atomtypes ]"):
            in_atom_section = True

    return default_section, atom_section, bond_section, angle_section, dihedral_section