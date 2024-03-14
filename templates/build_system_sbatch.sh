#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 1:00:00
#SBATCH -p single
#SBATCH -J build_system
#SBATCH -o {{folder}}/build_log.o
#SBATCH --mem-per-cpu=2000

# Bash script to generate GROMACS box. Automaticaly created by pyGROMACS.

# Load GROMACS
module purge
module load chem/gromacs/2023.3

# Go to working folder
cd {{folder}}

# Use GROMACS to build the box
{{gmx_command}}


# Delete old .gro files
rm -f \#*.gro.*#