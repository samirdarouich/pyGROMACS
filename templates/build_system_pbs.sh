#!/bin/bash
#PBS -q tiny
#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:20:00
#PBS -N build_system
#PBS -o {{folder}}/build_log.o 
#PBS -e {{folder}}/build_log.e 
#PBS -l mem=3000mb

# Bash script to generate GROMACS box. Automaticaly created by pyGROMACS.

# Load GROMACS
module purge
module load chem/gromacs/2022.4

# Go to working folder
cd {{folder}}

# Use GROMACS to build the box
{{gmx_command}}


# Delete old .gro files
rm -f \#*.gro.*#