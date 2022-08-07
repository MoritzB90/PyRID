#!/bin/bash
#SBatch -p medium
#SBATCH -t 06:00:00
#SBATCH -o job_output/job-%J.out
#SBATCH -C scratch
#SBATCH -n 1
#SBATCH -c 1

module purge
module load anaconda3
source activate pypatch-env

python3 -u /scratch/users/mbecker3/PyPatch/Espinosa/Espinosa_Validation.py $1 'Equil' -1

conda deactivate
