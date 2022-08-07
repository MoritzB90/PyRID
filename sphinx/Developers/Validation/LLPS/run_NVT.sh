#!/bin/bash
#SBatch -p medium
#SBATCH -t 48:00:00
#SBATCH -o job_output/job-%J.out
#SBATCH -C scratch
#SBATCH -n 1
#SBATCH -c 11
#SBATCH -a 0-10

module purge
module load anaconda3
source activate pypatch-env

python3 -u /scratch/users/mbecker3/PyPatch/Espinosa/Espinosa_Validation.py $1 'DC' $SLURM_ARRAY_TASK_ID

conda deactivate
