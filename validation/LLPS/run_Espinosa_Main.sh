#!/bin/bash

for exp in '5_sites' '4_sites' '3_sites'
do
        my_id=$(sbatch --parsable run_Espinosa_Equil.sh $exp)
        sbatch -d afterok:$my_id run_Espinosa.sh $exp
done