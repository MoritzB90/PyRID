#!/bin/bash

for exp in 'Pential' 'Tetrahedral' 'Tertial'
do
        my_id=$(sbatch --parsable run_Espinosa_Equil.sh $exp)
        sbatch -d afterok:$my_id run_Espinosa.sh $exp
done