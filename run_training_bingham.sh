#!/bin/bash

repo_dir_path=$(dirname $(realpath $0))
scripts_dir_path="$repo_dir_path/vaincapo/scripts"
bingham_dir_path="$HOME/code/torch_bingham/cam_reloc"

for sequence in "blue_chairs" "meeting_table" "staircase" "staircase_ext" "seminar"
do for num_coeff in 5 10 25 50
    do
        echo "starting sequence $sequence with $num_coeff components"
        python $bingham_dir_path/main.py --stage 0 --training --num_epochs 300 \
            --scene $sequence --num_coeff $num_coeff
        python $bingham_dir_path/main.py --stage 1 --training --restore --model model_299 --num_epochs 300 \
            --scene $sequence --num_coeff $num_coeff
        python $bingham_dir_path/main.py --stage 2 --training --restore --model model_299 --num_epochs 300 \
            --scene $sequence --num_coeff $num_coeff
        python $bingham_dir_path/main.py --restore --model model_299 \
            --scene $sequence --num_coeff $num_coeff
        python $scripts_dir_path/simulate_dist.py $sequence"_"$num_coeff
    done
done
