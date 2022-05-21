#!/bin/bash

bash_script_path=$(realpath $0)
scripts_dir_path="$(dirname $(dirname $bash_script_path))/scripts"

for sequence in "staircase" "staircase_ext"
do for top_percent in 0.2
    do for batch_size in 4
        do for kld_max_weight in 0.01
            do
                echo "starting sequence $sequence, top_percent $top_percent, batch_size $batch_size, kld_max_weight $kld_max_weight"
                CUDA_LAUNCH_BLOCKING=1 python $scripts_dir_path/train_pipeline.py \
                    --sequence $sequence \
                    --top_percent $top_percent \
                    --batch_size $batch_size \
                    --kld_max_weight $kld_max_weight
            done
        done
    done
done
