#!/bin/bash

bash_script_path=$(realpath $0)
scripts_dir_path=$(dirname $bash_script_path)

for sequence in "blue_chairs"
do for top_percent in 0.1 0.05 0.10 0.15 0.20 0.25 0.30
    do for batch_size in 4 8 16 32
        do for kld_max_weight in 0.0001 0.001 0.01 0.1 1.0
            do
                echo "starting sequence $sequence, top_percent $top_percent, batch_size $batch_size, kld_max_weight $kld_max_weight"
                python $scripts_dir_path/train_pipeline.py \
                    --sequence $sequence \
                    --top_percent $top_percent \
                    --batch_size $batch_size \
                    --kld_max_weight $kld_max_weight
            done
        done
    done
done
