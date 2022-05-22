#!/bin/bash

repo_dir_path=$(dirname $(realpath $0))
scripts_dir_path="$repo_dir_path/vaincapo/scripts"

for sequence in "blue_chairs" "meeting_table" "staircase" "staircase_ext" "seminar"
do for top_percent in 0.2
    do for batch_size in 4
        do for kld_max_weight in 0.01
            do for tra_weight in 15 20
                do for image_crop in 0.7 0.8 0.9
                    do
                        echo "starting sequence $sequence, top_percent $top_percent, batch_size $batch_size, kld_max_weight $kld_max_weight"
                        python $scripts_dir_path/train_pipeline.py \
                            --sequence $sequence \
                            --top_percent $top_percent \
                            --batch_size $batch_size \
                            --kld_max_weight $kld_max_weight \
                            --tra_weight $tra_weight \
                            --image_crop $image_crop \
                            --device cuda:1
                    done
                done
            done
        done
    done
done
