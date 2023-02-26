#!/bin/bash

repo_dir_path=$(dirname $(realpath $0))
scripts_dir_path="$repo_dir_path/vapor/scripts"

for sequence in "blue_chairs" "meeting_table" "staircase" "staircase_ext" "seminar"; do
        echo "starting sequence $sequence"
        python $scripts_dir_path/train_pipeline.py \
            --sequence $sequence \
            --dataset AmbiguousReloc \
            --top_percent 0.2 \
            --image_mode posenet \ 
            --epochs 2000
done
