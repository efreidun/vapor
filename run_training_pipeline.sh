#!/bin/bash

repo_dir_path=$(dirname $(realpath $0))
scripts_dir_path="$repo_dir_path/vapor/scripts"

for sequence in "Ceiling"; do
        echo "starting sequence $sequence"
        python $scripts_dir_path/train_pipeline.py \
            --sequence $sequence \
            --dataset Rig \
            --top_percent 0.05 \
            --image_mode posenet \
            --epochs 2000 \
            --map_depth 5
done
for sequence in "blue_chairs" "meeting_table" "seminar" "staircase" "staircase_ext"; do
        echo "starting sequence $sequence"
        python $scripts_dir_path/train_pipeline.py \
            --sequence $sequence \
            --dataset AmbiguousReloc \
            --top_percent 0.05 \
            --image_mode posenet \
            --epochs 2000 \
            --map_depth 5
done
