#!/bin/bash

ingp_dir_path="$HOME/code/instant-ngp"
dataset_dir_path="$HOME/data/Ambiguous_ReLoc_Dataset"

for scene in "meeting_table"
do
    echo "modelling NeRF of scene $scene using Instant-NGP"
    scene_dir_path="$dataset_dir_path/$scene"
    python $ingp_dir_path/scripts/run.py \
        --training_data $scene_dir_path/transforms.json \
        --mode nerf \
        --n_steps 100000 \
        --save_snapshot $scene_dir_path/nerf.msgpack
done
