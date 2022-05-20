#!/bin/bash

bash_script_path=$(realpath $0)
code_dir_path=$(dirname $(dirname $(dirname $(dirname $bash_script_path))))
parent_dir_path=$(dirname $code_dir_path)

for scene in "blue_chairs"
do
    echo "modelling NeRF of scene $scene using Instant-NGP"
    scene_dir_path="$parent_dir_path/data/Ambiguous_ReLoc_Dataset/$scene"
    python $code_dir_path/instant-ngp/scripts/run.py \
        --training_data $scene_dir_path/transforms.json \
        --mode nerf \
        --n_steps 100000 \
        --save_snapshot $scene_dir_path/nerf.msgpack
done
