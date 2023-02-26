#!/bin/bash

ingp_dir_path="$HOME/code/instant-ngp"
dataset_dir_path="$HOME/data/AmbiguousReloc"

render_height=240
render_width=135

for scene in "blue_chairs" "meeting_table" "staircase" "staircase_ext" "seminar"
do
    echo "rendering test images in scene $scene"
    scene_dir_path="$dataset_dir_path/$scene"
    python $ingp_dir_path/scripts/run.py \
        --mode nerf \
        --load_snapshot $scene_dir_path/nerf.msgpack \
        --screenshot_transforms $scene_dir_path/transforms_test.json \
        --screenshot_dir $scene_dir_path/renders \
        --width $render_width \
        --height $render_height
done
