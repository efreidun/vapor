#!/bin/bash

bash_script_path=$(realpath $0)
code_dir_path=$(dirname $(dirname $(dirname $(dirname $bash_script_path))))
parent_dir_path=$(dirname $code_dir_path)
data_dir_path="$parent_dir_path/data/Ambiguous_ReLoc_Dataset"

render_height=240
render_width=135

for scene in "blue_chairs" "meeting_table" "staircase" "staircase_ext" "seminar"
do
    echo "rendering test images in scene $scene"
    scene_dir_path="$data_dir_path/$scene"
    python $code_dir_path/instant-ngp/scripts/run.py \
        --mode nerf \
        --load_snapshot $scene_dir_path/nerf.msgpack \
        --screenshot_transforms $scene_dir_path/transforms_test.json \
        --screenshot_dir $scene_dir_path/renders \
        --width $render_width \
        --height $render_height
done
