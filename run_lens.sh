#!/bin/bash

repo_dir_path=$(dirname $(realpath $0))
scripts_dir_path="$repo_dir_path/vapor/scripts"
ingp_dir_path="$HOME/code/instant-ngp"

render_height=455
render_width=256

for scene in blue_chairs meeting_table seminar staircase staircase_ext
do
    dataset_dir_path="$HOME/data/AmbiguousReloc"
    scene_dir_path="$dataset_dir_path/$scene"
    lens_dir_path="$scene_dir_path/lens"
    echo $scene_dir_path
    echo "running lens on scene $scene"
    python $scripts_dir_path/interpolate_training_poses.py $scene
    python $ingp_dir_path/scripts/run.py \
        --mode nerf \
        --load_snapshot $scene_dir_path/nerf.ingp \
        --screenshot_transforms $lens_dir_path/transforms.json \
        --screenshot_dir $lens_dir_path/renders \
        --width $render_width \
        --height $render_height
done
