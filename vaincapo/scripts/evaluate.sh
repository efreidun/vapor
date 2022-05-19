#!/bin/bash

bash_script_path=$(realpath $0)
scripts_dir_path=$(dirname $bash_script_path)
repo_dir_path=$(dirname $(dirname $scripts_dir_path))
code_dir_path=$(dirname $repo_dir_path)
parent_dir_path=$(dirname $code_dir_path)

for run in "chocolate-microwave-113"
do
    run_dir_path="$repo_dir_path/runs/$run"
    scene=$(yq .sequence $run_dir_path/config.yaml)
    scene_dir_path="$parent_dir_path/data/Ambiguous_ReLoc_Dataset/$scene"
    echo "evaluating run $run in scene $scene"
    python $scripts_dir_path/evaluate_pipeline.py $run
    python $code_dir_path/instant-ngp/scripts/run.py \
        --training_data $scene_dir_path/transforms.json \
        --mode nerf \
        --load_snapshot $scene_dir_path/nerf.msgpack \
        --screenshot_transforms $run_dir_path/transforms.json \
        --screenshot_dir $run_dir_path/renders \
        --width 540 \
        --height 960
    python $scripts_dir_path/evaluate_renders.py $run
done
