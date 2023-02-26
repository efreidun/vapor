#!/bin/bash

repo_dir_path=$(dirname $(realpath $0))
scripts_dir_path="$repo_dir_path/vapor/scripts"
ingp_dir_path="$HOME/code/instant-ngp"

render_height=240
render_width=135

for run in $1
do
    run_dir_path="$repo_dir_path/runs/$run"
    scene=$(yq .sequence $run_dir_path/config.yaml)
    dataset=$(yq .dataset $run_dir_path/config.yaml)
    dataset_dir_path="$HOME/data/$dataset"
    scene_dir_path="$dataset_dir_path/$scene"
    echo $scene_dir_path
    echo "evaluating run $run in scene $scene"
    python $scripts_dir_path/evaluate_pipeline.py $run
    python $ingp_dir_path/scripts/run.py \
        --mode nerf \
        --load_snapshot $scene_dir_path/nerf.msgpack \
        --screenshot_transforms $run_dir_path/transforms.json \
        --screenshot_dir $run_dir_path/renders \
        --width $render_width \
        --height $render_height
    python $scripts_dir_path/evaluate_renders.py $run \
       --width $render_width --height $render_height
    python $scripts_dir_path/visualize_samples.py $run --dataset $dataset --norender
done
