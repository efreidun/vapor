#!/bin/bash

bash_script_path=$(realpath $0)
package_dir_path=$(dirname $(dirname $bash_script_path))
scripts_dir_path="$package_dir_path/scripts"
repo_dir_path=$(dirname $package_dir_path)
code_dir_path=$(dirname $repo_dir_path)
parent_dir_path=$(dirname $code_dir_path)

render_height=240
render_width=135

for run in "distinctive-planet-218"
do
    run_dir_path="$repo_dir_path/runs/$run"
    scene=$(yq .sequence $run_dir_path/config.yaml)
    scene_dir_path="$parent_dir_path/data/Ambiguous_ReLoc_Dataset/$scene"
    echo "evaluating run $run in scene $scene"
    python $scripts_dir_path/evaluate_pipeline.py $run
    python $code_dir_path/instant-ngp/scripts/run.py \
        --training_data $scene_dir_path/transforms.json \
        --mode nerf \
        --n_steps 0 \
        --load_snapshot $scene_dir_path/nerf.msgpack \
        --screenshot_transforms $run_dir_path/transforms.json \
        --screenshot_dir $run_dir_path/renders \
        --width $render_width \
        --height $render_height
    python $scripts_dir_path/evaluate_renders.py $run --width $render_width --height $render_height
done
