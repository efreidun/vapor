#!/bin/bash

repo_dir_path=$(dirname $(realpath $0))
scripts_dir_path="$repo_dir_path/vaincapo/scripts"
ingp_dir_path="$HOME/code/instant-ngp"
dataset_dir_path="$HOME/data/Ambiguous_ReLoc_Dataset"

render_height=240
render_width=135

for scene in "blue_chairs" "seminar" "meeting_table" "staircase" "staircase_ext"
do for num_coeff in 50 10
    do
        run=$scene"_"$num_coeff
        run_dir_path="$repo_dir_path/bingham_runs/$run"
        scene_dir_path="$dataset_dir_path/$scene"
        echo "evaluating run $run"
        python $scripts_dir_path/visualize_mixture.py $run
        python $scripts_dir_path/simulate_dist.py $run
        python $scripts_dir_path/evaluate_samples.py $run
        python $ingp_dir_path/scripts/run.py \
            --mode nerf \
            --load_snapshot $scene_dir_path/nerf.msgpack \
            --screenshot_transforms $run_dir_path/transforms.json \
            --screenshot_dir $run_dir_path/renders \
            --width $render_width \
            --height $render_height
        python $scripts_dir_path/evaluate_renders.py $run \
            --source bingham --width $render_width --height $render_height
        python $scripts_dir_path/visualize_samples.py $run \
            --source bingham --norender
    done
done
