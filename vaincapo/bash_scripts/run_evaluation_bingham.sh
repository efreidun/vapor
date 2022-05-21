#!/bin/bash

bash_script_path=$(realpath $0)
package_dir_path=$(dirname $(dirname $bash_script_path))
scripts_dir_path="$package_dir_path/scripts"
repo_dir_path=$(dirname $package_dir_path)
code_dir_path=$(dirname $repo_dir_path)
parent_dir_path=$(dirname $code_dir_path)

render_height=240
render_width=135

for scene in "blue_chairs" "meeting_table" "staircase" "staircase_ext" "seminar"
do for num_coeff in 5 10 25 50
    do
        run=$scene"_"$num_coeff
        run_dir_path="$repo_dir_path/bingham_runs/$run"
        scene_dir_path="$parent_dir_path/data/Ambiguous_ReLoc_Dataset/$scene"
        echo "evaluating run $run"
        python $scripts_dir_path/evaluate_samples.py $run
        python $code_dir_path/instant-ngp/scripts/run.py \
            --mode nerf \
            --load_snapshot $scene_dir_path/nerf.msgpack \
            --screenshot_transforms $run_dir_path/transforms.json \
            --screenshot_dir $run_dir_path/renders \
            --width $render_width \
            --height $render_height
        python $scripts_dir_path/evaluate_renders.py $run \
            --source bingham --width $render_width --height $render_height
    done
done
