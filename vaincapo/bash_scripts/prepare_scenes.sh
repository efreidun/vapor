#!/bin/bash

bash_script_path=$(realpath $0)
scripts_dir_path="$(dirname $(dirname $bash_script_path))/scripts"

for scene in "meeting_table" "blue_chairs" "seminar" "staircase" "staircase_ext"
do
    echo "starting scene $scene"
    python $scripts_dir_path/prepare_scene_json.py $scene
done
