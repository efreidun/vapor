#!/bin/bash

repo_dir_path=$(dirname $(realpath $0))
scripts_dir_path="$repo_dir_path/vaincapo/scripts"

for scene in "meeting_table" "blue_chairs" "seminar" "staircase" "staircase_ext"
do
    echo "starting scene $scene"
    python $scripts_dir_path/prepare_scene_json.py $scene --dataset AmbiguousReloc
    python $scripts_dir_path/prepare_scene_json.py $scene --split test --dataset AmbiguousReloc
done
