#!/bin/bash

repo_dir_path=$(dirname $(realpath $0))
scripts_dir_path="$repo_dir_path/vapor/scripts"

for sequence in seminar meeting_table staircase; do
        echo "starting sequence $sequence"
        python $scripts_dir_path/train_cvaepor.py \
            --sequence $sequence \
            --dataset AmbiguousReloc \
            --image_mode posenet \
            --kld_max_weight 0.1 \
            --iters 200000 \
            --num_flows 0 \
            --batch_size 16 \
            --lr 0.0001 \
            --latent_dim 4 \
            --weight_decay 0.001 \
            --bottleneck_dim 64 \
            --kld_warmup_start 1000 \
            --kld_warmup_period 4000 \
            --lens
done
