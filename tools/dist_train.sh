#!/usr/bin/env bash

# e.g. ./tools/dist_train.sh config/i3d_rgb_32x2x1_r50_3d_kinetics400_100e.py 8 12345 --validate
python -m torch.distributed.launch --master_port $3 --nproc_per_node=$2 $(dirname "$0")/train.py $1 --launcher pytorch ${@:4}
