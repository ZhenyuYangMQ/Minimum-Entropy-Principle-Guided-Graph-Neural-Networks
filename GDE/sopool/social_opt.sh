#!/bin/bash

DATA="${1-COLLAB}"
DEVICE="${2-0}"
BATCH_SIZE=32

python3 opt.py --dataset ${DATA} --device ${DEVICE} --fold_idx 0 --batch_size ${BATCH_SIZE} --degree_as_tag
