#!/bin/bash

DATA="${1-MUTAG}"
centers="${2-2}"
DEVICE="${3-1}"
BATCH_SIZE=32


python3 opt.py --dataset ${DATA} --device ${DEVICE} --fold_idx 0 --batch_size ${BATCH_SIZE}
