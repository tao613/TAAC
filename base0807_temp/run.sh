#!/bin/bash

# show ${RUNTIME_SCRIPT_DIR}
echo ${RUNTIME_SCRIPT_DIR}
# enter train workspace
cd ${RUNTIME_SCRIPT_DIR}

# write your code below
python -u main.py --batch_size 256 --num_epochs 4 --dropout_rate 0.2 --hidden_units 64 --num_blocks 2 --num_heads 2 --norm_first --maxlen 101 --l2_emb 1e-6 --mm_emb_id 81
