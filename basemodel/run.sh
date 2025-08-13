#!/bin/bash

# show ${RUNTIME_SCRIPT_DIR}
echo ${RUNTIME_SCRIPT_DIR}
# enter train workspace
cd ${RUNTIME_SCRIPT_DIR}

# write your code below
# python -u main.py

# InfoNCE Loss版本 (推荐)
python -u main.py --batch_size 256 --lr 0.0007 --num_epochs 4 --dropout_rate 0.2 --hidden_units 64 --num_blocks 2 --num_heads 2 --norm_first --maxlen 101 --l2_emb 1e-6 --mm_emb_id 81 --use_infonce --infonce_temperature 0.07

# 原始BCE Loss版本 (用于对比)
# python -u main.py --batch_size 256 --lr 0.0007 --num_epochs 4 --dropout_rate 0.2 --hidden_units 64 --num_blocks 2 --num_heads 2 --norm_first --maxlen 101 --l2_emb 1e-6 --mm_emb_id 81

# Baseline+
# python -u main.py --batch_size 256 --lr 0.001 --num_epochs 5 --dropout_rate 0.2 --hidden_units 32 --num_blocks 1 --num_heads 1 --norm_first --maxlen 101 --l2_emb 0.0 --mm_emb_id 81

# Capacity boost
# python -u main.py --batch_size 256 --lr 0.0007 --num_epochs 4 --dropout_rate 0.2 --hidden_units 64 --num_blocks 2 --num_heads 2 --norm_first --maxlen 101 --l2_emb 1e-6 --mm_emb_id 81

# Multimodal enrichment
# python -u main.py --batch_size 256 --lr 0.0007 --num_epochs 4 --dropout_rate 0.2 --hidden_units 64 --num_blocks 2 --num_heads 2 --norm_first --maxlen 101 --l2_emb 1e-6 --mm_emb_id 81 82

# Regularized deeper
# python -u main.py --batch_size 256 --lr 0.0006 --num_epochs 6 --dropout_rate 0.3 --hidden_units 64 --num_blocks 2 --num_heads 2 --norm_first --maxlen 101 --l2_emb 2e-6 --mm_emb_id 81