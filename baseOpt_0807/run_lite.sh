#!/bin/bash

# show ${RUNTIME_SCRIPT_DIR}
echo ${RUNTIME_SCRIPT_DIR}
# enter train workspace
cd ${RUNTIME_SCRIPT_DIR}

# 内存友好的配置：关闭in-batch negatives，使用传统BCE Loss
# 这个版本内存使用更少，但优化效果相对较弱
python -u main.py --batch_size 128 --num_epochs 4 --dropout_rate 0.2 --hidden_units 64 --num_blocks 2 --num_heads 2 --norm_first --maxlen 101 --l2_emb 1e-6 --mm_emb_id 81 --no-use_inbatch_negatives --no-use_infonce_loss
