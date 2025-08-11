#!/bin/bash

# 基础版本（原始方法，不启用任何优化）
echo "运行基础版本训练..."
cd ${RUNTIME_SCRIPT_DIR}

python -u main.py \
    --batch_size 128 \
    --lr 0.001 \
    --num_epochs 10 \
    --hidden_units 32 \
    --dropout_rate 0.2 \
    --l2_emb 0.0
