#!/bin/bash

# 轻量优化版本（仅启用自适应相似度）
echo "运行轻量优化版本训练..."
cd ${RUNTIME_SCRIPT_DIR}

python -u main.py \
    --use_adaptive_similarity \
    --similarity_types dot cosine \
    --norm_first \
    --batch_size 128 \
    --lr 0.001 \
    --num_epochs 10 \
    --hidden_units 32 \
    --dropout_rate 0.2 \
    --l2_emb 0.0
