#!/bin/bash

# 完整优化版本（启用所有优化功能）
echo "运行完整优化版本训练..."
cd ${RUNTIME_SCRIPT_DIR}

python -u main.py \
    --use_adaptive_similarity \
    --use_smart_sampling \
    --use_curriculum_learning \
    --similarity_types dot cosine scaled bilinear \
    --curriculum_schedule cosine \
    --norm_first \
    --batch_size 128 \
    --lr 0.001 \
    --num_epochs 10 \
    --hidden_units 64 \
    --num_heads 2 \
    --num_blocks 2 \
    --dropout_rate 0.2 \
    --l2_emb 0.001
