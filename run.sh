#!/bin/bash

# show ${RUNTIME_SCRIPT_DIR}
echo ${RUNTIME_SCRIPT_DIR}
# enter train workspace
cd ${RUNTIME_SCRIPT_DIR}

# 优化版本训练脚本
# 启用距离计算优化、智能负采样和课程学习
python -u main.py \
    --use_adaptive_similarity \
    --use_smart_sampling \
    --use_curriculum_learning \
    --similarity_types dot cosine scaled \
    --curriculum_schedule linear \
    --norm_first \
    --batch_size 128 \
    --lr 0.001 \
    --num_epochs 5 \
    --hidden_units 32 \
    --dropout_rate 0.2 \
    --l2_emb 0.0