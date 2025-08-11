#!/bin/bash

# 课程学习测试脚本
# 测试不同的课程学习调度策略
echo "运行课程学习测试版本..."
cd ${RUNTIME_SCRIPT_DIR}

echo "=== 测试线性课程学习 ==="
python -u main.py \
    --use_adaptive_similarity \
    --use_smart_sampling \
    --use_curriculum_learning \
    --similarity_types dot cosine \
    --curriculum_schedule linear \
    --norm_first \
    --batch_size 128 \
    --lr 0.001 \
    --num_epochs 5 \
    --hidden_units 32 \
    --dropout_rate 0.2 \
    --l2_emb 0.0

echo "\n=== 测试余弦课程学习 ==="
python -u main.py \
    --use_adaptive_similarity \
    --use_smart_sampling \
    --use_curriculum_learning \
    --similarity_types dot cosine \
    --curriculum_schedule cosine \
    --norm_first \
    --batch_size 128 \
    --lr 0.001 \
    --num_epochs 5 \
    --hidden_units 32 \
    --dropout_rate 0.2 \
    --l2_emb 0.0

echo "\n=== 测试指数课程学习 ==="
python -u main.py \
    --use_adaptive_similarity \
    --use_smart_sampling \
    --use_curriculum_learning \
    --similarity_types dot cosine \
    --curriculum_schedule exponential \
    --norm_first \
    --batch_size 128 \
    --lr 0.001 \
    --num_epochs 5 \
    --hidden_units 32 \
    --dropout_rate 0.2 \
    --l2_emb 0.0

echo "课程学习测试完成！"
