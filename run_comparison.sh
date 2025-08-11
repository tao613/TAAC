#!/bin/bash

# 对比测试脚本：先运行baseline，再运行优化版本
echo "开始对比测试..."
cd ${RUNTIME_SCRIPT_DIR}

echo "=================================================="
echo "第一轮：运行基础版本（Baseline）"
echo "=================================================="
python -u main.py \
    --batch_size 128 \
    --lr 0.001 \
    --num_epochs 5 \
    --hidden_units 32 \
    --dropout_rate 0.2 \
    --l2_emb 0.0

echo ""
echo "=================================================="
echo "第二轮：运行优化版本（With Optimizations）"
echo "=================================================="
python -u main.py \
    --use_adaptive_similarity \
    --use_smart_sampling \
    --similarity_types dot cosine scaled \
    --norm_first \
    --batch_size 128 \
    --lr 0.001 \
    --num_epochs 5 \
    --hidden_units 32 \
    --dropout_rate 0.2 \
    --l2_emb 0.0

echo ""
echo "=================================================="
echo "对比测试完成！请查看TensorBoard日志比较两次训练效果"
echo "=================================================="
