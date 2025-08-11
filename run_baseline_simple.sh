#!/bin/bash

# 简化版训练脚本 - 回归baseModel性能
# 目标：恢复到0.02左右的NDCG性能

echo "=== 基线性能恢复训练 ==="
echo "移除所有优化特性，使用最简单的配置"

# 设置环境变量（请根据实际环境调整）
export TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/workspace/train_data}"
export TRAIN_LOG_PATH="${TRAIN_LOG_PATH:-/workspace/logs}"
export TRAIN_TF_EVENTS_PATH="${TRAIN_TF_EVENTS_PATH:-/workspace/tensorboard}"
export TRAIN_CKPT_PATH="${TRAIN_CKPT_PATH:-/workspace/checkpoints}"

# 创建必要目录
mkdir -p "$TRAIN_LOG_PATH" "$TRAIN_TF_EVENTS_PATH" "$TRAIN_CKPT_PATH"

# 基线训练配置 - 与default/main.py保持一致
python main.py \
    --batch_size 128 \
    --lr 0.001 \
    --maxlen 101 \
    --hidden_units 32 \
    --num_blocks 1 \
    --num_epochs 3 \
    --num_heads 1 \
    --dropout_rate 0.2 \
    --l2_emb 0.0 \
    --device cuda \
    --norm_first \
    --mm_emb_id 81 \
    2>&1 | tee "$TRAIN_LOG_PATH/baseline_simple.log"

echo "=== 训练完成 ==="
echo "检查点保存在: $TRAIN_CKPT_PATH"
echo "日志保存在: $TRAIN_LOG_PATH/baseline_simple.log"
echo ""
echo "下一步："
echo "1. 运行推理验证性能是否恢复到0.02水平"
echo "2. 如果性能恢复，再逐步测试单个优化特性"
