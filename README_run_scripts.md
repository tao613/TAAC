# 训练脚本使用指南

我们提供了多个训练脚本，对应不同的优化级别和使用场景：

## 脚本说明

### 1. `run.sh` - 推荐优化版本
**用途**: 生产环境推荐配置，平衡性能和稳定性
**特点**: 启用核心优化功能，包括渐进式训练
```bash
bash run.sh
```
**启用的优化**:
- ✅ 自适应相似度计算 (dot + cosine + scaled)
- ✅ 智能负采样
- ✅ 课程学习 (linear schedule)
- ✅ Pre-LayerNorm
- 🔧 保守的超参数设置

### 2. `run_baseline.sh` - 原始基线版本
**用途**: 作为性能对比的基线，或在优化版本出现问题时的备选方案
**特点**: 完全使用原始方法，不启用任何优化
```bash
bash run_baseline.sh
```
**配置**:
- ❌ 不使用任何优化
- 🔧 原始的BCE Loss + 随机负采样
- 🔧 标准点积相似度计算

### 3. `run_lite.sh` - 轻量优化版本
**用途**: 保守的优化尝试，适合初次测试优化效果
**特点**: 仅启用最核心的相似度优化
```bash
bash run_lite.sh
```
**启用的优化**:
- ✅ 自适应相似度计算 (仅 dot + cosine)
- ❌ 不使用智能负采样
- ✅ Pre-LayerNorm
- 🔧 保守的参数配置

### 4. `run_full.sh` - 完整优化版本
**用途**: 使用所有优化功能，适合追求最佳性能
**特点**: 启用所有可用的优化功能
```bash
bash run_full.sh
```
**启用的优化**:
- ✅ 自适应相似度计算 (dot + cosine + scaled + bilinear)
- ✅ 智能负采样
- ✅ 课程学习 (cosine schedule)
- ✅ Pre-LayerNorm
- 🔧 更大的模型参数 (hidden_units=64, num_heads=2, num_blocks=2)

### 5. `run_comparison.sh` - 对比测试版本
**用途**: 直接对比baseline和优化版本的效果

### 6. `run_curriculum.sh` - 课程学习测试版本
**用途**: 测试不同的课程学习调度策略效果
**特点**: 依次测试linear、cosine、exponential三种调度策略
```bash
bash run_curriculum.sh
```
**测试内容**:
- 🔬 线性调度 (difficulty增长均匀)
- 🔬 余弦调度 (difficulty增长先慢后快再慢)
- 🔬 指数调度 (difficulty增长先慢后快)
**特点**: 先后运行两个版本，便于效果对比
```bash
bash run_comparison.sh
```
**流程**:
1. 🚀 运行5轮基线版本训练
2. 🚀 运行5轮优化版本训练
3. 📊 输出对比提示信息

## 推荐使用流程

### 第一次使用
```bash
# 1. 先运行轻量版本测试
bash run_lite.sh

# 2. 如果效果良好，尝试推荐版本
bash run.sh

# 3. 如果想要最佳性能，使用完整版本
bash run_full.sh
```

### 性能对比
```bash
# 运行对比测试，直观比较优化效果
bash run_comparison.sh
```

### 问题排查
```bash
# 如果优化版本出现问题，回退到基线版本
bash run_baseline.sh
```

## 超参数说明

| 参数 | baseline | lite | 推荐 | full | curriculum | 说明 |
|-----|----------|------|------|------|-----------|------|
| `hidden_units` | 32 | 32 | 32 | 64 | 32 | 隐藏层维度 |
| `num_heads` | 1 | 1 | 1 | 2 | 1 | 注意力头数 |
| `num_blocks` | 1 | 1 | 1 | 2 | 1 | Transformer层数 |
| `batch_size` | 128 | 128 | 128 | 128 | 128 | 批次大小 |
| `lr` | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 学习率 |
| `num_epochs` | 5 | 5 | 5 | 10 | 5 | 训练轮数 |
| `dropout_rate` | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | Dropout比例 |
| `l2_emb` | 0.0 | 0.0 | 0.0 | 0.001 | 0.0 | L2正则化 |
| **优化功能** | | | | | | |
| `use_adaptive_similarity` | ❌ | ✅ | ✅ | ✅ | ✅ | 自适应相似度 |
| `use_smart_sampling` | ❌ | ❌ | ✅ | ✅ | ✅ | 智能负采样 |
| `use_curriculum_learning` | ❌ | ❌ | ✅ | ✅ | ✅ | 课程学习 |
| `curriculum_schedule` | - | - | linear | cosine | 全部测试 | 难度调度策略 |

## 监控和调试

### TensorBoard监控
所有脚本都会在TensorBoard中记录训练指标：
```bash
# 查看训练日志
tensorboard --logdir $TRAIN_TF_EVENTS_PATH
```

**重要指标**:
- `Loss/train`: 训练损失
- `Loss/valid`: 验证损失
- `Similarity/alpha`: 点积相似度权重
- `Similarity/beta`: 余弦相似度权重
- `Similarity/gamma`: 缩放点积权重
- `Similarity/temperature`: 温度参数
- `Curriculum/difficulty_factor`: 课程学习难度因子 (0-1)
- `Curriculum/epoch_progress`: 训练进度

### 日志文件
训练日志保存在 `$TRAIN_LOG_PATH/train.log`：
```bash
# 实时查看训练日志
tail -f $TRAIN_LOG_PATH/train.log
```

### 模型检查点
模型检查点保存在 `$TRAIN_CKPT_PATH/`：
- `model.pt`: 模型权重
- `config.json`: 训练配置

## 性能预期

根据优化策略，预期性能提升：

| 版本 | AUC提升 | 召回率提升 | 训练稳定性 | 计算开销 | 特色功能 |
|------|---------|------------|------------|----------|----------|
| baseline | - | - | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 原始算法 |
| lite | +1-2% | +1-2% | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 相似度优化 |
| 推荐 | +3-5% | +3-5% | ⭐⭐⭐⭐ | ⭐⭐⭐ | 负采样+课程学习 |
| full | +4-7% | +4-7% | ⭐⭐⭐ | ⭐⭐ | 全部优化+大模型 |
| curriculum | +3-5% | +3-5% | ⭐⭐⭐⭐ | ⭐⭐⭐ | 课程学习对比 |

## 故障排除

### 如果训练失败
1. 检查环境变量是否设置正确
2. 尝试运行 `run_baseline.sh` 确认基础环境正常
3. 检查GPU内存是否足够（full版本需要更多内存）

### 如果性能没有提升
1. 先运行 `run_comparison.sh` 直接对比
2. 检查TensorBoard中的相似度权重是否在学习
3. 尝试调整学习率或训练轮数

### 如果内存不足
1. 减少 `batch_size`
2. 使用 `run_lite.sh` 而不是 `run_full.sh`
3. 减少 `hidden_units` 大小
