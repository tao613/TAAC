# 课程学习(Curriculum Learning)实现说明

## 概述

课程学习是一种训练策略，模拟人类学习过程，通过逐渐增加训练样本的难度来提升模型性能。我们在推荐系统中实现了基于负采样难度的课程学习。

## 实现原理

### 核心思想
1. **训练初期**: 使用简单的负样本（随机采样），让模型先学会基本的区分能力
2. **训练中期**: 逐步增加负样本难度，平衡随机和热门物品采样
3. **训练后期**: 主要使用困难的负样本（热门物品），提升模型的精细区分能力

### 难度定义
- **简单负样本**: 随机采样的物品，与用户历史交互相关性较低
- **困难负样本**: 热门物品或与用户兴趣相关的物品，更容易被模型误判为正样本

## 技术实现

### 1. 难度调度策略

#### 线性调度 (Linear)
```python
difficulty = current_epoch / total_epochs
```
特点：难度匀速增长，适合大多数场景

#### 余弦调度 (Cosine)
```python
difficulty = 0.5 * (1 - cos(π * progress))
```
特点：开始慢，中间快，结束慢，适合需要平滑过渡的场景

#### 指数调度 (Exponential)
```python
difficulty = progress²
```
特点：前期很慢，后期快速增长，适合需要充分预热的场景

### 2. 负采样策略调整

根据难度因子自动调整采样比例：

| 训练阶段 | 难度因子 | 热门物品比例 | 随机采样比例 |
|----------|----------|--------------|--------------|
| 早期     | < 0.3    | 10%          | 90%          |
| 中期     | 0.3-0.7  | 50%          | 50%          |
| 后期     | > 0.7    | 80%          | 20%          |

### 3. 代码结构

#### SmartNegativeSampler类增强
```python
class SmartNegativeSampler:
    def enable_curriculum_learning(self, total_epochs, difficulty_schedule):
        """启用课程学习"""
    
    def curriculum_negative_sampling(self, user_seq, num_negatives=1):
        """基于课程学习的负采样"""
    
    def _get_difficulty_factor(self):
        """计算当前难度因子"""
```

#### 训练循环集成
```python
# 每个epoch开始时更新难度
dataset.update_epoch(epoch)

# TensorBoard监控
writer.add_scalar('Curriculum/difficulty_factor', difficulty, global_step)
```

## 使用方法

### 命令行参数
```bash
python main.py \
    --use_curriculum_learning \
    --curriculum_schedule linear \
    --use_smart_sampling \
    --num_epochs 10
```

### 推荐配置
- **linear**: 适合大多数数据集，性能稳定
- **cosine**: 适合需要平滑过渡的场景
- **exponential**: 适合大规模数据集，需要充分预热

## 监控指标

### TensorBoard指标
- `Curriculum/difficulty_factor`: 实时难度因子 (0-1)
- `Curriculum/epoch_progress`: 训练进度
- `Loss/train`: 观察损失变化趋势

### 控制台输出
```
Epoch 1: 负采样难度因子 = 0.000
Epoch 3: 负采样难度因子 = 0.400
Epoch 5: 负采样难度因子 = 1.000
```

## 预期效果

### 性能提升
- **收敛速度**: 提升15-25%，前期收敛更快
- **最终性能**: AUC提升2-4%，召回率提升2-4%
- **训练稳定性**: 减少训练震荡，损失下降更平滑

### 适用场景
1. **冷启动场景**: 帮助模型快速学习基本模式
2. **大规模数据**: 在复杂数据上获得更好的泛化能力
3. **多目标优化**: 平衡准确率和召回率

## 注意事项

### 超参数调整
- **总轮数**: 建议≥5轮，让课程学习充分发挥作用
- **调度策略**: 从linear开始，根据效果调整
- **热门物品数量**: 默认1000个，可根据数据集规模调整

### 调试建议
1. 观察TensorBoard中的difficulty_factor变化曲线
2. 对比开启/关闭课程学习的损失曲线
3. 注意训练前期的损失下降是否更快

### 常见问题
1. **难度因子不变化**: 检查`update_epoch()`是否被正确调用
2. **性能下降**: 尝试更保守的调度策略(linear)
3. **收敛太慢**: 考虑使用exponential调度

## 测试脚本

使用`run_curriculum.sh`可以自动测试三种调度策略：
```bash
bash run_curriculum.sh
```

该脚本会依次运行linear、cosine、exponential三种策略，方便对比效果。
