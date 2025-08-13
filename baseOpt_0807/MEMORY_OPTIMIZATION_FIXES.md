# 内存优化解决方案

## 问题分析

训练过程中出现了CUDA内存不足错误：
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 38.23 GiB of which 4.00 MiB is free.
```

**原因分析：**
1. **In-batch Negatives策略**内存消耗较大：为每个查询构建负样本矩阵
2. **原始实现**存在内存效率问题：使用Python列表和循环操作
3. **批次大小过大**：batch_size=256在使用in-batch negatives时内存占用过高

## 解决方案

### 1. 算法优化

**原始实现问题：**
- 使用Python列表存储负样本
- 多次内存分配和复制
- 没有限制负样本数量

**优化后实现：**
- 直接使用张量操作，避免Python列表
- 限制每个查询的最大负样本数量
- 使用随机采样而非全量负样本

### 2. 参数调整

**新增参数：**
- `--max_negatives_per_query`：限制每个查询的负样本数量（默认100）
- 降低batch_size：从256降到64
- 提供可选的关闭in-batch negatives的选项

**运行配置：**

#### 方案1：内存优化的In-batch Negatives（推荐）
```bash
bash run.sh  # batch_size=64, max_negatives_per_query=50
```

#### 方案2：内存友好的传统方法（备用）
```bash
bash run_lite.sh  # 关闭in-batch negatives，使用BCE Loss
```

### 3. 内存使用对比

| 配置 | Batch Size | 负采样策略 | 损失函数 | 预估内存使用 |
|------|-----------|-----------|----------|-------------|
| 原始 | 256 | In-batch | InfoNCE | ~38GB (溢出) |
| 优化1 | 64 | In-batch (限制50个) | InfoNCE | ~20GB |
| 备用 | 128 | 随机采样 | BCE | ~15GB |

### 4. 性能预期

- **优化1**：保持大部分in-batch negatives的优势，轻微的性能损失
- **备用方案**：回到传统方法，但仍有第一阶段的所有优化（时间特征、学习率调度、早停等）

## 使用建议

1. **首选**：使用`run.sh`（优化的in-batch negatives）
2. **如果内存仍然不足**：使用`run_lite.sh`
3. **可以根据GPU内存情况调整**：
   - `--batch_size`：32-128之间
   - `--max_negatives_per_query`：20-100之间

## 文件修改清单

- ✅ `model.py`: 优化in-batch negatives实现
- ✅ `eval/model.py`: 同步优化
- ✅ `main.py`: 添加内存控制参数
- ✅ `run.sh`: 调整为内存友好参数
- ✅ `run_lite.sh`: 提供传统方法备用方案

这些优化确保了在有限的GPU内存下仍能享受到推荐的优化策略。
