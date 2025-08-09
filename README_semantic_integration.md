# Semantic ID Integration in Inference

本文档说明如何在推理阶段使用semantic_id特征来增强推荐效果。

## 🎯 功能概述

在`evalu/infer.py`中集成了semantic_id特征的完整支持，包括：

1. **自动加载semantic_id特征**
2. **智能特征填充**
3. **统计信息输出**
4. **向后兼容性保证**

## 🔄 工作流程

### 1. **Semantic ID加载阶段**
```python
# 从测试数据集中加载semantic_id特征
semantic_id_dict = test_dataset.semantic_id_dict
if semantic_id_dict:
    print(f"成功加载 {len(semantic_id_dict)} 个物品的semantic_id特征")
else:
    print("未找到semantic_id特征，将使用默认值")
```

### 2. **候选物品特征处理**
```python
# 为每个候选物品添加semantic_id特征
for feat_id in feat_types.get('semantic_array', []):
    if feat_id not in feature:
        # 优先使用RQ-VAE生成的semantic_id
        if semantic_id_dict and creative_id in semantic_id_dict:
            feature[feat_id] = semantic_id_dict[creative_id]
            semantic_id_used_count += 1
        else:
            # 回退到默认值
            feature[feat_id] = feat_default_value[feat_id]
```

### 3. **使用统计输出**
```python
semantic_coverage = semantic_id_used_count / total_candidates * 100
print(f"Semantic ID使用统计:")
print(f"  总候选物品数: {total_candidates}")
print(f"  使用semantic_id的物品数: {semantic_id_used_count}")
print(f"  覆盖率: {semantic_coverage:.2f}%")
```

## 📊 预期输出示例

### 情况1：有semantic_id文件
```
成功加载 50000 个物品的semantic_id特征
Processing candidate items...
Semantic ID使用统计:
  总候选物品数: 100000
  使用semantic_id的物品数: 48000
  覆盖率: 48.00%
```

### 情况2：无semantic_id文件（向后兼容）
```
未找到semantic_id特征，将使用默认值
Processing candidate items...
Semantic ID使用统计:
  总候选物品数: 100000
  使用semantic_id的物品数: 0
  覆盖率: 0.00%
```

## 🚀 使用方法

### 方法1：完整功能（推荐）
```bash
# 1. 生成semantic_id特征
python train_rqvae.py \
    --data_dir /path/to/train/data \
    --output_dir /path/to/semantic_output \
    --mm_emb_id 81 82 83

# 2. 复制到推理数据目录
cp /path/to/semantic_output/semantic_id_dict.json /path/to/eval/data/

# 3. 运行推理
python evalu/infer.py
```

### 方法2：向后兼容模式
```bash
# 直接运行推理（使用默认semantic_id值）
python evalu/infer.py
```

## 🔍 技术细节

### Semantic ID特征格式
```json
{
  "item_12345": [156, 89, 200, 45],
  "item_67890": [78, 234, 12, 167],
  ...
}
```

每个物品对应一个长度为`num_quantizers`的整数序列，表示其在不同量化器中的语义ID。

### 默认值处理
```python
# 默认semantic_id值
feat_default_value['semantic_id'] = [0]  # 表示"无语义ID"
feat_statistics['semantic_id'] = 256     # 与codebook_size一致
```

### 模型集成
Semantic ID特征在模型中被处理为：
1. **Embedding查找**: `sparse_emb['semantic_id'](tensor_feature)`
2. **序列聚合**: `.sum(2)` 对语义ID序列求和
3. **特征融合**: 与其他物品特征拼接后通过`itemdnn`

## 📈 性能影响

### 正面影响
1. **语义增强**: 相似物品具有相似的semantic_id，增强表示能力
2. **冷启动改善**: 新物品可以通过语义聚类获得更好的初始表示
3. **泛化能力**: 语义抽象提高模型对未见物品的推理能力

### 计算开销
1. **内存**: 每个物品增加约4-8个整数的存储（取决于num_quantizers）
2. **计算**: 增加一次embedding查找和求和操作，开销极小
3. **I/O**: 加载semantic_id_dict.json文件的一次性开销

## 🐛 故障排除

### 问题1：semantic_id文件未找到
```
警告: 未找到semantic_id文件 /data/semantic_id_dict.json
```
**解决方案**: 运行`train_rqvae.py`生成semantic_id特征，或继续使用默认值。

### 问题2：覆盖率过低
```
覆盖率: 5.00%  # 过低
```
**可能原因**:
- semantic_id训练数据与推理数据物品重叠度低
- 物品ID格式不匹配

**解决方案**:
- 检查物品ID映射是否正确
- 使用更全面的训练数据重新训练RQ-VAE

### 问题3：性能未提升
**可能原因**:
- RQ-VAE训练不充分
- semantic_id维度设置不当
- 需要重新训练推荐模型

**解决方案**:
- 调整RQ-VAE参数重新训练
- 增加训练轮数或调整学习率
- 使用包含semantic_id的数据重新训练推荐模型

## 📝 配置参数

### RQ-VAE相关参数
```python
--num_quantizers 4      # 语义ID序列长度
--codebook_size 256     # 每个位置的取值范围
--commitment_cost 0.25  # 量化损失权重
```

### 推理相关参数
```python
feat_statistics['semantic_id'] = 256  # 需与codebook_size一致
feat_default_value['semantic_id'] = [0]  # 默认值
```

## 🎉 预期效果

集成semantic_id特征后，推荐系统应该表现出：

1. **更好的物品表示**: 相似物品在语义空间中距离更近
2. **改善的推荐质量**: 特别是对长尾物品和新物品
3. **增强的泛化能力**: 模型能更好地理解物品间的语义关系
4. **保持的计算效率**: 几乎不增加推理时间

---

通过这种渐进式的集成方式，系统既能利用semantic_id的优势，又保持了对旧版本的兼容性，确保平滑的功能升级。
