# InfoNCE + In-Batch Negatives 改进实现

## 🎯 改进要点

根据用户要求，对InfoNCE loss进行了以下关键改进：

### 1. 正样本相似度计算 ✅
- **使用余弦相似度**: `F.cosine_similarity(seq_embs_norm, pos_embs_norm, dim=-1)`
- **L2归一化**: 对序列表示和正样本表示进行L2归一化，确保计算稳定性

### 2. In-Batch Negatives策略 ✅
- **负样本池构建**: 使用batch内所有随机负样本作为候选池
- **避免False Negatives**: 只使用随机负样本(neg_embs)，不包括其他样本的正样本(pos_embs)
- **高效计算**: 通过矩阵乘法一次性计算所有负样本相似度

### 3. Padding位置掩码处理 ✅
- **损失掩码**: 使用`loss_mask`过滤padding位置
- **有效位置**: 只对`loss_mask==1`的位置计算损失
- **边界处理**: 当没有有效位置时返回零损失

### 4. 正样本位置标签 ✅
- **标签为0**: 正样本在拼接后的第0位，交叉熵标签为0
- **logits拼接**: `[pos_logits, neg_logits]` 确保正样本在第一位

### 5. 推理一致性 ✅
- **归一化一致**: 推理时的user embedding和item embedding都进行L2归一化
- **检测机制**: 通过`hasattr(self, 'temp')`自动检测是否使用InfoNCE
- **向后兼容**: 不使用InfoNCE时保持原有行为

## 🔧 技术实现细节

### In-Batch Negatives构建
```python
# 1. 将所有随机负样本作为候选池
neg_pool = neg_embs_norm.view(batch_size * maxlen, hidden_size)

# 2. 计算每个位置与所有负样本的相似度
seq_flat = seq_embs_norm.view(batch_size * maxlen, hidden_size)
neg_logits_all = torch.matmul(seq_flat, neg_pool.t())

# 3. Reshape回原始维度
neg_logits = neg_logits_all.view(batch_size, maxlen, batch_size * maxlen)
```

### 曝光偏差避免
- **只用随机负样本**: 避免使用其他样本的pos_embs作为负样本
- **False Negative问题**: 其他用户的正样本可能恰好是当前用户的真实偏好
- **安全策略**: 只使用确认的随机负样本，保证训练信号质量

### Padding处理
```python
# 1. 获取有效位置掩码
valid_positions = loss_mask.bool()

# 2. 只提取有效位置的logits
valid_pos_logits = pos_logits[valid_positions]
valid_neg_logits = neg_logits[valid_positions]

# 3. 边界情况处理
if valid_pos_logits.size(0) == 0:
    return torch.tensor(0.0, device=seq_embs.device, requires_grad=True)
```

### 推理时归一化
```python
# 用户序列表示归一化
if hasattr(self, 'temp'):  # InfoNCE训练标志
    final_feat = final_feat / (final_feat.norm(dim=-1, keepdim=True) + 1e-8)

# 候选item embedding归一化
if hasattr(self, 'temp'):
    batch_emb = batch_emb / (batch_emb.norm(dim=-1, keepdim=True) + 1e-8)
```

## 📊 关键优势

### 1. 更高质量的负样本
- **In-Batch策略**: 每个位置有`batch_size * maxlen`个负样本
- **避免偏差**: 不使用可能的false negatives
- **计算高效**: 矩阵乘法并行计算

### 2. 更稳定的训练
- **L2归一化**: 防止embedding爆炸，提升训练稳定性
- **Padding过滤**: 只对有效位置计算损失，避免噪声
- **温度控制**: 通过温度参数(0.07)控制分布锐度

### 3. 推理一致性
- **归一化对齐**: 训练和推理使用相同的归一化策略
- **自动检测**: 无需手动指定，自动保持一致性
- **向后兼容**: 不影响BCE loss模式的使用

## 🚀 使用方法

### 训练命令
```bash
cd basemodel
./run.sh  # 使用温度参数0.07的InfoNCE loss
```

### 推理命令
```bash
cd basemodel
python -u eval/infer.py  # 自动使用一致的归一化
```

## 📈 预期效果

1. **更好的表示学习**: 余弦相似度 + L2归一化
2. **更强的负样本对比**: In-batch negatives增加难度
3. **更准确的训练信号**: 避免false negatives
4. **更稳定的收敛**: 温度参数和归一化控制
5. **更好的推理性能**: 训练推理一致性

## ⚠️ 注意事项

1. **批次大小影响**: 更大的batch_size提供更多负样本
2. **温度参数**: 当前使用0.07，可根据效果调整
3. **内存占用**: In-batch策略增加内存使用
4. **计算复杂度**: 负样本数量从1个增加到`batch_size * maxlen`个

---

所有改进已完成，现在InfoNCE loss使用更科学的in-batch negatives策略！
