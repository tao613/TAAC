# InfoNCE 损失函数修复说明

## 问题分析

训练过程中出现异常高的loss值（9-13万），原因分析：

### 1. **原始InfoNCE实现问题**
- ❌ 没有对embedding进行L2归一化
- ❌ 温度参数太小（0.07），导致数值不稳定
- ❌ L2正则化项被计入主损失，影响loss数值

### 2. **具体问题**
```python
# 原始问题代码
pos_sim = torch.sum(query * pos_key, dim=-1) / temperature  # 没有归一化
neg_sim = torch.bmm(neg_keys, query.unsqueeze(-1)).squeeze(-1) / temperature

# 当embedding值很大时，相似度会爆炸性增长
# temperature=0.07太小，进一步放大了这个问题
```

## 解决方案

### 1. **添加L2归一化**
```python
# 修复后的代码
query = F.normalize(query, p=2, dim=-1)
pos_key = F.normalize(pos_key, p=2, dim=-1) 
neg_keys = F.normalize(neg_keys, p=2, dim=-1)
```

**作用：**
- 将所有embedding向量归一化到单位球面
- 相似度计算变为余弦相似度，数值稳定
- 防止embedding幅度影响损失计算

### 2. **调整温度参数**
```python
# 从 temperature=0.07 改为 temperature=0.2
```

**原因：**
- 0.07太小，会导致softmax过于尖锐
- 0.2是对比学习中常用的稳定值
- 更大的温度参数提供更平滑的梯度

### 3. **分离L2正则化**
```python
# 原始代码：直接加到主损失
loss += args.l2_emb * torch.norm(param)

# 修复后：分离计算
l2_loss = args.l2_emb * torch.norm(param)
total_loss = loss + l2_loss
total_loss.backward()

# 日志记录分别记录主损失和总损失
log_json = {'loss': loss.item(), 'l2_loss': l2_loss.item(), 'total_loss': total_loss.item()}
```

## 预期效果

### 修复前的loss特征：
- 数值范围：90,000 - 130,000
- 不稳定：波动很大
- 数值爆炸：embedding未归一化导致

### 修复后的预期loss特征：
- 数值范围：1 - 10（正常的交叉熵损失范围）
- 稳定性：更平滑的下降趋势
- 可解释性：loss值具有明确的概率意义

### InfoNCE Loss的理论范围：
- 最小值：接近0（完美分类）
- 最大值：ln(K+1)，其中K是负样本数量
- 对于50个负样本：最大约为ln(51) ≈ 3.93

## 使用建议

1. **重新启动训练**：修复后需要重新开始训练
2. **监控指标**：关注主损失(loss)而不是总损失(total_loss)
3. **调试参数**：如果loss仍然异常，可以进一步调大temperature到0.5

## 修改文件清单

- ✅ `main.py`: 修复InfoNCE损失函数实现
- ✅ `run.sh`: 更新温度参数
- ✅ 日志格式：分别记录主损失和正则化损失

这些修复确保了InfoNCE损失函数的正确实现和数值稳定性。
