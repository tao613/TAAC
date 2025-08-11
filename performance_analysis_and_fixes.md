# 模型性能分析与优化方案

## 🔍 当前性能问题分析

### 性能指标对比
| 模型版本 | Score | NDCG@10 | HitRate@10 | 说明 |
|---------|-------|---------|------------|------|
| 当前模型 | 0.0146 | 0.0111 | 0.0225 | 复杂优化版本 |
| baseModel | ~0.02 | - | - | 简单基线版本 |

### 🎯 核心问题识别

1. **过度工程化**：引入了太多同时运行的优化策略
   - 自适应相似度计算
   - 智能负采样
   - 课程学习  
   - semantic_id特征

2. **特征维度不匹配**：新增特征可能破坏了原有平衡
   - semantic_id等新特征增加了模型复杂度
   - itemdnn输入维度可能不匹配

3. **训练不稳定**：多种优化策略相互干扰
   - 自适应相似度权重需要时间收敛
   - 课程学习改变了训练样本分布

## 🚀 分阶段解决方案

### 阶段一：紧急修复（立即执行）

#### 1.1 回归基线配置
```bash
# 使用最简单的配置重新训练
python main.py \
    --batch_size 128 \
    --lr 0.001 \
    --hidden_units 32 \
    --num_blocks 1 \
    --num_epochs 3 \
    --num_heads 1 \
    --dropout_rate 0.2 \
    --norm_first \
    --mm_emb_id 81
    # 不使用任何优化参数
```

#### 1.2 确保使用原始相似度计算
- 确认模型中`hasattr(self, 'adaptive_similarity')`返回False
- 使用简单点积：`(log_feats * pos_embs).sum(dim=-1)`

#### 1.3 移除复杂特征
- 暂时不使用semantic_id特征（--mm_emb_id只用81）
- 保持与default模型相同的特征配置

### 阶段二：单点优化验证（逐步测试）

#### 2.1 测试Pre-LayerNorm效果
```bash
# 基线 vs Pre-LayerNorm
python main.py --norm_first  # 已默认启用
```

#### 2.2 测试自适应相似度
```bash
# 只测试简单的自适应相似度
python main.py --use_adaptive_similarity --similarity_types dot cosine
```

#### 2.3 测试课程学习
```bash
# 单独测试课程学习
python main.py --use_curriculum_learning --curriculum_schedule linear
```

### 阶段三：优化组合（确认各自有效后）

#### 3.1 最佳组合筛选
基于单点测试结果，选择真正有效的优化：
- 如果Pre-LayerNorm有效 → 保留
- 如果自适应相似度有效 → 逐步增加相似度类型
- 如果课程学习有效 → 调优调度策略

#### 3.2 参数微调
```bash
# 示例：如果自适应相似度有效
python main.py \
    --use_adaptive_similarity \
    --similarity_types dot cosine scaled \
    --lr 0.0005  # 可能需要降低学习率
```

## 🔧 具体修复步骤

### 步骤1：立即验证基线性能

1. **运行简化训练**：
   ```bash
   ./run_baseline_simple.sh
   ```

2. **验证性能恢复**：
   ```bash
   # 训练完成后立即进行推理测试
   cd evalu && python infer.py
   ```

3. **预期结果**：NDCG@10应该回到0.02左右

### 步骤2：问题根因定位

如果基线性能仍然不佳，检查：

1. **数据一致性**：
   ```bash
   # 确认使用相同的训练数据
   ls -la $TRAIN_DATA_PATH
   ```

2. **模型权重初始化**：
   ```python
   # 检查是否正确初始化
   model.pos_emb.weight.data[0, :] = 0
   model.item_emb.weight.data[0, :] = 0
   ```

3. **损失函数**：
   ```python
   # 确认使用BCE损失
   bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
   ```

### 步骤3：渐进式优化

1. **单个特性验证**：每次只启用一个优化特性
2. **性能对比**：与基线比较，确认提升
3. **组合测试**：只组合确认有效的优化

## 📊 监控指标

### 训练过程监控
- Loss收敛曲线
- 验证集Loss
- 自适应相似度权重变化（如果启用）

### 推理性能监控  
- NDCG@10：主要指标
- HitRate@10：召回能力
- Score：综合评分

## 🎯 预期目标

1. **短期目标**：恢复到baseModel的0.02性能
2. **中期目标**：通过单点优化达到0.025+
3. **长期目标**：通过组合优化达到0.03+

## ⚠️ 风险控制

1. **避免同时测试多个优化**：容易混淆效果来源
2. **保留基线checkpoint**：确保随时可以回退
3. **充分验证**：每个优化都要在多次运行中验证稳定性

---

**执行建议**：先运行`./run_baseline_simple.sh`进行基线恢复，确认性能恢复后再考虑逐步优化。
