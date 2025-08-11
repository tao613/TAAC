# 基于距离计算优化的保守优化方案

## 问题分析

通过分析原始代码，发现当前的策略（InfoNCE Loss + In-batch Negatives + RQ-VAE semantic_id）过于复杂化，可能反而降低了性能。原始代码使用：

1. **简单的BCE损失** + **随机负采样**
2. **点积相似度**：`(log_feats * pos_embs).sum(dim=-1)`
3. **原始的特征处理方式**

## 优化策略（保守版本）

### 核心思路：不改变BaseModel结构，只优化距离计算和训练策略

## 第一阶段：恢复原始训练策略并优化距离计算

### 1. 恢复原始训练方式
- 恢复BCE Loss + 随机负采样
- 保持原始的forward方法签名
- 移除InfoNCE和In-batch Negatives

### 2. 优化距离计算方法

#### 2.1 引入多种相似度度量
```python
class SimilarityCalculator:
    @staticmethod
    def cosine_similarity(x, y, eps=1e-8):
        """余弦相似度，对向量长度不敏感"""
        x_norm = x / (x.norm(dim=-1, keepdim=True) + eps)
        y_norm = y / (y.norm(dim=-1, keepdim=True) + eps)
        return (x_norm * y_norm).sum(dim=-1)
    
    @staticmethod
    def scaled_dot_product(x, y, scale_factor=None):
        """缩放点积，添加温度参数"""
        if scale_factor is None:
            scale_factor = 1.0 / (x.size(-1) ** 0.5)
        return (x * y).sum(dim=-1) * scale_factor
    
    @staticmethod
    def bilinear_similarity(x, y, weight_matrix):
        """双线性相似度：x^T W y"""
        return torch.bmm(x.unsqueeze(1), torch.bmm(weight_matrix, y.unsqueeze(-1))).squeeze()
```

#### 2.2 自适应相似度融合
```python
def adaptive_similarity(self, seq_repr, item_repr):
    """融合多种相似度度量"""
    # 基础点积
    dot_sim = (seq_repr * item_repr).sum(dim=-1)
    
    # 余弦相似度
    cos_sim = SimilarityCalculator.cosine_similarity(seq_repr, item_repr)
    
    # 缩放点积
    scaled_sim = SimilarityCalculator.scaled_dot_product(seq_repr, item_repr)
    
    # 可学习的权重融合
    similarity = (self.alpha * dot_sim + 
                 self.beta * cos_sim + 
                 self.gamma * scaled_sim)
    
    return similarity
```

### 3. 改进负采样策略

#### 3.1 难负样本采样
```python
def hard_negative_sampling(self, user_seq, num_negatives=5):
    """采样与正样本相似但不在序列中的负样本"""
    # 计算正样本的平均表示
    pos_repr = self.get_average_representation(user_seq)
    
    # 找到与正样本相似的候选
    candidates = self.find_similar_items(pos_repr, top_k=num_negatives*10)
    
    # 过滤掉已在序列中的item
    negatives = [item for item in candidates if item not in user_seq]
    
    return negatives[:num_negatives]
```

#### 3.2 多样化负采样
```python
def diversified_negative_sampling(self, user_seq, ratio_hard=0.5, ratio_random=0.3, ratio_popular=0.2):
    """混合多种负采样策略"""
    negatives = []
    
    # 难负样本
    hard_negs = self.hard_negative_sampling(user_seq, int(num_negatives * ratio_hard))
    negatives.extend(hard_negs)
    
    # 随机负样本
    random_negs = self.random_negative_sampling(user_seq, int(num_negatives * ratio_random))
    negatives.extend(random_negs)
    
    # 热门item负样本
    popular_negs = self.popular_negative_sampling(user_seq, int(num_negatives * ratio_popular))
    negatives.extend(popular_negs)
    
    return negatives
```

### 4. 特征处理优化

#### 4.1 批量特征计算
```python
def batch_feature_processing(self, features):
    """批量处理特征，减少循环开销"""
    # 预分配tensor
    batch_size, seq_len = len(features), len(features[0])
    
    # 批量处理稀疏特征
    sparse_features = self.batch_sparse_features(features)
    
    # 批量处理连续特征
    continual_features = self.batch_continual_features(features)
    
    # 批量处理数组特征
    array_features = self.batch_array_features(features)
    
    return sparse_features, continual_features, array_features
```

#### 4.2 特征缓存机制
```python
class FeatureCache:
    def __init__(self, cache_size=10000):
        self.cache = {}
        self.cache_size = cache_size
        self.access_count = {}
    
    def get_or_compute(self, item_id, compute_func):
        if item_id in self.cache:
            self.access_count[item_id] += 1
            return self.cache[item_id]
        
        if len(self.cache) >= self.cache_size:
            # LRU eviction
            lru_item = min(self.access_count, key=self.access_count.get)
            del self.cache[lru_item]
            del self.access_count[lru_item]
        
        result = compute_func(item_id)
        self.cache[item_id] = result
        self.access_count[item_id] = 1
        
        return result
```

## 第二阶段：渐进式训练策略

### 1. Curriculum Learning
```python
class CurriculumTrainer:
    def __init__(self, model, easy_epochs=5, medium_epochs=5, hard_epochs=5):
        self.model = model
        self.phases = {
            'easy': {'epochs': easy_epochs, 'neg_ratio': 0.8, 'hard_neg_ratio': 0.1},
            'medium': {'epochs': medium_epochs, 'neg_ratio': 0.6, 'hard_neg_ratio': 0.3},
            'hard': {'epochs': hard_epochs, 'neg_ratio': 0.4, 'hard_neg_ratio': 0.6}
        }
    
    def train_phase(self, phase_name, train_loader):
        phase_config = self.phases[phase_name]
        for epoch in range(phase_config['epochs']):
            self.train_epoch(train_loader, phase_config)
```

### 2. 动态负样本数量
```python
def dynamic_negative_sampling(self, epoch, max_epochs):
    """随着训练进展增加负样本数量"""
    base_negatives = 1
    max_negatives = 5
    
    progress = epoch / max_epochs
    current_negatives = int(base_negatives + (max_negatives - base_negatives) * progress)
    
    return current_negatives
```

## 第三阶段：模型集成优化

### 1. 多头相似度计算
```python
class MultiHeadSimilarity(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, 1)
    
    def forward(self, seq_repr, item_repr):
        batch_size = seq_repr.size(0)
        
        Q = self.q_linear(seq_repr).view(batch_size, -1, self.num_heads, self.head_dim)
        K = self.k_linear(item_repr).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # 计算每个头的相似度
        similarities = []
        for head in range(self.num_heads):
            q_head = Q[:, :, head, :]
            k_head = K[:, :, head, :]
            sim = (q_head * k_head).sum(dim=-1)
            similarities.append(sim)
        
        # 融合多头结果
        combined = torch.stack(similarities, dim=-1)
        output = self.output_linear(combined).squeeze(-1)
        
        return output
```

### 2. 对比学习的轻量版本
```python
def contrastive_loss_light(self, seq_repr, pos_repr, neg_repr, margin=1.0):
    """轻量级对比学习损失"""
    pos_sim = (seq_repr * pos_repr).sum(dim=-1)
    neg_sim = (seq_repr * neg_repr).sum(dim=-1)
    
    # 简单的margin loss
    loss = torch.clamp(margin - pos_sim + neg_sim, min=0.0)
    
    return loss.mean()
```

## 实施优先级

1. **高优先级**：恢复原始训练策略 + 优化相似度计算
2. **中优先级**：改进负采样策略 + 特征处理优化
3. **低优先级**：渐进式训练 + 模型集成优化

## 预期收益

1. **性能提升**：相比原始方法，预期AUC提升2-5%
2. **训练稳定性**：避免复杂策略带来的不稳定性
3. **计算效率**：优化后的特征处理和缓存机制提升训练速度20-30%
4. **易于调试**：保持模型结构简单，便于问题定位

## 风险控制

1. **渐进式修改**：每次只修改一个组件，观察效果
2. **A/B测试**：对比修改前后的效果
3. **回滚机制**：每个阶段都保存检查点，便于回滚
