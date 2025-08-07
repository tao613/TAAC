# 广告算法比赛代码流程分析

## 项目概述

这是一个基于深度学习的推荐系统项目，采用Transformer架构处理用户行为序列，用于广告推荐算法比赛。项目包含完整的模型训练和评估流程，支持多种特征类型融合和多模态特征处理。

## 整体架构

```
taac_git/
├── main.py                    # 主训练脚本
├── model.py                   # 基础推荐模型
├── model_rqvae.py            # RQ-VAE向量量化模型
├── dataset.py                 # 数据加载和预处理
├── evalu/                     # 评估模块
│   ├── infer.py              # 推理脚本
│   ├── model.py              # 评估用模型
│   ├── model_rqvae.py        # 评估用RQ-VAE模型
│   └── dataset.py            # 评估用数据集
└── run.sh                    # 运行脚本
```

## 训练流程 (main.py)

### 1. 环境初始化
```python
# 创建日志和模型保存目录
Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)

# 初始化日志文件和TensorBoard记录器
log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
```

### 2. 数据加载和预处理
```python
# 创建数据集，包含用户序列、物品特征、多模态特征等
dataset = MyDataset(data_path, args)

# 按9:1比例划分训练集和验证集
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])

# 创建数据加载器，使用自定义的collate_fn处理变长序列
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                         collate_fn=dataset.collate_fn)
```

### 3. 模型初始化
```python
# 创建基础推荐模型，包含用户/物品Embedding、多模态特征处理、Transformer等
model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args)

# 使用Xavier正态分布初始化模型参数
for name, param in model.named_parameters():
    try:
        torch.nn.init.xavier_normal_(param.data)
    except Exception:
        pass

# 将padding位置的embedding初始化为0
model.pos_emb.weight.data[0, :] = 0
model.item_emb.weight.data[0, :] = 0
model.user_emb.weight.data[0, :] = 0
```

### 4. 训练循环
```python
for epoch in range(epoch_start_idx, args.num_epochs + 1):
    model.train()
    for step, batch in enumerate(train_loader):
        # 解包批次数据
        seq, pos, neg, token_type, next_token_type, next_action_type, \
        seq_feat, pos_feat, neg_feat = batch
        
        # 模型前向传播，计算正负样本的logits
        pos_logits, neg_logits = model(seq, pos, neg, token_type, 
                                      next_token_type, next_action_type, 
                                      seq_feat, pos_feat, neg_feat)
        
        # 计算损失（只对item token计算损失）
        indices = np.where(next_token_type == 1)
        loss = bce_criterion(pos_logits[indices], pos_labels[indices])
        loss += bce_criterion(neg_logits[indices], neg_labels[indices])
        
        # 反向传播和参数更新
        loss.backward()
        optimizer.step()
```

### 5. 验证和模型保存
```python
# 验证阶段
model.eval()
with torch.no_grad():
    for batch in valid_loader:
        # 计算验证损失
        ...

# 保存模型权重
save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), 
               f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
torch.save(model.state_dict(), save_dir / "model.pt")
```

## 模型架构 (model.py)

### 1. BaselineModel 核心组件

#### Flash多头注意力机制
```python
class FlashMultiHeadAttention(torch.nn.Module):
    """
    使用PyTorch 2.0+的内置Flash Attention优化计算效率
    支持因果掩码和padding掩码
    """
    def forward(self, query, key, value, attn_mask=None):
        # 使用Flash Attention或降级到标准注意力
        if hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
        else:
            # 标准注意力机制实现
            ...
```

#### 特征融合架构
```python
def feat2emb(self, seq, feature_array, mask=None, include_user=False):
    """
    融合多种特征类型：
    - 用户/物品稀疏特征：通过Embedding表映射
    - 数组特征：先Embedding再求和
    - 连续特征：直接使用原始值
    - 多模态特征：通过线性变换映射到模型维度
    """
    # 处理不同类型特征
    for feat_type in ['sparse', 'array', 'continual', 'emb']:
        # 特征处理逻辑
        ...
    
    # 特征拼接和全连接变换
    all_item_emb = torch.cat(item_feat_list, dim=2)
    all_item_emb = torch.relu(self.itemdnn(all_item_emb))
```

### 2. 序列编码流程
```python
def log2feats(self, log_seqs, mask, seq_feature):
    """
    用户行为序列编码流程：
    1. 特征Embedding和融合
    2. 位置编码
    3. 因果掩码+padding掩码
    4. 多层Transformer编码
    """
    # 特征融合
    seqs = self.feat2emb(log_seqs, seq_feature, mask=mask, include_user=True)
    
    # 位置编码
    seqs += self.pos_emb(poss)
    
    # Transformer编码
    for i in range(len(self.attention_layers)):
        mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask)
        seqs = self.attention_layernorms[i](seqs + mha_outputs)
```

## 数据处理流程 (dataset.py)

### 1. 训练数据集 (MyDataset)

#### 数据加载优化
```python
def _load_data_and_offsets(self):
    """
    使用偏移量索引实现高效随机访问：
    - 避免将所有数据加载到内存
    - 支持大规模数据集训练
    """
    self.data_file = open(self.data_dir / "seq.jsonl", 'rb')
    with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
        self.seq_offsets = pickle.load(f)
```

#### 序列构建和padding
```python
def __getitem__(self, uid):
    """
    数据预处理流程：
    1. 构建用户-物品交替序列
    2. 生成正负样本对
    3. Left-padding到固定长度
    4. 特征缺失值填充
    """
    # 构建扩展用户序列（用户token和物品token交替）
    for record_tuple in user_sequence:
        u, i, user_feat, item_feat, action_type, _ = record_tuple
        if u and user_feat:
            ext_user_sequence.insert(0, (u, user_feat, 2, action_type))  # 用户token
        if i and item_feat:
            ext_user_sequence.append((i, item_feat, 1, action_type))     # 物品token
    
    # Left-padding和正负样本生成
    for record_tuple in reversed(ext_user_sequence[:-1]):
        # 为物品token生成正负样本
        if next_type == 1 and next_i != 0:
            pos[idx] = next_i  # 正样本：下一个真实交互的物品
            neg_id = self._random_neq(1, self.itemnum + 1, ts)  # 负采样
            neg[idx] = neg_id
```

### 2. 特征类型和处理

#### 六种特征类型
```python
# 用户特征
feat_types['user_sparse'] = ['103', '104', '105', '109']      # 稀疏特征
feat_types['user_array'] = ['106', '107', '108', '110']       # 数组特征
feat_types['user_continual'] = []                             # 连续特征

# 物品特征  
feat_types['item_sparse'] = ['100', '117', '111', ...]       # 稀疏特征
feat_types['item_array'] = []                                # 数组特征
feat_types['item_continual'] = []                            # 连续特征
feat_types['item_emb'] = self.mm_emb_ids                     # 多模态特征
```

#### 特征默认值填充
```python
def fill_missing_feat(self, feat, item_id):
    """
    特征缺失值处理：
    1. 一般特征：用预定义默认值填充
    2. 多模态特征：从预训练embedding中获取
    """
    # 填充一般特征的默认值
    missing_fields = set(all_feat_ids) - set(feat.keys())
    for feat_id in missing_fields:
        filled_feat[feat_id] = self.feature_default_value[feat_id]
    
    # 处理多模态特征
    for feat_id in self.feature_types['item_emb']:
        if item_id != 0 and self.indexer_i_rev[item_id] in self.mm_emb_dict[feat_id]:
            filled_feat[feat_id] = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]
```

## 评估流程 (evalu/infer.py)

### 1. 模型加载和初始化
```python
def infer():
    """
    完整推理流程：
    1. 加载训练好的模型
    2. 生成用户query embedding
    3. 生成候选物品embedding  
    4. 执行ANN检索
    5. 返回top-k推荐结果
    """
    # 加载训练好的模型
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args)
    ckpt_path = get_ckpt_path()
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
```

### 2. 用户Query Embedding生成
```python
# 为每个用户生成query embedding
for step, batch in enumerate(test_loader):
    seq, token_type, seq_feat, user_id = batch
    seq = seq.to(args.device)
    
    # 使用模型的predict方法生成用户表示
    logits = model.predict(seq, seq_feat, token_type)
    
    # 收集所有用户的embedding
    for i in range(logits.shape[0]):
        emb = logits[i].unsqueeze(0).detach().cpu().numpy().astype(np.float32)
        all_embs.append(emb)
```

### 3. 候选物品处理
```python
def get_candidate_emb(indexer, feat_types, feat_default_value, mm_emb_dict, model):
    """
    处理候选物品池：
    1. 读取候选物品特征
    2. 处理冷启动特征
    3. 生成物品embedding
    4. 保存为二进制文件用于ANN检索
    """
    # 读取候选物品集合
    with open(candidate_path, 'r') as f:
        for line in f:
            # 处理物品特征和冷启动问题
            feature = process_cold_start_feat(line['features'])
            # 填充多模态特征
            for feat_id in feat_types['item_emb']:
                if creative_id in mm_emb_dict[feat_id]:
                    feature[feat_id] = mm_emb_dict[feat_id][creative_id]
    
    # 生成候选库embedding
    model.save_item_emb(item_ids, retrieval_ids, features, save_path)
```

### 4. ANN检索和结果处理
```python
# 执行ANN检索
ann_cmd = (
    f"{faiss_demo_path} "
    f"--dataset_vector_file_path={embedding_file} "
    f"--query_vector_file_path={query_file} "
    f"--result_id_file_path={result_file} "
    f"--query_ann_top_k=10 --faiss_M=64 --faiss_ef_construction=1280"
)
os.system(ann_cmd)

# 处理检索结果
top10s_retrieved = read_result_ids(result_file)
# 将检索ID转换为creative_id
top10s = convert_retrieval_ids_to_creative_ids(top10s_retrieved, retrieve_id2creative_id)
```

## RQ-VAE向量量化模型 (model_rqvae.py)

### 1. 模型用途
RQ-VAE用于将高维多模态特征（如图像、文本embedding）转换为低维离散的语义ID：

```python
"""
RQ-VAE应用流程：
1. 使用MmEmbDataset读取多模态embedding数据
2. 训练RQ-VAE模型进行向量量化
3. 将连续向量转换为离散语义ID
4. 将语义ID作为稀疏特征加入推荐模型
"""
```

### 2. 核心组件

#### 残差量化器
```python
class RQ(torch.nn.Module):
    """
    残差量化的核心思想：
    1. 第一个codebook量化原始向量
    2. 第二个codebook量化第一次的残差  
    3. 逐步量化，获得更精确的重建
    """
    def quantize(self, data):
        res_emb = data.detach().clone()  # 初始残差
        for i in range(self.num_codebooks):
            vq_emb, semantic_id = self.vqmodules[i](res_emb)
            res_emb -= vq_emb      # 更新残差
            vq_emb_aggre += vq_emb # 累积量化结果
```

#### 向量量化Embedding
```python
class VQEmbedding(torch.nn.Embedding):
    """
    向量量化过程：
    1. 使用K-means初始化codebook
    2. 计算输入向量到codebook的距离
    3. 选择最近的codebook向量
    4. 返回对应的语义ID
    """
    def forward(self, data):
        self._create_codebook(data)           # 创建codebook
        semantic_id = self._create_semantic_id(data)  # 生成语义ID
        update_emb = self._update_emb(semantic_id)    # 获取量化向量
        return update_emb, semantic_id
```

## 关键设计特性

### 1. 高效的数据加载
- 使用文件偏移量索引实现随机访问
- 避免将大规模数据全部加载到内存
- 支持多进程数据加载

### 2. 多特征融合架构
- 支持六种不同类型的特征
- 统一的特征处理和融合框架
- 灵活的特征默认值处理

### 3. Flash Attention优化
- 使用PyTorch 2.0+的内置Flash Attention
- 支持因果掩码和padding掩码
- 降级兼容标准注意力机制

### 4. 冷启动处理
- 评估时处理训练时未见过的特征值
- 将字符串特征值转换为默认数值
- 保证模型推理的稳定性

### 5. ANN检索优化
- 使用FAISS进行高效的近似最近邻搜索
- 支持大规模候选物品库检索
- 可配置的检索参数

## 使用方法

### 1. 训练模型
```bash
python main.py --batch_size 128 --lr 0.001 --num_epochs 3 \
                --hidden_units 32 --num_blocks 1 --num_heads 1 \
                --mm_emb_id 81 --device cuda
```

### 2. 模型评估
```bash
cd evalu/
python infer.py --batch_size 128 --hidden_units 32 --num_blocks 1 \
                --num_heads 1 --mm_emb_id 81 --device cuda
```

### 3. 环境变量配置
```bash
export TRAIN_DATA_PATH="path/to/train/data"
export TRAIN_LOG_PATH="path/to/logs"
export TRAIN_TF_EVENTS_PATH="path/to/tensorboard"
export TRAIN_CKPT_PATH="path/to/checkpoints"
export EVAL_DATA_PATH="path/to/eval/data"
export EVAL_RESULT_PATH="path/to/results"
export MODEL_OUTPUT_PATH="path/to/model"
```

## 总结

该项目实现了一个完整的端到端推荐系统，具有以下特点：

1. **高效的序列建模**：使用Transformer架构处理用户行为序列
2. **多特征融合**：支持稀疏、连续、数组、多模态等多种特征类型
3. **向量量化技术**：使用RQ-VAE将连续特征转换为离散语义ID
4. **大规模检索**：使用ANN技术支持大规模候选物品检索
5. **工程优化**：包含数据加载、内存管理、计算效率等多方面优化

整个系统设计兼顾了模型效果和工程实现，适用于大规模广告推荐场景。 