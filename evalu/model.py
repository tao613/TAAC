from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import save_emb


class SimilarityCalculator:
    """
    高级相似度计算器，提供多种相似度度量方法
    """
    
    @staticmethod
    def cosine_similarity(x, y, eps=1e-8):
        """
        余弦相似度，对向量长度不敏感，更适合捕捉方向性相似度
        
        Args:
            x, y: 输入张量 [batch_size, seq_len, hidden_dim]
            eps: 防止除零的小常数
            
        Returns:
            similarity: 余弦相似度 [batch_size, seq_len]
        """
        x_norm = x / (x.norm(dim=-1, keepdim=True) + eps)
        y_norm = y / (y.norm(dim=-1, keepdim=True) + eps)
        return (x_norm * y_norm).sum(dim=-1)
    
    @staticmethod
    def scaled_dot_product(x, y, scale_factor=None):
        """
        缩放点积相似度，添加温度参数控制
        
        Args:
            x, y: 输入张量 [batch_size, seq_len, hidden_dim]
            scale_factor: 缩放因子，默认为 1/sqrt(hidden_dim)
            
        Returns:
            similarity: 缩放点积相似度 [batch_size, seq_len]
        """
        if scale_factor is None:
            scale_factor = 1.0 / (x.size(-1) ** 0.5)
        return (x * y).sum(dim=-1) * scale_factor
    
    @staticmethod
    def bilinear_similarity(x, y, weight_matrix):
        """
        双线性相似度：x^T W y，学习更复杂的相似度函数
        
        Args:
            x, y: 输入张量 [batch_size, seq_len, hidden_dim]
            weight_matrix: 权重矩阵 [hidden_dim, hidden_dim]
            
        Returns:
            similarity: 双线性相似度 [batch_size, seq_len]
        """
        # x: [batch_size, seq_len, hidden_dim]
        # weight_matrix: [hidden_dim, hidden_dim]
        # y: [batch_size, seq_len, hidden_dim]
        
        batch_size, seq_len, hidden_dim = x.shape
        
        # 计算 x @ W
        x_transformed = torch.matmul(x.view(-1, hidden_dim), weight_matrix)  # [batch_size*seq_len, hidden_dim]
        x_transformed = x_transformed.view(batch_size, seq_len, hidden_dim)  # [batch_size, seq_len, hidden_dim]
        
        # 计算 (x @ W) * y
        return (x_transformed * y).sum(dim=-1)
    
    @staticmethod
    def euclidean_distance(x, y, eps=1e-8):
        """
        欧几里得距离的负值作为相似度（距离越小，相似度越高）
        
        Args:
            x, y: 输入张量 [batch_size, seq_len, hidden_dim]
            eps: 防止数值不稳定的小常数
            
        Returns:
            similarity: 负欧几里得距离 [batch_size, seq_len]
        """
        distance = torch.norm(x - y, dim=-1, p=2)
        # 转换为相似度（距离越小，相似度越高）
        return -distance / (distance.max() + eps)


class AdaptiveSimilarity(torch.nn.Module):
    """
    自适应相似度融合模块，学习不同相似度度量的最优组合
    """
    
    def __init__(self, hidden_dim, similarity_types=['dot', 'cosine', 'scaled']):
        super(AdaptiveSimilarity, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.similarity_types = similarity_types
        
        # 可学习的融合权重
        self.alpha = torch.nn.Parameter(torch.ones(1) * 0.5)  # 原始点积权重
        self.beta = torch.nn.Parameter(torch.ones(1) * 0.3)   # 余弦相似度权重
        self.gamma = torch.nn.Parameter(torch.ones(1) * 0.2)  # 缩放点积权重
        
        # 双线性相似度的权重矩阵
        if 'bilinear' in similarity_types:
            self.bilinear_weight = torch.nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
            self.delta = torch.nn.Parameter(torch.ones(1) * 0.1)  # 双线性相似度权重
        
        # 温度参数
        self.temperature = torch.nn.Parameter(torch.ones(1) * 1.0)
        
    def forward(self, seq_repr, item_repr):
        """
        计算自适应相似度
        
        Args:
            seq_repr: 序列表示 [batch_size, seq_len, hidden_dim]
            item_repr: 物品表示 [batch_size, seq_len, hidden_dim]
            
        Returns:
            similarity: 融合后的相似度 [batch_size, seq_len]
        """
        similarities = []
        
        # 基础点积相似度
        if 'dot' in self.similarity_types:
            dot_sim = (seq_repr * item_repr).sum(dim=-1)
            similarities.append(self.alpha * dot_sim)
        
        # 余弦相似度
        if 'cosine' in self.similarity_types:
            cos_sim = SimilarityCalculator.cosine_similarity(seq_repr, item_repr)
            similarities.append(self.beta * cos_sim)
        
        # 缩放点积相似度
        if 'scaled' in self.similarity_types:
            scaled_sim = SimilarityCalculator.scaled_dot_product(seq_repr, item_repr)
            similarities.append(self.gamma * scaled_sim)
        
        # 双线性相似度
        if 'bilinear' in self.similarity_types and hasattr(self, 'bilinear_weight'):
            bilinear_sim = SimilarityCalculator.bilinear_similarity(seq_repr, item_repr, self.bilinear_weight)
            similarities.append(self.delta * bilinear_sim)
        
        # 欧几里得距离
        if 'euclidean' in self.similarity_types:
            euclidean_sim = SimilarityCalculator.euclidean_distance(seq_repr, item_repr)
            similarities.append(0.1 * euclidean_sim)  # 通常权重较小
        
        # 融合所有相似度
        if len(similarities) == 1:
            final_similarity = similarities[0]
        else:
            final_similarity = torch.stack(similarities, dim=-1).sum(dim=-1)
        
        # 应用温度缩放
        final_similarity = final_similarity * self.temperature
        
        return final_similarity


class FlashMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(FlashMultiHeadAttention, self).__init__()

        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate

        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"

        self.q_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.k_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.v_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.out_linear = torch.nn.Linear(hidden_units, hidden_units)

    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, _ = query.size()

        # 计算Q, K, V
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # reshape为multi-head格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ 使用内置的Flash Attention
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, dropout_p=self.dropout_rate if self.training else 0.0, attn_mask=attn_mask.unsqueeze(1)
            )
        else:
            # 降级到标准注意力机制
            scale = (self.head_dim) ** -0.5
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

            if attn_mask is not None:
                scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)
            attn_output = torch.matmul(attn_weights, V)

        # reshape回原来的格式
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)

        # 最终的线性变换
        output = self.out_linear(attn_output)

        return output, None


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        return outputs


class BaselineModel(torch.nn.Module):
    """
    Args:
        user_num: 用户数量
        item_num: 物品数量
        feat_statistics: 特征统计信息，key为特征ID，value为特征数量
        feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表，包括user和item的sparse, array, emb, continual类型
        args: 全局参数

    Attributes:
        user_num: 用户数量
        item_num: 物品数量
        dev: 设备
        norm_first: 是否先归一化
        maxlen: 序列最大长度
        item_emb: Item Embedding Table
        user_emb: User Embedding Table
        sparse_emb: 稀疏特征Embedding Table
        emb_transform: 多模态特征的线性变换
        userdnn: 用户特征拼接后经过的全连接层
        itemdnn: 物品特征拼接后经过的全连接层
    """

    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):  #
        super(BaselineModel, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.maxlen = args.maxlen
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch

        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(2 * args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.sparse_emb = torch.nn.ModuleDict()
        self.emb_transform = torch.nn.ModuleDict()

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self._init_feat_info(feat_statistics, feat_types)

        userdim = args.hidden_units * (
            len(self.USER_SPARSE_FEAT) + 1 + len(self.USER_ARRAY_FEAT) + len(self.TIME_SPARSE_FEAT)
        ) + len(self.USER_CONTINUAL_FEAT) + len(self.TIME_CONTINUAL_FEAT)
        
        itemdim = (
            args.hidden_units * (
                len(self.ITEM_SPARSE_FEAT) + 1 + len(self.ITEM_ARRAY_FEAT) + len(self.TIME_SPARSE_FEAT) + len(self.SEMANTIC_ARRAY_FEAT)
            )
            + len(self.ITEM_CONTINUAL_FEAT) + len(self.TIME_CONTINUAL_FEAT)
            + args.hidden_units * len(self.ITEM_EMB_FEAT)
        )

        self.userdnn = torch.nn.Linear(userdim, args.hidden_units)
        self.itemdnn = torch.nn.Linear(itemdim, args.hidden_units)

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = FlashMultiHeadAttention(
                args.hidden_units, args.num_heads, args.dropout_rate
            )  # 优化：用FlashAttention替代标准Attention
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        for k in self.USER_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.USER_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_EMB_FEAT:
            self.emb_transform[k] = torch.nn.Linear(self.ITEM_EMB_FEAT[k], args.hidden_units)
            
        # ==================== 初始化时间特征Embedding ====================
        # 时间特征的embedding表
        for k in self.TIME_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.TIME_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
            
        # ==================== 初始化semantic_id特征Embedding ====================
        # semantic_id作为数组特征，需要embedding表
        for k in self.SEMANTIC_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.SEMANTIC_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)

    def _init_feat_info(self, feat_statistics, feat_types):
        """
        将特征统计信息（特征数量）按特征类型分组产生不同的字典，方便声明稀疏特征的Embedding Table

        Args:
            feat_statistics: 特征统计信息，key为特征ID，value为特征数量
            feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表，包括user和item的sparse, array, emb, continual类型
        """
        # 按特征类型分组特征统计信息
        self.USER_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['user_sparse']}     # 用户稀疏特征
        self.USER_CONTINUAL_FEAT = feat_types['user_continual']                               # 用户连续特征
        self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['item_sparse']}   # 物品稀疏特征
        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']                               # 物品连续特征
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['user_array']}     # 用户数组特征
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['item_array']}     # 物品数组特征
        
        # ==================== 时间特征处理 ====================
        # 时间特征同时适用于用户和物品token
        self.TIME_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types.get('time_sparse', [])}     # 时间稀疏特征
        self.TIME_CONTINUAL_FEAT = feat_types.get('time_continual', [])                               # 时间连续特征
        
        # 多模态特征的维度映射（81-86对应不同的多模态特征类型）
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in feat_types['item_emb']}  # 记录的是不同多模态特征的维度
        
        # ==================== 添加semantic_id特征支持 ====================
        # semantic_id作为特殊的数组特征处理
        self.SEMANTIC_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types.get('semantic_array', [])}

    def feat2tensor(self, seq_feature, k):
        """
        Args:
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典，形状为 [batch_size, maxlen]
            k: 特征ID

        Returns:
            batch_data: 特征值的tensor，形状为 [batch_size, maxlen, max_array_len(if array)]
        """
        batch_size = len(seq_feature)

        if k in self.ITEM_ARRAY_FEAT or k in self.USER_ARRAY_FEAT or k in self.SEMANTIC_ARRAY_FEAT:
            # 如果特征是Array类型，需要先对array进行padding，然后转换为tensor
            max_array_len = 0
            max_seq_len = 0

            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                max_seq_len = max(max_seq_len, len(seq_data))
                max_array_len = max(max_array_len, max(len(item_data) for item_data in seq_data))

            batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                for j, item_data in enumerate(seq_data):
                    actual_len = min(len(item_data), max_array_len)
                    batch_data[i, j, :actual_len] = item_data[:actual_len]

            return torch.from_numpy(batch_data).to(self.dev)
        else:
            # 如果特征是Sparse类型，直接转换为tensor
            max_seq_len = max(len(seq_feature[i]) for i in range(batch_size))
            batch_data = np.zeros((batch_size, max_seq_len), dtype=np.int64)

            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                batch_data[i] = seq_data

            return torch.from_numpy(batch_data).to(self.dev)

    def feat2emb(self, seq, feature_array, mask=None, include_user=False):
        """
        Args:
            seq: 序列ID
            feature_array: 特征list，每个元素为当前时刻的特征字典
            mask: 掩码，1表示item，2表示user
            include_user: 是否处理用户特征，在两种情况下不打开：1) 训练时在转换正负样本的特征时（因为正负样本都是item）;2) 生成候选库item embedding时。

        Returns:
            seqs_emb: 序列特征的Embedding
        """
        seq = seq.to(self.dev)
        # pre-compute embedding
        if include_user:
            user_mask = (mask == 2).to(self.dev)
            item_mask = (mask == 1).to(self.dev)
            user_embedding = self.user_emb(user_mask * seq)
            item_embedding = self.item_emb(item_mask * seq)
            item_feat_list = [item_embedding]
            user_feat_list = [user_embedding]
        else:
            item_embedding = self.item_emb(seq)
            item_feat_list = [item_embedding]

        # batch-process all feature types
        all_feat_types = [
            (self.ITEM_SPARSE_FEAT, 'item_sparse', item_feat_list),    # 物品稀疏特征
            (self.ITEM_ARRAY_FEAT, 'item_array', item_feat_list),      # 物品数组特征
            (self.ITEM_CONTINUAL_FEAT, 'item_continual', item_feat_list), # 物品连续特征
            (self.TIME_SPARSE_FEAT, 'time_sparse', item_feat_list),    # 时间稀疏特征（应用于物品token）
            (self.TIME_CONTINUAL_FEAT, 'time_continual', item_feat_list), # 时间连续特征（应用于物品token）
            (self.SEMANTIC_ARRAY_FEAT, 'semantic_array', item_feat_list),  # 添加semantic_id特征
        ]

        if include_user:
            all_feat_types.extend(
                [
                    (self.USER_SPARSE_FEAT, 'user_sparse', user_feat_list),      # 用户稀疏特征
                    (self.USER_ARRAY_FEAT, 'user_array', user_feat_list),        # 用户数组特征
                    (self.USER_CONTINUAL_FEAT, 'user_continual', user_feat_list), # 用户连续特征
                    (self.TIME_SPARSE_FEAT, 'time_sparse', user_feat_list),      # 时间稀疏特征（应用于用户token）
                    (self.TIME_CONTINUAL_FEAT, 'time_continual', user_feat_list), # 时间连续特征（应用于用户token）
                ]
            )

        # batch-process each feature type
        for feat_dict, feat_type, feat_list in all_feat_types:
            if not feat_dict:
                continue

            for k in feat_dict:
                tensor_feature = self.feat2tensor(feature_array, k)

                if feat_type.endswith('sparse'):
                    feat_list.append(self.sparse_emb[k](tensor_feature))
                elif feat_type.endswith('array'):
                    feat_list.append(self.sparse_emb[k](tensor_feature).sum(2))
                elif feat_type == 'semantic_array':
                    # semantic_id特征的特殊处理：对序列求和
                    feat_list.append(self.sparse_emb[k](tensor_feature).sum(2))
                elif feat_type.endswith('continual'):
                    feat_list.append(tensor_feature.unsqueeze(2))

        for k in self.ITEM_EMB_FEAT:
            # collect all data to numpy, then batch-convert
            batch_size = len(feature_array)
            emb_dim = self.ITEM_EMB_FEAT[k]
            seq_len = len(feature_array[0])

            # pre-allocate tensor
            batch_emb_data = np.zeros((batch_size, seq_len, emb_dim), dtype=np.float32)

            for i, seq in enumerate(feature_array):
                for j, item in enumerate(seq):
                    if k in item:
                        batch_emb_data[i, j] = item[k]

            # batch-convert and transfer to GPU
            tensor_feature = torch.from_numpy(batch_emb_data).to(self.dev)
            item_feat_list.append(self.emb_transform[k](tensor_feature))

        # merge features
        all_item_emb = torch.cat(item_feat_list, dim=2)
        all_item_emb = torch.relu(self.itemdnn(all_item_emb))
        if include_user:
            all_user_emb = torch.cat(user_feat_list, dim=2)
            all_user_emb = torch.relu(self.userdnn(all_user_emb))
            seqs_emb = all_item_emb + all_user_emb
        else:
            seqs_emb = all_item_emb
        return seqs_emb

    def log2feats(self, log_seqs, mask, seq_feature):
        """
        Args:
            log_seqs: 序列ID
            mask: token类型掩码，1表示item token，2表示user token
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典

        Returns:
            seqs_emb: 序列的Embedding，形状为 [batch_size, maxlen, hidden_units]
        """
        batch_size = log_seqs.shape[0]
        maxlen = log_seqs.shape[1]
        seqs = self.feat2emb(log_seqs, seq_feature, mask=mask, include_user=True)
        seqs *= self.item_emb.embedding_dim**0.5
        poss = torch.arange(1, maxlen + 1, device=self.dev).unsqueeze(0).expand(batch_size, -1).clone()
        poss *= log_seqs != 0
        seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)

        maxlen = seqs.shape[1]
        ones_matrix = torch.ones((maxlen, maxlen), dtype=torch.bool, device=self.dev)
        attention_mask_tril = torch.tril(ones_matrix)
        attention_mask_pad = (mask != 0).to(self.dev)
        attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)

        for i in range(len(self.attention_layers)):
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x, attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(
        self, user_item, pos_seqs, neg_seqs, mask, next_mask, next_action_type, seq_feature, pos_feature, neg_feature
    ):
        """
        训练时调用，计算正负样本的logits

        Args:
            user_item: 用户序列ID
            pos_seqs: 正样本序列ID
            neg_seqs: 负样本序列ID
            mask: token类型掩码，1表示item token，2表示user token
            next_mask: 下一个token类型掩码，1表示item token，2表示user token
            next_action_type: 下一个token动作类型，0表示曝光，1表示点击
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典
            pos_feature: 正样本特征list，每个元素为当前时刻的特征字典
            neg_feature: 负样本特征list，每个元素为当前时刻的特征字典

        Returns:
            pos_logits: 正样本logits，形状为 [batch_size, maxlen]
            neg_logits: 负样本logits，形状为 [batch_size, maxlen]
        """
        log_feats = self.log2feats(user_item, mask, seq_feature)
        loss_mask = (next_mask == 1).to(self.dev)

        pos_embs = self.feat2emb(pos_seqs, pos_feature, include_user=False)
        neg_embs = self.feat2emb(neg_seqs, neg_feature, include_user=False)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        pos_logits = pos_logits * loss_mask
        neg_logits = neg_logits * loss_mask

        return pos_logits, neg_logits

    def predict(self, log_seqs, seq_feature, mask):
        """
        计算用户序列的表征
        Args:
            log_seqs: 用户序列ID
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典
            mask: token类型掩码，1表示item token，2表示user token
        Returns:
            final_feat: 用户序列的表征，形状为 [batch_size, hidden_units]
        """
        log_feats = self.log2feats(log_seqs, mask, seq_feature)

        final_feat = log_feats[:, -1, :]

        return final_feat

    def save_item_emb(self, item_ids, retrieval_ids, feat_dict, save_path, batch_size=1024):
        """
        生成候选库item embedding，用于检索

        Args:
            item_ids: 候选item ID（re-id形式）
            retrieval_ids: 候选item ID（检索ID，从0开始编号，检索脚本使用）
            feat_dict: 训练集所有item特征字典，key为特征ID，value为特征值
            save_path: 保存路径
            batch_size: 批次大小
        """
        all_embs = []

        for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
            end_idx = min(start_idx + batch_size, len(item_ids))

            item_seq = torch.tensor(item_ids[start_idx:end_idx], device=self.dev).unsqueeze(0)
            batch_feat = []
            for i in range(start_idx, end_idx):
                batch_feat.append(feat_dict[i])

            batch_feat = np.array(batch_feat, dtype=object)

            batch_emb = self.feat2emb(item_seq, [batch_feat], include_user=False).squeeze(0)

            all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))

        # 合并所有批次的结果并保存
        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)
        save_emb(final_embs, Path(save_path, 'embedding.fbin'))
        save_emb(final_ids, Path(save_path, 'id.u64bin'))
