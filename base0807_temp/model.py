from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import save_emb


class FlashMultiHeadAttention(torch.nn.Module):
    """
    Flash多头注意力机制实现
    
    使用PyTorch 2.0+的内置Flash Attention优化计算效率，
    如果不支持则降级到标准注意力机制
    
    Args:
        hidden_units: 隐藏层维度
        num_heads: 注意力头数
        dropout_rate: Dropout概率
    """
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(FlashMultiHeadAttention, self).__init__()

        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads  # 每个注意力头的维度
        self.dropout_rate = dropout_rate

        # 确保隐藏层维度能被注意力头数整除
        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"

        # 定义Q、K、V的线性变换层和输出线性层
        self.q_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.k_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.v_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.out_linear = torch.nn.Linear(hidden_units, hidden_units)

    def forward(self, query, key, value, attn_mask=None):
        """
        前向传播
        
        Args:
            query: 查询张量，形状为 [batch_size, seq_len, hidden_units]
            key: 键张量，形状为 [batch_size, seq_len, hidden_units]  
            value: 值张量，形状为 [batch_size, seq_len, hidden_units]
            attn_mask: 注意力掩码，形状为 [batch_size, seq_len]
            
        Returns:
            output: 注意力输出，形状为 [batch_size, seq_len, hidden_units]
            None: 兼容性返回值
        """
        batch_size, seq_len, _ = query.size()

        # 计算Q, K, V矩阵
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # 重塑为多头格式: [batch_size, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 尝试使用PyTorch 2.0+的Flash Attention
        if hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ 使用内置的Flash Attention，计算效率更高
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, dropout_p=self.dropout_rate if self.training else 0.0, attn_mask=attn_mask.unsqueeze(1)
            )
        else:
            # 降级到标准注意力机制实现
            scale = (self.head_dim) ** -0.5  # 缩放因子，防止softmax饱和
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

            # 应用注意力掩码（如因果掩码）
            if attn_mask is not None:
                scores.masked_fill_(attn_mask.unsqueeze(1).logical_not(), float('-inf'))

            # 计算注意力权重并应用dropout
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)
            attn_output = torch.matmul(attn_weights, V)

        # 重塑回原始格式: [batch_size, seq_len, hidden_units]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)

        # 最终的线性变换
        output = self.out_linear(attn_output)

        return output, None


class PointWiseFeedForward(torch.nn.Module):
    """
    逐点前馈网络（Position-wise Feed-Forward Networks）
    
    Transformer中的前馈子层，使用1D卷积实现
    结构: Linear -> ReLU -> Dropout -> Linear -> Dropout
    
    Args:
        hidden_units: 隐藏层维度
        dropout_rate: Dropout概率
    """
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        # 使用1D卷积实现逐点变换，kernel_size=1相当于全连接层
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        """
        前向传播
        
        Args:
            inputs: 输入张量，形状为 [batch_size, seq_len, hidden_units]
            
        Returns:
            outputs: 输出张量，形状为 [batch_size, seq_len, hidden_units]
        """
        # 转置以适应Conv1D的输入格式 (N, C, L)
        # 应用: Conv1D -> Dropout -> ReLU -> Dropout -> Conv1D
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        # 转置回原始格式
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        return outputs


class BaselineModel(torch.nn.Module):
    """
    基础推荐模型，基于Transformer架构
    
    主要功能：
    1. 处理用户序列数据（包含用户和物品交互）
    2. 融合多种特征类型：稀疏特征、数组特征、连续特征、多模态特征
    3. 使用Transformer编码用户行为序列
    4. 预测用户对物品的偏好
    
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
        self.norm_first = args.norm_first  # 是否在注意力前先进行Layer Normalization
        self.maxlen = args.maxlen
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch

        # ==================== Embedding层定义 ====================
        # 物品、用户、位置的Embedding表，padding_idx=0表示索引0用作padding
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(2 * args.maxlen + 1, args.hidden_units, padding_idx=0)  # 位置编码
        
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        
        # 稀疏特征的Embedding表（用字典管理不同特征ID的Embedding）
        self.sparse_emb = torch.nn.ModuleDict()
        # 多模态特征的线性变换层（将预训练的多模态特征映射到模型维度）
        self.emb_transform = torch.nn.ModuleDict()

        # ==================== Transformer层定义 ====================
        # 存储多个Transformer块的LayerNorm和Layer
        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        # 初始化特征信息
        self._init_feat_info(feat_statistics, feat_types)

        # ==================== 特征融合层定义 ====================
        # 计算用户和物品特征拼接后的维度（包含时间特征）
        userdim = args.hidden_units * (
            len(self.USER_SPARSE_FEAT) + 1 + len(self.USER_ARRAY_FEAT) + len(self.TIME_SPARSE_FEAT)
        ) + len(self.USER_CONTINUAL_FEAT) + len(self.TIME_CONTINUAL_FEAT)
        
        itemdim = (
            args.hidden_units * (
                len(self.ITEM_SPARSE_FEAT) + 1 + len(self.ITEM_ARRAY_FEAT) + len(self.TIME_SPARSE_FEAT)
            )
            + len(self.ITEM_CONTINUAL_FEAT) + len(self.TIME_CONTINUAL_FEAT)
            + args.hidden_units * len(self.ITEM_EMB_FEAT)
        )

        # 用户和物品特征的全连接层
        self.userdnn = torch.nn.Linear(userdim, args.hidden_units)
        self.itemdnn = torch.nn.Linear(itemdim, args.hidden_units)

        # 最终的LayerNorm层
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        # ==================== 构建Transformer块 ====================
        for _ in range(args.num_blocks):
            # 注意力子层的LayerNorm
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            # Flash多头注意力层
            new_attn_layer = FlashMultiHeadAttention(
                args.hidden_units, args.num_heads, args.dropout_rate
            )  # 优化：用FlashAttention替代标准Attention
            self.attention_layers.append(new_attn_layer)

            # 前馈子层的LayerNorm
            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            # 逐点前馈网络
            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        # ==================== 初始化特征Embedding ====================
        # 为不同类型的特征创建对应的Embedding表
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

    def feat2tensor(self, seq_feature, k):
        """
        将特征转换为张量格式
        
        Args:
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典，形状为 [batch_size, maxlen]
            k: 特征ID

        Returns:
            batch_data: 特征值的tensor，形状为 [batch_size, maxlen, max_array_len(if array)]
        """
        batch_size = len(seq_feature)

        if k in self.ITEM_ARRAY_FEAT or k in self.USER_ARRAY_FEAT:
            # 如果特征是Array类型，需要先对array进行padding，然后转换为tensor
            max_array_len = 0
            max_seq_len = 0

            # 找出批次中数组的最大长度，用于padding
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                max_seq_len = max(max_seq_len, len(seq_data))
                max_array_len = max(max_array_len, max(len(item_data) for item_data in seq_data))

            # 创建padding后的数组
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
        将特征转换为Embedding表示
        
        Args:
            seq: 序列ID
            feature_array: 特征list，每个元素为当前时刻的特征字典
            mask: 掩码，1表示item，2表示user
            include_user: 是否处理用户特征，在两种情况下不打开：1) 训练时在转换正负样本的特征时（因为正负样本都是item）;2) 生成候选库item embedding时。

        Returns:
            seqs_emb: 序列特征的Embedding
        """
        seq = seq.to(self.dev)
        
        # ==================== 基础ID Embedding ====================
        # 预计算用户和物品的基础embedding
        if include_user:
            # 分别处理用户和物品token
            user_mask = (mask == 2).to(self.dev)  # 用户token掩码
            item_mask = (mask == 1).to(self.dev)  # 物品token掩码
            user_embedding = self.user_emb(user_mask * seq)
            item_embedding = self.item_emb(item_mask * seq)
            item_feat_list = [item_embedding]
            user_feat_list = [user_embedding]
        else:
            # 只处理物品token
            item_embedding = self.item_emb(seq)
            item_feat_list = [item_embedding]

        # ==================== 批量处理各种特征类型 ====================
        # 定义需要处理的特征类型
        all_feat_types = [
            (self.ITEM_SPARSE_FEAT, 'item_sparse', item_feat_list),    # 物品稀疏特征
            (self.ITEM_ARRAY_FEAT, 'item_array', item_feat_list),      # 物品数组特征
            (self.ITEM_CONTINUAL_FEAT, 'item_continual', item_feat_list), # 物品连续特征
            (self.TIME_SPARSE_FEAT, 'time_sparse', item_feat_list),    # 时间稀疏特征（应用于物品token）
            (self.TIME_CONTINUAL_FEAT, 'time_continual', item_feat_list), # 时间连续特征（应用于物品token）
        ]

        # 如果需要处理用户特征，添加用户特征类型
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

        # 批量处理每种特征类型
        for feat_dict, feat_type, feat_list in all_feat_types:
            if not feat_dict:
                continue

            for k in feat_dict:
                tensor_feature = self.feat2tensor(feature_array, k)

                if feat_type.endswith('sparse'):
                    # 稀疏特征：直接通过Embedding表
                    feat_list.append(self.sparse_emb[k](tensor_feature))
                elif feat_type.endswith('array'):
                    # 数组特征：先通过Embedding表，再对数组维度求和
                    feat_list.append(self.sparse_emb[k](tensor_feature).sum(2))
                elif feat_type.endswith('continual'):
                    # 连续特征：直接使用原始值，增加一个维度
                    feat_list.append(tensor_feature.unsqueeze(2))

        # ==================== 处理多模态特征 ====================
        for k in self.ITEM_EMB_FEAT:
            # 批量收集所有多模态特征数据
            batch_size = len(feature_array)
            emb_dim = self.ITEM_EMB_FEAT[k]
            seq_len = len(feature_array[0])

            # 预分配张量空间
            batch_emb_data = np.zeros((batch_size, seq_len, emb_dim), dtype=np.float32)

            # 填充多模态特征数据
            for i, seq in enumerate(feature_array):
                for j, item in enumerate(seq):
                    if k in item:
                        batch_emb_data[i, j] = item[k]

            # 批量转换并传输到GPU，然后通过线性变换
            tensor_feature = torch.from_numpy(batch_emb_data).to(self.dev)
            item_feat_list.append(self.emb_transform[k](tensor_feature))

        # ==================== 特征融合 ====================
        # 将所有物品特征拼接并通过全连接层
        all_item_emb = torch.cat(item_feat_list, dim=2)
        all_item_emb = torch.relu(self.itemdnn(all_item_emb))
        
        if include_user:
            # 如果包含用户特征，也进行相同处理并相加融合
            all_user_emb = torch.cat(user_feat_list, dim=2)
            all_user_emb = torch.relu(self.userdnn(all_user_emb))
            seqs_emb = all_item_emb + all_user_emb
        else:
            seqs_emb = all_item_emb
            
        return seqs_emb

    def log2feats(self, log_seqs, mask, seq_feature):
        """
        将用户行为序列转换为特征表示
        
        Args:
            log_seqs: 序列ID
            mask: token类型掩码，1表示item token，2表示user token
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典

        Returns:
            seqs_emb: 序列的Embedding，形状为 [batch_size, maxlen, hidden_units]
        """
        batch_size = log_seqs.shape[0]
        maxlen = log_seqs.shape[1]
        
        # ==================== 特征Embedding ====================
        # 获取融合后的序列特征embedding
        seqs = self.feat2emb(log_seqs, seq_feature, mask=mask, include_user=True)
        
        # 缩放embedding（类似Transformer中的做法）
        seqs *= self.item_emb.embedding_dim**0.5
        
        # ==================== 位置编码 ====================
        # 生成位置索引，只对非padding位置（log_seqs != 0）添加位置编码
        poss = torch.arange(1, maxlen + 1, device=self.dev).unsqueeze(0).expand(batch_size, -1).clone()
        poss *= log_seqs != 0  # 将padding位置的位置编码设为0
        seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)

        # ==================== 注意力掩码 ====================
        maxlen = seqs.shape[1]
        # 创建因果掩码（下三角矩阵），确保只能看到过去的信息
        ones_matrix = torch.ones((maxlen, maxlen), dtype=torch.bool, device=self.dev)
        attention_mask_tril = torch.tril(ones_matrix)
        
        # 创建padding掩码，过滤掉padding位置
        attention_mask_pad = (mask != 0).to(self.dev)
        
        # 组合因果掩码和padding掩码
        attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)

        # ==================== Transformer编码 ====================
        # 通过多个Transformer块进行编码
        for i in range(len(self.attention_layers)):
            if self.norm_first:
                # Pre-LayerNorm结构：先LayerNorm再Attention/FFN
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x, attn_mask=attention_mask)
                seqs = seqs + mha_outputs  # 残差连接
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                # Post-LayerNorm结构：先Attention/FFN再LayerNorm
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)  # 残差连接
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        # 最终的LayerNorm
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
        # ==================== 序列编码 ====================
        # 获取用户行为序列的表示
        log_feats = self.log2feats(user_item, mask, seq_feature)
        
        # 创建损失掩码，只对item token计算损失
        loss_mask = (next_mask == 1).to(self.dev)

        # ==================== 候选物品编码 ====================
        # 获取正负样本的embedding（不包含用户特征，因为它们都是物品）
        pos_embs = self.feat2emb(pos_seqs, pos_feature, include_user=False)
        neg_embs = self.feat2emb(neg_seqs, neg_feature, include_user=False)

        # ==================== 计算相似度得分 ====================
        # 计算序列表示和候选物品的内积作为logits
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        
        # 应用损失掩码，只保留item位置的logits
        pos_logits = pos_logits * loss_mask
        neg_logits = neg_logits * loss_mask

        return pos_logits, neg_logits

    def predict(self, log_seqs, seq_feature, mask):
        """
        推理时调用，计算用户序列的表征
        
        Args:
            log_seqs: 用户序列ID
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典
            mask: token类型掩码，1表示item token，2表示user token
            
        Returns:
            final_feat: 用户序列的表征，形状为 [batch_size, hidden_units]
        """
        # 获取序列的完整表示
        log_feats = self.log2feats(log_seqs, mask, seq_feature)

        # 取最后一个位置的特征作为用户的最终表示
        final_feat = log_feats[:, -1, :]

        return final_feat

    def save_item_emb(self, item_ids, retrieval_ids, feat_dict, save_path, batch_size=1024):
        """
        生成候选库item embedding，用于检索
        
        这个方法为候选物品池中的所有物品生成embedding，
        用于后续的近似最近邻搜索（ANN）

        Args:
            item_ids: 候选item ID（re-id形式）
            retrieval_ids: 候选item ID（检索ID，从0开始编号，检索脚本使用）
            feat_dict: 训练集所有item特征字典，key为特征ID，value为特征值
            save_path: 保存路径
            batch_size: 批次大小
        """
        all_embs = []

        # 分批处理候选物品，避免内存溢出
        for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
            end_idx = min(start_idx + batch_size, len(item_ids))

            # 准备当前批次的数据
            item_seq = torch.tensor(item_ids[start_idx:end_idx], device=self.dev).unsqueeze(0)
            batch_feat = []
            for i in range(start_idx, end_idx):
                batch_feat.append(feat_dict[i])

            batch_feat = np.array(batch_feat, dtype=object)

            # 计算当前批次的item embedding
            batch_emb = self.feat2emb(item_seq, [batch_feat], include_user=False).squeeze(0)

            # 转换为numpy并添加到结果列表
            all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))

        # ==================== 保存结果 ====================
        # 合并所有批次的结果并保存为二进制文件
        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)
        
        # 保存embedding和对应的ID
        save_emb(final_embs, Path(save_path, 'embedding.fbin'))  # 物品embedding
        save_emb(final_ids, Path(save_path, 'id.u64bin'))       # 物品ID
