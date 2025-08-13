import json
import pickle
import struct
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm


class MyDataset(torch.utils.data.Dataset):
    """
    用户序列数据集（训练用）
    
    这个数据集类负责加载和预处理用户行为序列数据，包括：
    1. 用户交互序列（用户ID、物品ID交替出现）
    2. 多种类型的特征（稀疏、数组、连续、多模态特征）
    3. 正负样本生成用于训练

    Args:
        data_dir: 数据文件目录
        args: 全局参数

    Attributes:
        data_dir: 数据文件目录
        maxlen: 最大长度
        item_feat_dict: 物品特征字典
        mm_emb_ids: 激活的mm_emb特征ID
        mm_emb_dict: 多模态特征字典
        itemnum: 物品数量
        usernum: 用户数量
        indexer_i_rev: 物品索引字典 (reid -> item_id)
        indexer_u_rev: 用户索引字典 (reid -> user_id)
        indexer: 索引字典
        feature_default_value: 特征缺省值
        feature_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feat_statistics: 特征统计信息，包括user和item的特征数量
    """

    def __init__(self, data_dir, args):
        """
        初始化数据集
        
        加载索引文件、特征文件、多模态特征等必要数据
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        
        # 加载用户序列数据和偏移量索引
        self._load_data_and_offsets()
        
        self.maxlen = args.maxlen  # 序列最大长度
        self.mm_emb_ids = args.mm_emb_id  # 激活的多模态特征ID列表

        # ==================== 加载特征和索引数据 ====================
        # 加载物品特征字典
        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
        
        # 加载指定的多模态特征
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)
        
        # 加载用户和物品的索引映射
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])  # 物品总数
            self.usernum = len(indexer['u'])  # 用户总数
            
        # 创建反向映射（重新编号ID -> 原始ID）
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer

        # 初始化特征信息（特征类型、默认值、统计信息）
        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()

    def _load_data_and_offsets(self):
        """
        加载用户序列数据和每一行的文件偏移量(预处理好的), 用于快速随机访问数据并I/O
        
        使用偏移量索引可以避免将所有数据加载到内存中，支持随机访问
        """
        # 打开序列数据文件（二进制模式）
        self.data_file = open(self.data_dir / "seq.jsonl", 'rb')
        
        # 加载偏移量索引，用于快速定位每个用户的数据
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _load_user_data(self, uid):
        """
        从数据文件中加载单个用户的数据
        
        使用偏移量直接跳转到指定用户的数据位置，避免顺序读取

        Args:
            uid: 用户ID(reid)

        Returns:
            data: 用户序列数据，格式为[(user_id, item_id, user_feat, item_feat, action_type, timestamp)]
        """
        # 跳转到用户数据的文件位置
        self.data_file.seek(self.seq_offsets[uid])
        
        # 读取该行数据并解析JSON
        line = self.data_file.readline()
        data = json.loads(line)
        return data

    def _random_neq(self, l, r, s):
        """
        生成一个不在序列s中的随机整数, 用于训练时的负采样
        
        确保负样本不会与用户历史交互的物品重复

        Args:
            l: 随机整数的最小值
            r: 随机整数的最大值
            s: 序列（用户历史交互物品集合）

        Returns:
            t: 不在序列s中的随机整数
        """
        t = np.random.randint(l, r)
        
        # 循环采样直到找到不在历史序列中且存在特征的物品
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式
        
        这是数据集类的核心方法，负责：
        1. 加载用户原始序列数据
        2. 构建用户-物品交替序列
        3. 生成正负样本
        4. 进行序列padding
        5. 处理各种特征

        Args:
            uid: 用户ID(reid)

        Returns:
            seq: 用户序列ID
            pos: 正样本ID（即下一个真实访问的item）
            neg: 负样本ID
            token_type: 用户序列类型，1表示item，2表示user
            next_token_type: 下一个token类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            pos_feat: 正样本特征，每个元素为字典，key为特征ID，value为特征值
            neg_feat: 负样本特征，每个元素为字典，key为特征ID，value为特征值
        """
        # ==================== 加载用户序列数据 ====================
        user_sequence = self._load_user_data(uid)  # 动态加载用户数据

        # ==================== 构建扩展用户序列 ====================
        # 将原始序列转换为用户-物品交替的扩展序列，同时提取时间特征
        ext_user_sequence = []
        timestamps = []  # 收集时间戳用于计算时间间隔
        
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, timestamp = record_tuple
            
            # ==================== 提取时间特征 ====================
            time_features = {}
            if timestamp:
                # 将时间戳转换为datetime对象
                dt = datetime.fromtimestamp(timestamp)
                
                # 提取时间特征
                time_features['time_of_day'] = dt.hour          # 0-23，表示一天中的小时
                time_features['day_of_week'] = dt.weekday()     # 0-6，表示一周中的天（0=周一）
                
                timestamps.append(timestamp)
            else:
                # 如果没有时间戳，使用默认值
                time_features['time_of_day'] = 0
                time_features['day_of_week'] = 0
                timestamps.append(0)
            
            # ==================== 计算时间间隔特征 ====================
            if len(timestamps) > 1 and timestamps[-1] > 0 and timestamps[-2] > 0:
                # time_delta: 与上一个行为的时间间隔（秒）
                time_features['time_delta'] = timestamps[-1] - timestamps[-2]
            else:
                time_features['time_delta'] = 0  # 第一个行为或无效时间戳的默认值
            
            # 添加用户token（插入到序列开头）
            if u and user_feat:
                # 将时间特征合并到用户特征中
                enhanced_user_feat = {**user_feat, **time_features}
                ext_user_sequence.insert(0, (u, enhanced_user_feat, 2, action_type))  # type=2表示用户token
                
            # 添加物品token（追加到序列末尾）
            if i and item_feat:
                # 将时间特征合并到物品特征中
                enhanced_item_feat = {**item_feat, **time_features}
                ext_user_sequence.append((i, enhanced_item_feat, 1, action_type))     # type=1表示物品token

        # ==================== 初始化输出数组 ====================
        # 预分配固定长度的数组（maxlen+1），用于存储序列数据
        seq = np.zeros([self.maxlen + 1], dtype=np.int32)                    # 序列ID
        pos = np.zeros([self.maxlen + 1], dtype=np.int32)                    # 正样本ID
        neg = np.zeros([self.maxlen + 1], dtype=np.int32)                    # 负样本ID
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)             # token类型
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)        # 下一个token类型
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)       # 下一个动作类型

        # 特征数组（使用object类型存储字典）
        seq_feat = np.empty([self.maxlen + 1], dtype=object)                 # 序列特征
        pos_feat = np.empty([self.maxlen + 1], dtype=object)                 # 正样本特征
        neg_feat = np.empty([self.maxlen + 1], dtype=object)                 # 负样本特征

        # ==================== 序列填充处理 ====================
        nxt = ext_user_sequence[-1]  # 下一个token（用于生成正样本）
        idx = self.maxlen           # 从右向左填充

        # 收集用户历史交互的所有物品（用于负采样）
        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:  # 只收集物品token
                ts.add(record_tuple[0])

        # left-padding: 从后往前遍历，将用户序列填充到maxlen+1的长度
        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_, act_type = record_tuple
            next_i, next_feat, next_type, next_act_type = nxt
            
            # ==================== 特征处理 ====================
            # 填充缺失的特征
            feat = self.fill_missing_feat(feat, i)
            next_feat = self.fill_missing_feat(next_feat, next_i)
            
            # 填充序列信息
            seq[idx] = i                           # 当前token ID
            token_type[idx] = type_                # 当前token类型
            next_token_type[idx] = next_type       # 下一个token类型
            
            if next_act_type is not None:
                next_action_type[idx] = next_act_type  # 下一个动作类型
                
            seq_feat[idx] = feat                   # 当前token特征

            # ==================== 正负样本生成 ====================
            # 只为物品token生成正负样本
            if next_type == 1 and next_i != 0:
                pos[idx] = next_i                  # 正样本：下一个真实交互的物品
                pos_feat[idx] = next_feat          # 正样本特征
                
                # 负采样：随机选择用户未交互过的物品
                neg_id = self._random_neq(1, self.itemnum + 1, ts)
                neg[idx] = neg_id                  # 负样本ID
                neg_feat[idx] = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)  # 负样本特征
                
            nxt = record_tuple  # 更新下一个token
            idx -= 1           # 向左移动填充位置
            
            if idx == -1:      # 超出序列长度限制
                break

        # ==================== 特征默认值填充 ====================
        # 将None值替换为默认特征值
        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        pos_feat = np.where(pos_feat == None, self.feature_default_value, pos_feat)
        neg_feat = np.where(neg_feat == None, self.feature_default_value, neg_feat)

        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat

    def __len__(self):
        """
        返回数据集长度，即用户数量

        Returns:
            usernum: 用户数量
        """
        return len(self.seq_offsets)

    def _init_feat_info(self):
        """
        初始化特征信息, 包括特征缺省值和特征类型
        
        定义了六种特征类型：
        - user_sparse: 用户稀疏特征（如性别、年龄段等）
        - item_sparse: 物品稀疏特征（如类别、品牌等）
        - user_array: 用户数组特征（如兴趣标签列表等）
        - item_array: 物品数组特征（如标签列表等）
        - user_continual: 用户连续特征（如活跃度等）
        - item_continual: 物品连续特征（如价格等）
        - item_emb: 物品多模态特征（如图像、文本embeddings）

        Returns:
            feat_default_value: 特征缺省值，每个元素为字典，key为特征ID，value为特征缺省值
            feat_types: 特征类型，key为特征类型名称，value为包含的特征ID列表
        """
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        
        # ==================== 特征类型定义 ====================
        # 用户特征
        feat_types['user_sparse'] = ['103', '104', '105', '109']              # 用户稀疏特征ID
        feat_types['user_array'] = ['106', '107', '108', '110']               # 用户数组特征ID
        feat_types['user_continual'] = []                                     # 用户连续特征ID（当前为空）
        
        # 物品特征
        feat_types['item_sparse'] = [                                         # 物品稀疏特征ID
            '100', '117', '111', '118', '101', '102', '119', '120', 
            '114', '112', '121', '115', '122', '116',
        ]
        feat_types['item_array'] = []                                         # 物品数组特征ID（当前为空）
        feat_types['item_continual'] = []                                     # 物品连续特征ID（当前为空）
        feat_types['item_emb'] = self.mm_emb_ids                             # 物品多模态特征ID
        
        # ==================== 时间特征定义 ====================
        # 时间特征：这些特征同时适用于用户和物品token
        feat_types['time_sparse'] = ['time_of_day', 'day_of_week']           # 时间稀疏特征
        feat_types['time_continual'] = ['time_delta']                        # 时间连续特征

        # ==================== 特征默认值和统计信息 ====================
        # 用户稀疏特征
        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0                                   # 稀疏特征默认值为0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])        # 特征值数量
            
        # 物品稀疏特征
        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
            
        # 物品数组特征
        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]                                 # 数组特征默认值为[0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
            
        # 用户数组特征
        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
            
        # 连续特征
        for feat_id in feat_types['user_continual']:
            feat_default_value[feat_id] = 0                                   # 连续特征默认值为0
        for feat_id in feat_types['item_continual']:
            feat_default_value[feat_id] = 0
            
        # 多模态特征
        for feat_id in feat_types['item_emb']:
            # 多模态特征默认值为零向量，维度与预训练embedding一致
            feat_default_value[feat_id] = np.zeros(
                list(self.mm_emb_dict[feat_id].values())[0].shape[0], dtype=np.float32
            )
            
        # ==================== 时间特征默认值和统计信息 ====================
        # 时间稀疏特征
        for feat_id in feat_types['time_sparse']:
            feat_default_value[feat_id] = 0                                   # 时间稀疏特征默认值为0
            if feat_id == 'time_of_day':
                feat_statistics[feat_id] = 24                                 # 0-23小时，共24种取值
            elif feat_id == 'day_of_week':
                feat_statistics[feat_id] = 7                                  # 0-6天，共7种取值
                
        # 时间连续特征
        for feat_id in feat_types['time_continual']:
            feat_default_value[feat_id] = 0                                   # 时间连续特征默认值为0

        return feat_default_value, feat_types, feat_statistics

    def fill_missing_feat(self, feat, item_id):
        """
        对于原始数据中缺失的特征进行填充缺省值
        
        确保每个样本都包含所有必要的特征，缺失的特征用默认值填充

        Args:
            feat: 特征字典
            item_id: 物品ID

        Returns:
            filled_feat: 填充后的特征字典
        """
        if feat == None:
            feat = {}
            
        filled_feat = {}
        
        # 复制现有特征
        for k in feat.keys():
            filled_feat[k] = feat[k]

        # ==================== 填充缺失的一般特征 ====================
        # 获取所有特征ID
        all_feat_ids = []
        for feat_type in self.feature_types.values():
            all_feat_ids.extend(feat_type)
            
        # 找出缺失的特征并填充默认值
        missing_fields = set(all_feat_ids) - set(feat.keys())
        for feat_id in missing_fields:
            filled_feat[feat_id] = self.feature_default_value[feat_id]
            
        # ==================== 特殊处理多模态特征 ====================
        # 多模态特征需要根据item_id从预训练embedding中获取
        for feat_id in self.feature_types['item_emb']:
            if item_id != 0 and self.indexer_i_rev[item_id] in self.mm_emb_dict[feat_id]:
                # 如果物品存在对应的多模态特征，则使用预训练的embedding
                if type(self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]) == np.ndarray:
                    filled_feat[feat_id] = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]

        return filled_feat

    @staticmethod
    def collate_fn(batch):
        """
        批处理函数，将多个样本组合成一个批次
        
        PyTorch DataLoader使用此函数将多个__getitem__返回的数据组合成批次

        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            pos: 正样本ID, torch.Tensor形式
            neg: 负样本ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            next_token_type: 下一个token类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            pos_feat: 正样本特征, list形式
            neg_feat: 负样本特征, list形式
        """
        # 解包批次数据
        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = zip(*batch)
        
        # 将numpy数组转换为PyTorch张量
        seq = torch.from_numpy(np.array(seq))
        pos = torch.from_numpy(np.array(pos))
        neg = torch.from_numpy(np.array(neg))
        token_type = torch.from_numpy(np.array(token_type))
        next_token_type = torch.from_numpy(np.array(next_token_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))
        
        # 特征保持为list格式（因为包含字典，无法直接转换为张量）
        seq_feat = list(seq_feat)
        pos_feat = list(pos_feat)
        neg_feat = list(neg_feat)
        
        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat


class MyTestDataset(MyDataset):
    """
    测试数据集（评估用）
    
    继承自MyDataset，但针对测试场景做了特殊处理：
    1. 使用不同的数据文件（predict_seq.jsonl）
    2. 不生成正负样本（只需要用户序列表示）
    3. 处理冷启动问题（训练时未见过的特征值）
    4. 返回user_id用于结果匹配
    """

    def __init__(self, data_dir, args):
        """
        初始化测试数据集
        
        大部分初始化逻辑与训练数据集相同，但数据文件不同
        """
        super().__init__(data_dir, args)

    def _load_data_and_offsets(self):
        """
        加载测试数据文件和偏移量索引
        
        测试数据使用predict_seq.jsonl文件
        """
        self.data_file = open(self.data_dir / "predict_seq.jsonl", 'rb')
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _process_cold_start_feat(self, feat):
        """
        处理冷启动特征
        
        训练集未出现过的特征value为字符串，默认转换为0
        可以根据实际需求设计更好的冷启动处理方法
        
        Args:
            feat: 原始特征字典
            
        Returns:
            dict: 处理后的特征字典
        """
        processed_feat = {}
        for feat_id, feat_value in feat.items():
            if type(feat_value) == list:
                # 处理数组类型特征
                value_list = []
                for v in feat_value:
                    if type(v) == str:
                        value_list.append(0)  # 未见过的字符串特征值设为0
                    else:
                        value_list.append(v)
                processed_feat[feat_id] = value_list
            elif type(feat_value) == str:
                # 处理单值字符串特征
                processed_feat[feat_id] = 0
            else:
                # 保持原值
                processed_feat[feat_id] = feat_value
        return processed_feat

    def __getitem__(self, uid):
        """
        获取单个用户的测试数据
        
        与训练数据集的主要区别：
        1. 不生成正负样本
        2. 处理冷启动特征
        3. 返回user_id用于结果匹配

        Args:
            uid: 用户在self.data_file中储存的行号
            
        Returns:
            seq: 用户序列ID
            token_type: 用户序列类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            user_id: user_id eg. user_xxxxxx ,便于后面对照答案
        """
        # ==================== 加载用户序列数据 ====================
        user_sequence = self._load_user_data(uid)  # 动态加载用户数据

        # ==================== 构建扩展用户序列 ====================
        ext_user_sequence = []
        timestamps = []  # 收集时间戳用于计算时间间隔
        
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, _, timestamp = record_tuple
            
            # ==================== 提取时间特征（测试数据集） ====================
            time_features = {}
            if timestamp:
                # 将时间戳转换为datetime对象
                dt = datetime.fromtimestamp(timestamp)
                
                # 提取时间特征
                time_features['time_of_day'] = dt.hour          # 0-23，表示一天中的小时
                time_features['day_of_week'] = dt.weekday()     # 0-6，表示一周中的天（0=周一）
                
                timestamps.append(timestamp)
            else:
                # 如果没有时间戳，使用默认值
                time_features['time_of_day'] = 0
                time_features['day_of_week'] = 0
                timestamps.append(0)
            
            # ==================== 计算时间间隔特征 ====================
            if len(timestamps) > 1 and timestamps[-1] > 0 and timestamps[-2] > 0:
                # time_delta: 与上一个行为的时间间隔（秒）
                time_features['time_delta'] = timestamps[-1] - timestamps[-2]
            else:
                time_features['time_delta'] = 0  # 第一个行为或无效时间戳的默认值
            
            # 处理用户信息
            if u:
                if type(u) == str:  # 如果是字符串，说明是user_id
                    user_id = u
                else:  # 如果是int，说明是re_id
                    user_id = self.indexer_u_rev[u]
                    
            # 添加用户token
            if u and user_feat:
                if type(u) == str:
                    u = 0  # 字符串用户ID在序列中用0表示
                if user_feat:
                    user_feat = self._process_cold_start_feat(user_feat)  # 处理冷启动特征
                    # 将时间特征合并到用户特征中
                    enhanced_user_feat = {**user_feat, **time_features}
                ext_user_sequence.insert(0, (u, enhanced_user_feat, 2))  # type=2表示用户token

            # 添加物品token
            if i and item_feat:
                # 对于训练时没见过的item，不会直接赋0，而是保留creative_id
                # creative_id远大于训练时的itemnum，这里将其设为0
                if i > self.itemnum:
                    i = 0
                if item_feat:
                    item_feat = self._process_cold_start_feat(item_feat)  # 处理冷启动特征
                    # 将时间特征合并到物品特征中
                    enhanced_item_feat = {**item_feat, **time_features}
                ext_user_sequence.append((i, enhanced_item_feat, 1))  # type=1表示物品token

        # ==================== 初始化输出数组 ====================
        seq = np.zeros([self.maxlen + 1], dtype=np.int32)         # 序列ID
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)  # token类型
        seq_feat = np.empty([self.maxlen + 1], dtype=object)      # 序列特征

        idx = self.maxlen  # 从右向左填充

        # 收集历史交互物品（虽然测试时不需要，但保持代码一致性）
        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        # ==================== 序列填充处理 ====================
        # left-padding: 从后往前填充序列
        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_ = record_tuple
            
            # 填充缺失特征
            feat = self.fill_missing_feat(feat, i)
            
            # 填充序列信息
            seq[idx] = i
            token_type[idx] = type_
            seq_feat[idx] = feat
            
            idx -= 1
            if idx == -1:
                break

        # ==================== 特征默认值填充 ====================
        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)

        return seq, token_type, seq_feat, user_id

    def __len__(self):
        """
        返回测试数据集长度

        Returns:
            len(self.seq_offsets): 用户数量
        """
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            temp = pickle.load(f)
        return len(temp)

    @staticmethod
    def collate_fn(batch):
        """
        测试数据集的批处理函数
        
        与训练数据集的collate_fn类似，但不包含正负样本

        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            user_id: user_id, str
        """
        seq, token_type, seq_feat, user_id = zip(*batch)
        
        # 转换为PyTorch张量
        seq = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))
        seq_feat = list(seq_feat)

        return seq, token_type, seq_feat, user_id


def save_emb(emb, save_path):
    """
    将Embedding保存为二进制文件
    
    保存格式：
    - 前8字节：num_points (uint32) + num_dimensions (uint32)
    - 后续：embedding数据 (float32)

    Args:
        emb: 要保存的Embedding，形状为 [num_points, num_dimensions]
        save_path: 保存路径
    """
    num_points = emb.shape[0]     # 数据点数量
    num_dimensions = emb.shape[1] # 向量的维度
    
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        # 写入头信息：数据点数量和维度
        f.write(struct.pack('II', num_points, num_dimensions))
        # 写入embedding数据
        emb.tofile(f)


def load_mm_emb(mm_path, feat_ids):
    """
    加载多模态特征Embedding
    
    支持加载不同类型的多模态特征，包括图像、文本等预训练embedding

    Args:
        mm_path: 多模态特征Embedding路径
        feat_ids: 要加载的多模态特征ID列表

    Returns:
        mm_emb_dict: 多模态特征Embedding字典，key为特征ID，value为特征Embedding字典（key为item ID，value为Embedding）
    """
    # 不同多模态特征的维度映射
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mm_emb_dict = {}
    
    # 逐个加载指定的多模态特征
    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
        shape = SHAPE_DICT[feat_id]  # 获取特征维度
        emb_dict = {}
        
        if feat_id != '81':
            # ==================== 加载JSON格式的多模态特征 ====================
            try:
                base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
                
                # 遍历所有JSON文件
                for json_file in base_path.glob('*.json'):
                    with open(json_file, 'r', encoding='utf-8') as file:
                        for line in file:
                            # 解析每行JSON数据
                            data_dict_origin = json.loads(line.strip())
                            insert_emb = data_dict_origin['emb']
                            
                            # 确保embedding是numpy数组格式
                            if isinstance(insert_emb, list):
                                insert_emb = np.array(insert_emb, dtype=np.float32)
                                
                            # 使用anonymous_cid作为key
                            data_dict = {data_dict_origin['anonymous_cid']: insert_emb}
                            emb_dict.update(data_dict)
                            
            except Exception as e:
                print(f"transfer error: {e}")
                
        if feat_id == '81':
            # ==================== 加载PKL格式的多模态特征 ====================
            # 特征ID 81使用pickle格式存储
            with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                emb_dict = pickle.load(f)
                
        mm_emb_dict[feat_id] = emb_dict
        print(f'Loaded #{feat_id} mm_emb')
        
    return mm_emb_dict
