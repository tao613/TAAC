import json
import pickle
import struct
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


class SmartNegativeSampler:
    """
    智能负采样器，提供多种负采样策略和渐进式训练
    """
    
    def __init__(self, item_feat_dict, itemnum, popularity_factor=0.75):
        self.item_feat_dict = item_feat_dict
        self.itemnum = itemnum
        self.popularity_factor = popularity_factor
        
        # 计算物品热度分布（基于物品ID的频率）
        self.item_popularity = self._compute_item_popularity()
        
        # 缓存热门物品列表
        self.popular_items = self._get_popular_items()
        
        # 渐进式训练参数
        self.curriculum_enabled = False
        self.current_epoch = 1
        self.total_epochs = 1
        self.difficulty_schedule = "linear"  # linear, cosine, exponential
        
    def _compute_item_popularity(self):
        """
        计算物品热度分布（简化版，假设物品ID越小越热门）
        """
        # 简化假设：物品ID越小，热度越高
        popularity = {}
        for item_id in range(1, self.itemnum + 1):
            # 使用幂律分布模拟热度
            popularity[item_id] = 1.0 / (item_id ** self.popularity_factor)
        
        # 归一化
        total = sum(popularity.values())
        for item_id in popularity:
            popularity[item_id] /= total
            
        return popularity
    
    def _get_popular_items(self, top_k=1000):
        """
        获取热门物品列表
        """
        sorted_items = sorted(self.item_popularity.items(), key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in sorted_items[:top_k]]
    
    def random_negative_sampling(self, user_seq, num_negatives=1):
        """
        原始随机负采样
        """
        negatives = []
        for _ in range(num_negatives):
            neg_id = self._random_neq(1, self.itemnum + 1, set(user_seq))
            negatives.append(neg_id)
        return negatives
    
    def popular_negative_sampling(self, user_seq, num_negatives=1):
        """
        热门物品负采样：从热门物品中采样负样本
        """
        user_seq_set = set(user_seq)
        available_popular = [item for item in self.popular_items if item not in user_seq_set]
        
        if len(available_popular) < num_negatives:
            # 如果热门物品不够，补充随机采样
            negatives = available_popular[:]
            remaining = num_negatives - len(negatives)
            negatives.extend(self.random_negative_sampling(user_seq, remaining))
        else:
            negatives = np.random.choice(available_popular, num_negatives, replace=False).tolist()
        
        return negatives
    
    def mixed_negative_sampling(self, user_seq, num_negatives=1, popular_ratio=0.3, random_ratio=0.7):
        """
        混合负采样策略：结合热门物品和随机采样
        """
        num_popular = int(num_negatives * popular_ratio)
        num_random = num_negatives - num_popular
        
        negatives = []
        
        # 热门物品负采样
        if num_popular > 0:
            popular_negs = self.popular_negative_sampling(user_seq, num_popular)
            negatives.extend(popular_negs)
        
        # 随机负采样
        if num_random > 0:
            random_negs = self.random_negative_sampling(user_seq, num_random)
            negatives.extend(random_negs)
        
        return negatives
    
    def enable_curriculum_learning(self, total_epochs, difficulty_schedule="linear"):
        """
        启用渐进式训练
        
        Args:
            total_epochs: 总训练轮数
            difficulty_schedule: 难度调度策略 ("linear", "cosine", "exponential")
        """
        self.curriculum_enabled = True
        self.total_epochs = total_epochs
        self.difficulty_schedule = difficulty_schedule
        print(f"渐进式训练已启用: {difficulty_schedule} schedule, {total_epochs} epochs")
    
    def update_epoch(self, epoch):
        """
        更新当前epoch，用于调整负采样难度
        
        Args:
            epoch: 当前训练轮数
        """
        self.current_epoch = epoch
    
    def _get_difficulty_factor(self):
        """
        根据当前epoch计算难度因子 (0.0 - 1.0)
        0.0 表示最简单（随机采样），1.0 表示最困难（热门物品采样）
        """
        if not self.curriculum_enabled:
            return 0.5  # 默认中等难度
        
        progress = (self.current_epoch - 1) / max(1, self.total_epochs - 1)
        progress = min(1.0, max(0.0, progress))  # 限制在 [0, 1]
        
        if self.difficulty_schedule == "linear":
            return progress
        elif self.difficulty_schedule == "cosine":
            # 余弦调度：开始慢，中间快，结束慢
            return 0.5 * (1 - np.cos(np.pi * progress))
        elif self.difficulty_schedule == "exponential":
            # 指数调度：开始很慢，后期快速增长
            return progress ** 2
        else:
            return progress
    
    def curriculum_negative_sampling(self, user_seq, num_negatives=1):
        """
        基于课程学习的负采样策略
        根据当前训练进度动态调整负样本难度
        
        Args:
            user_seq: 用户交互序列
            num_negatives: 需要采样的负样本数量
            
        Returns:
            negatives: 负样本列表
        """
        if not self.curriculum_enabled:
            return self.mixed_negative_sampling(user_seq, num_negatives)
        
        difficulty = self._get_difficulty_factor()
        
        # 根据难度因子调整采样策略
        if difficulty < 0.3:
            # 早期：主要使用随机采样（简单负样本）
            popular_ratio = 0.1
            random_ratio = 0.9
        elif difficulty < 0.7:
            # 中期：平衡采样
            popular_ratio = 0.5
            random_ratio = 0.5
        else:
            # 后期：主要使用热门物品采样（困难负样本）
            popular_ratio = 0.8
            random_ratio = 0.2
        
        return self.mixed_negative_sampling(
            user_seq, num_negatives, popular_ratio, random_ratio
        )

    def _random_neq(self, l, r, s):
        """
        生成一个不在序列s中的随机整数
        """
        t = np.random.randint(l, r)
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t


class MyDataset(torch.utils.data.Dataset):
    """
    用户序列数据集

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
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self._load_data_and_offsets()
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id

        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)
            self.itemnum = len(indexer['i'])
            self.usernum = len(indexer['u'])
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer

        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()
        
        # 初始化智能负采样器（优化项）
        self.smart_sampler = SmartNegativeSampler(self.item_feat_dict, self.itemnum)
        
        # 如果启用课程学习，配置智能采样器
        if hasattr(args, 'use_curriculum_learning') and args.use_curriculum_learning:
            schedule = getattr(args, 'curriculum_schedule', 'linear')
            self.smart_sampler.enable_curriculum_learning(
                total_epochs=args.num_epochs,
                difficulty_schedule=schedule
            )

    def _load_data_and_offsets(self):
        """
        加载用户序列数据和每一行的文件偏移量(预处理好的), 用于快速随机访问数据并I/O
        """
        self.data_file = open(self.data_dir / "seq.jsonl", 'rb')
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _load_user_data(self, uid):
        """
        从数据文件中加载单个用户的数据

        Args:
            uid: 用户ID(reid)

        Returns:
            data: 用户序列数据，格式为[(user_id, item_id, user_feat, item_feat, action_type, timestamp)]
        """
        self.data_file.seek(self.seq_offsets[uid])
        line = self.data_file.readline()
        data = json.loads(line)
        return data

    def _random_neq(self, l, r, s):
        """
        生成一个不在序列s中的随机整数, 用于训练时的负采样

        Args:
            l: 随机整数的最小值
            r: 随机整数的最大值
            s: 序列

        Returns:
            t: 不在序列s中的随机整数
        """
        t = np.random.randint(l, r)
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

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
        user_sequence = self._load_user_data(uid)  # 动态加载用户数据

        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, _ = record_tuple
            if u and user_feat:
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type))
            if i and item_feat:
                ext_user_sequence.append((i, item_feat, 1, action_type))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        pos = np.zeros([self.maxlen + 1], dtype=np.int32)
        neg = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)

        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        pos_feat = np.empty([self.maxlen + 1], dtype=object)
        neg_feat = np.empty([self.maxlen + 1], dtype=object)

        nxt = ext_user_sequence[-1]
        idx = self.maxlen

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        # left-padding, 从后往前遍历，将用户序列填充到maxlen+1的长度
        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_, act_type = record_tuple
            next_i, next_feat, next_type, next_act_type = nxt
            feat = self.fill_missing_feat(feat, i)
            next_feat = self.fill_missing_feat(next_feat, next_i)
            seq[idx] = i
            token_type[idx] = type_
            next_token_type[idx] = next_type
            if next_act_type is not None:
                next_action_type[idx] = next_act_type
            seq_feat[idx] = feat
            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                pos_feat[idx] = next_feat
                
                # 使用智能负采样器（优化项）
                if hasattr(self, 'smart_sampler'):
                    # 优先使用课程学习负采样，如果未启用则使用混合负采样
                    if self.smart_sampler.curriculum_enabled:
                        neg_ids = self.smart_sampler.curriculum_negative_sampling(list(ts), num_negatives=1)
                    else:
                        neg_ids = self.smart_sampler.mixed_negative_sampling(list(ts), num_negatives=1)
                    neg_id = neg_ids[0]
                else:
                    # 原始随机负采样
                    neg_id = self._random_neq(1, self.itemnum + 1, ts)
                
                neg[idx] = neg_id
                neg_feat[idx] = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)
            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break

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
    
    def update_epoch(self, epoch):
        """
        更新当前训练epoch，用于课程学习中的负采样难度调整
        
        Args:
            epoch: 当前训练轮数
        """
        if hasattr(self, 'smart_sampler') and self.smart_sampler.curriculum_enabled:
            self.smart_sampler.update_epoch(epoch)
            difficulty = self.smart_sampler._get_difficulty_factor()
            print(f"Epoch {epoch}: 负采样难度因子 = {difficulty:.3f}")
        
    def get_curriculum_status(self):
        """
        获取课程学习状态信息
        
        Returns:
            dict: 包含课程学习状态的字典
        """
        if hasattr(self, 'smart_sampler') and self.smart_sampler.curriculum_enabled:
            difficulty = self.smart_sampler._get_difficulty_factor()
            return {
                'enabled': True,
                'current_epoch': self.smart_sampler.current_epoch,
                'total_epochs': self.smart_sampler.total_epochs,
                'difficulty_factor': difficulty,
                'schedule': self.smart_sampler.difficulty_schedule
            }
        else:
            return {'enabled': False}

    def _init_feat_info(self):
        """
        初始化特征信息, 包括特征缺省值和特征类型

        Returns:
            feat_default_value: 特征缺省值，每个元素为字典，key为特征ID，value为特征缺省值
            feat_types: 特征类型，key为特征类型名称，value为包含的特征ID列表
        """
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        feat_types['user_sparse'] = ['103', '104', '105', '109']
        feat_types['item_sparse'] = [
            '100',
            '117',
            '111',
            '118',
            '101',
            '102',
            '119',
            '120',
            '114',
            '112',
            '121',
            '115',
            '122',
            '116',
        ]
        feat_types['item_array'] = []
        feat_types['user_array'] = ['106', '107', '108', '110']
        feat_types['item_emb'] = self.mm_emb_ids
        feat_types['user_continual'] = []
        feat_types['item_continual'] = []

        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_emb']:
            feat_default_value[feat_id] = np.zeros(
                list(self.mm_emb_dict[feat_id].values())[0].shape[0], dtype=np.float32
            )

        return feat_default_value, feat_types, feat_statistics

    def fill_missing_feat(self, feat, item_id):
        """
        对于原始数据中缺失的特征进行填充缺省值

        Args:
            feat: 特征字典
            item_id: 物品ID

        Returns:
            filled_feat: 填充后的特征字典
        """
        if feat == None:
            feat = {}
        filled_feat = {}
        for k in feat.keys():
            filled_feat[k] = feat[k]

        all_feat_ids = []
        for feat_type in self.feature_types.values():
            all_feat_ids.extend(feat_type)
        missing_fields = set(all_feat_ids) - set(feat.keys())
        for feat_id in missing_fields:
            filled_feat[feat_id] = self.feature_default_value[feat_id]
        for feat_id in self.feature_types['item_emb']:
            if item_id != 0 and self.indexer_i_rev[item_id] in self.mm_emb_dict[feat_id]:
                if type(self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]) == np.ndarray:
                    filled_feat[feat_id] = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]

        return filled_feat

    @staticmethod
    def collate_fn(batch):
        """
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
        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        pos = torch.from_numpy(np.array(pos))
        neg = torch.from_numpy(np.array(neg))
        token_type = torch.from_numpy(np.array(token_type))
        next_token_type = torch.from_numpy(np.array(next_token_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))
        seq_feat = list(seq_feat)
        pos_feat = list(pos_feat)
        neg_feat = list(neg_feat)
        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat


class MyTestDataset(MyDataset):
    """
    测试数据集
    """

    def __init__(self, data_dir, args):
        super().__init__(data_dir, args)

    def _load_data_and_offsets(self):
        self.data_file = open(self.data_dir / "predict_seq.jsonl", 'rb')
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _process_cold_start_feat(self, feat):
        """
        处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.可设计替换为更好的方法。
        """
        processed_feat = {}
        for feat_id, feat_value in feat.items():
            if type(feat_value) == list:
                value_list = []
                for v in feat_value:
                    if type(v) == str:
                        value_list.append(0)
                    else:
                        value_list.append(v)
                processed_feat[feat_id] = value_list
            elif type(feat_value) == str:
                processed_feat[feat_id] = 0
            else:
                processed_feat[feat_id] = feat_value
        return processed_feat

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户在self.data_file中储存的行号
        Returns:
            seq: 用户序列ID
            token_type: 用户序列类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            user_id: user_id eg. user_xxxxxx ,便于后面对照答案
        """
        user_sequence = self._load_user_data(uid)  # 动态加载用户数据

        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, _, _ = record_tuple
            if u:
                if type(u) == str:  # 如果是字符串，说明是user_id
                    user_id = u
                else:  # 如果是int，说明是re_id
                    user_id = self.indexer_u_rev[u]
            if u and user_feat:
                if type(u) == str:
                    u = 0
                if user_feat:
                    user_feat = self._process_cold_start_feat(user_feat)
                ext_user_sequence.insert(0, (u, user_feat, 2))

            if i and item_feat:
                # 序列对于训练时没见过的item，不会直接赋0，而是保留creative_id，creative_id远大于训练时的itemnum
                if i > self.itemnum:
                    i = 0
                if item_feat:
                    item_feat = self._process_cold_start_feat(item_feat)
                ext_user_sequence.append((i, item_feat, 1))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat = np.empty([self.maxlen + 1], dtype=object)

        idx = self.maxlen

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_ = record_tuple
            feat = self.fill_missing_feat(feat, i)
            seq[idx] = i
            token_type[idx] = type_
            seq_feat[idx] = feat
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)

        return seq, token_type, seq_feat, user_id

    def __len__(self):
        """
        Returns:
            len(self.seq_offsets): 用户数量
        """
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            temp = pickle.load(f)
        return len(temp)

    @staticmethod
    def collate_fn(batch):
        """
        将多个__getitem__返回的数据拼接成一个batch

        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            user_id: user_id, str
        """
        seq, token_type, seq_feat, user_id = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))
        seq_feat = list(seq_feat)

        return seq, token_type, seq_feat, user_id


def save_emb(emb, save_path):
    """
    将Embedding保存为二进制文件

    Args:
        emb: 要保存的Embedding，形状为 [num_points, num_dimensions]
        save_path: 保存路径
    """
    num_points = emb.shape[0]  # 数据点数量
    num_dimensions = emb.shape[1]  # 向量的维度
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)


def load_mm_emb(mm_path, feat_ids):
    """
    加载多模态特征Embedding

    Args:
        mm_path: 多模态特征Embedding路径
        feat_ids: 要加载的多模态特征ID列表

    Returns:
        mm_emb_dict: 多模态特征Embedding字典，key为特征ID，value为特征Embedding字典（key为item ID，value为Embedding）
    """
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mm_emb_dict = {}
    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
        shape = SHAPE_DICT[feat_id]
        emb_dict = {}
        if feat_id != '81':
            try:
                base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
                for json_file in base_path.glob('*.json'):
                    with open(json_file, 'r', encoding='utf-8') as file:
                        for line in file:
                            data_dict_origin = json.loads(line.strip())
                            insert_emb = data_dict_origin['emb']
                            if isinstance(insert_emb, list):
                                insert_emb = np.array(insert_emb, dtype=np.float32)
                            data_dict = {data_dict_origin['anonymous_cid']: insert_emb}
                            emb_dict.update(data_dict)
            except Exception as e:
                print(f"transfer error: {e}")
        if feat_id == '81':
            with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
                emb_dict = pickle.load(f)
        mm_emb_dict[feat_id] = emb_dict
        print(f'Loaded #{feat_id} mm_emb')
    return mm_emb_dict