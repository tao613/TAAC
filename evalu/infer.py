import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MyTestDataset, save_emb
from model import BaselineModel


def get_ckpt_path():
    """
    获取训练好的模型权重文件路径
    
    Returns:
        str: 模型权重文件的完整路径
    """
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if ckpt_path is None:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    
    # 遍历目录，找到.pt文件
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)


def get_args():
    """
    解析评估推理的命令行参数
    
    Returns:
        args: 包含所有评估参数的命名空间对象
    """
    parser = argparse.ArgumentParser()

    # 训练参数（需要与训练时保持一致）
    parser.add_argument('--batch_size', default=128, type=int, help='推理批次大小')
    parser.add_argument('--lr', default=0.001, type=float, help='学习率（仅为兼容性）')
    parser.add_argument('--maxlen', default=101, type=int, help='用户序列最大长度')

    # Baseline 模型构建参数（需要与训练时保持一致）
    parser.add_argument('--hidden_units', default=32, type=int, help='隐藏层维度')
    parser.add_argument('--num_blocks', default=1, type=int, help='Transformer块数量')
    parser.add_argument('--num_epochs', default=3, type=int, help='训练轮数（仅为兼容性）')
    parser.add_argument('--num_heads', default=1, type=int, help='多头注意力头数')
    parser.add_argument('--dropout_rate', default=0.2, type=float, help='Dropout概率')
    parser.add_argument('--l2_emb', default=0.0, type=float, help='Embedding L2正则化权重')
    parser.add_argument('--device', default='cuda', type=str, help='推理设备')
    parser.add_argument('--inference_only', action='store_true', help='仅推理模式')
    parser.add_argument('--state_dict_path', default=None, type=str, help='预训练模型权重路径')
    parser.add_argument('--norm_first', action='store_true', help='是否在注意力层前先进行LayerNorm')

    # 多模态特征ID配置
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, 
                       choices=[str(s) for s in range(81, 87)],
                       help='激活的多模态特征ID列表')

    args = parser.parse_args()

    return args


def read_result_ids(file_path):
    """
    读取ANN检索结果文件
    
    Args:
        file_path: 检索结果文件路径
        
    Returns:
        numpy.ndarray: 检索结果，形状为 [num_queries, top_k]
    """
    with open(file_path, 'rb') as f:
        # 读取文件头信息
        num_points_query = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes，查询数量
        query_ann_top_k = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes，每个查询返回的top-k数量

        print(f"num_points_query: {num_points_query}, query_ann_top_k: {query_ann_top_k}")

        # 计算结果ID的总数量
        num_result_ids = num_points_query * query_ann_top_k

        # 读取结果ID（uint64_t，每个值8字节）
        result_ids = np.fromfile(f, dtype=np.uint64, count=num_result_ids)

        # 重塑为二维数组：[查询数量, top-k]
        return result_ids.reshape((num_points_query, query_ann_top_k))


def process_cold_start_feat(feat):
    """
    处理冷启动特征
    
    训练集中未出现过的特征值为字符串，默认转换为0
    可以根据实际需求设计更好的处理方法
    
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
                    value_list.append(0)  # 未见过的字符串值设为0
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


def get_candidate_emb(indexer, feat_types, feat_default_value, mm_emb_dict, model):
    """
    生成候选库item的ID和embedding
    
    这个函数处理候选物品池，为每个候选物品生成embedding用于检索
    
    Args:
        indexer: 索引字典，用于ID映射
        feat_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feature_default_value: 特征缺省值
        mm_emb_dict: 多模态特征字典
        model: 训练好的推荐模型
        
    Returns:
        dict: retrieve_id到creative_id的映射字典
    """
    # 多模态特征维度映射
    EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    
    # 读取候选物品集合
    candidate_path = Path(os.environ.get('EVAL_DATA_PATH'), 'predict_set.jsonl')
    item_ids, creative_ids, retrieval_ids, features = [], [], [], []
    retrieve_id2creative_id = {}

    print("Processing candidate items...")
    with open(candidate_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            
            # ==================== 特征处理 ====================
            # 读取item特征，并补充缺失值
            feature = line['features']
            creative_id = line['creative_id']  # 原始物品ID
            retrieval_id = line['retrieval_id']  # 检索用ID（从0开始）
            
            # 将creative_id映射为训练时的item_id，如果不存在则设为0
            item_id = indexer[creative_id] if creative_id in indexer else 0
            
            # 找出缺失的特征并用默认值填充
            missing_fields = set(
                feat_types['item_sparse'] + feat_types['item_array'] + feat_types['item_continual']
            ) - set(feature.keys())
            
            # 处理冷启动特征
            feature = process_cold_start_feat(feature)
            
            # 填充缺失特征的默认值
            for feat_id in missing_fields:
                feature[feat_id] = feat_default_value[feat_id]
                
            # 添加多模态特征
            for feat_id in feat_types['item_emb']:
                if creative_id in mm_emb_dict[feat_id]:
                    feature[feat_id] = mm_emb_dict[feat_id][creative_id]
                else:
                    # 如果多模态特征不存在，用零向量填充
                    feature[feat_id] = np.zeros(EMB_SHAPE_DICT[feat_id], dtype=np.float32)

            # 收集处理后的数据
            item_ids.append(item_id)
            creative_ids.append(creative_id)
            retrieval_ids.append(retrieval_id)
            features.append(feature)
            retrieve_id2creative_id[retrieval_id] = creative_id

    # ==================== 生成候选库Embedding ====================
    print("Generating candidate item embeddings...")
    # 调用模型方法生成并保存候选库embedding
    model.save_item_emb(item_ids, retrieval_ids, features, os.environ.get('EVAL_RESULT_PATH'))
    
    # 保存检索ID到创意ID的映射关系
    with open(Path(os.environ.get('EVAL_RESULT_PATH'), "retrive_id2creative_id.json"), "w") as f:
        json.dump(retrieve_id2creative_id, f)
        
    return retrieve_id2creative_id


def infer():
    """
    主推理函数
    
    执行完整的推理流程：
    1. 加载测试数据和训练好的模型
    2. 生成用户query embedding
    3. 生成候选物品embedding
    4. 执行ANN检索
    5. 返回top-k推荐结果
    
    Returns:
        tuple: (top10s推荐结果列表, 用户ID列表)
    """
    print("Starting inference...")
    
    # ==================== 参数和数据准备 ====================
    args = get_args()
    data_path = os.environ.get('EVAL_DATA_PATH')
    
    # 创建测试数据集和数据加载器
    test_dataset = MyTestDataset(data_path, args)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=test_dataset.collate_fn
    )
    
    # 获取数据集统计信息
    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    feat_statistics, feat_types = test_dataset.feat_statistics, test_dataset.feature_types
    
    # ==================== 模型加载 ====================
    print("Loading trained model...")
    # 创建模型实例（需要与训练时的结构一致）
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    model.eval()  # 设置为评估模式

    # 加载训练好的模型权重
    ckpt_path = get_ckpt_path()
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(args.device)))
    print(f"Model loaded from: {ckpt_path}")
    
    # ==================== 生成用户Query Embedding ====================
    print("Generating user query embeddings...")
    all_embs = []  # 存储所有用户的embedding
    user_list = []  # 存储所有用户ID
    
    # 遍历测试数据，为每个用户生成query embedding
    for step, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        seq, token_type, seq_feat, user_id = batch
        seq = seq.to(args.device)
        
        # 使用模型的predict方法生成用户表示
        logits = model.predict(seq, seq_feat, token_type)
        
        # 将结果转换为numpy格式并收集
        for i in range(logits.shape[0]):
            emb = logits[i].unsqueeze(0).detach().cpu().numpy().astype(np.float32)
            all_embs.append(emb)
        user_list += user_id

    # ==================== 生成候选库Embedding ====================
    print("Processing candidate items...")
    # 生成候选库的embedding以及ID映射文件
    retrieve_id2creative_id = get_candidate_emb(
        test_dataset.indexer['i'],              # 物品索引映射
        test_dataset.feature_types,             # 特征类型信息
        test_dataset.feature_default_value,     # 特征默认值
        test_dataset.mm_emb_dict,              # 多模态特征字典
        model,                                 # 训练好的模型
    )
    
    # 合并所有用户的query embedding
    all_embs = np.concatenate(all_embs, axis=0)
    
    # ==================== 保存Query文件 ====================
    # 保存用户query embedding用于ANN检索
    save_emb(all_embs, Path(os.environ.get('EVAL_RESULT_PATH'), 'query.fbin'))
    print(f"Saved {len(all_embs)} user query embeddings")
    
    # ==================== 执行ANN检索 ====================
    print("Performing ANN search...")
    # 构建ANN检索命令
    ann_cmd = (
        str(Path("/workspace", "faiss-based-ann", "faiss_demo"))  # ANN检索可执行文件
        + " --dataset_vector_file_path="
        + str(Path(os.environ.get("EVAL_RESULT_PATH"), "embedding.fbin"))  # 候选库embedding文件
        + " --dataset_id_file_path="
        + str(Path(os.environ.get("EVAL_RESULT_PATH"), "id.u64bin"))      # 候选库ID文件
        + " --query_vector_file_path="
        + str(Path(os.environ.get("EVAL_RESULT_PATH"), "query.fbin"))     # 用户query文件
        + " --result_id_file_path="
        + str(Path(os.environ.get("EVAL_RESULT_PATH"), "id100.u64bin"))   # 检索结果文件
        + " --query_ann_top_k=10"           # 每个query返回top-10结果
        + " --faiss_M=64"                   # HNSW图的连接数
        + " --faiss_ef_construction=1280"   # 构建时的候选数量
        + " --query_ef_search=640"          # 搜索时的候选数量
        + " --faiss_metric_type=0"          # 距离度量类型（0表示内积）
    )
    
    # 执行ANN检索
    os.system(ann_cmd)
    print("ANN search completed")

    # ==================== 处理检索结果 ====================
    print("Processing search results...")
    # 读取ANN检索结果
    top10s_retrieved = read_result_ids(Path(os.environ.get("EVAL_RESULT_PATH"), "id100.u64bin"))
    
    # 将检索ID转换为creative_id
    top10s_untrimmed = []
    for top10 in tqdm(top10s_retrieved, desc="Converting retrieval IDs"):
        for item in top10:
            # 通过映射字典将检索ID转换为原始创意ID
            top10s_untrimmed.append(retrieve_id2creative_id.get(int(item), 0))

    # 重新组织为每个用户的top-10列表
    top10s = [top10s_untrimmed[i : i + 10] for i in range(0, len(top10s_untrimmed), 10)]
    
    print(f"Generated recommendations for {len(top10s)} users")
    return top10s, user_list
