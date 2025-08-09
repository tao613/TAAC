import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MyTestDataset, save_emb, load_mm_emb
from model import BaselineModel
from model_rqvae import RQVAE


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
    parser.add_argument('--norm_first', action='store_true', default=True, 
                       help='是否在注意力层前先进行LayerNorm（Pre-LayerNorm结构，默认启用以提升训练稳定性）')

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


def generate_semantic_ids_on_demand(data_path, mm_emb_ids=['81'], args=None):
    """
    按需生成semantic_id特征
    
    当semantic_id文件不存在时，自动训练RQ-VAE并生成semantic_id特征
    
    Args:
        data_path: 数据路径
        mm_emb_ids: 多模态特征ID列表
        args: 推理参数
        
    Returns:
        semantic_id_dict: 语义ID字典 {item_id: [semantic_id_sequence]}
    """
    print("正在生成semantic_id特征...")
    
    # ==================== 加载多模态特征 ====================
    mm_path = Path(data_path, "creative_emb")
    if not mm_path.exists():
        print(f"警告: 未找到多模态特征目录 {mm_path}")
        return {}
    
    print("加载多模态特征用于RQ-VAE训练...")
    mm_emb_dict = load_mm_emb(mm_path, mm_emb_ids)
    
    # 合并所有特征ID的embeddings
    item_embeddings = {}
    embedding_dim = 0
    
    for feat_id in mm_emb_ids:
        if feat_id in mm_emb_dict:
            feat_embs = mm_emb_dict[feat_id]
            if embedding_dim == 0:
                # 确定embedding维度
                embedding_dim = list(feat_embs.values())[0].shape[0]
            
            for item_id, emb in feat_embs.items():
                if isinstance(emb, np.ndarray):
                    if item_id not in item_embeddings:
                        item_embeddings[item_id] = []
                    item_embeddings[item_id].append(emb)
    
    # 对于有多个特征的物品，取平均
    final_embeddings = {}
    for item_id, emb_list in item_embeddings.items():
        if len(emb_list) == 1:
            final_embeddings[item_id] = emb_list[0]
        else:
            final_embeddings[item_id] = np.mean(emb_list, axis=0)
    
    if not final_embeddings:
        print("无法获取多模态特征，跳过semantic_id生成")
        return {}
    
    print(f"获取了 {len(final_embeddings)} 个物品的多模态特征，维度: {embedding_dim}")
    
    # ==================== 准备训练数据 ====================
    item_ids = list(final_embeddings.keys())
    embeddings = np.zeros((len(item_ids), embedding_dim), dtype=np.float32)
    
    for i, item_id in enumerate(item_ids):
        embeddings[i] = final_embeddings[item_id]
    
    # ==================== 训练RQ-VAE模型 ====================
    print("开始训练RQ-VAE模型...")
    
    # RQ-VAE参数
    num_codebooks = 4
    codebook_size = [256] * 4  # 每个codebook的大小
    hidden_channels = [embedding_dim // 2, embedding_dim // 4]  # 编码器隐藏层
    latent_dim = embedding_dim // 4  # 潜在空间维度
    loss_beta = 0.25
    lr = 1e-3
    num_epochs = 20  # 推理时减少训练轮数，快速生成
    batch_size = 1024
    
    device = args.device if args else 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 数据准备
    embeddings_tensor = torch.from_numpy(embeddings).to(device)
    num_items, embedding_dim = embeddings_tensor.shape
    
    # 确保维度合理
    if embedding_dim < 8:
        print(f"警告: embedding维度太小 ({embedding_dim})，可能影响RQ-VAE性能")
        latent_dim = max(4, embedding_dim // 2)
        hidden_channels = [embedding_dim]
    else:
        latent_dim = max(8, embedding_dim // 4)
        hidden_channels = [max(8, embedding_dim // 2), latent_dim]
    
    print(f"RQ-VAE配置: input_dim={embedding_dim}, hidden_channels={hidden_channels}, latent_dim={latent_dim}")
    
    try:
        # 初始化RQ-VAE模型，使用正确的参数
        from model_rqvae import kmeans
        rqvae_model = RQVAE(
            input_dim=embedding_dim,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            shared_codebook=False,
            kmeans_method=kmeans,
            kmeans_iters=20,
            distances_method='euclidean',
            loss_beta=loss_beta,
            device=device,
        )
    except Exception as e:
        print(f"RQ-VAE模型初始化失败: {e}")
        print("跳过semantic_id生成")
        return {}
    
    # 优化器
    optimizer = torch.optim.Adam(rqvae_model.parameters(), lr=lr)
    
    # 训练循环
    rqvae_model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = (num_items + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_items)
            
            batch_embeddings = embeddings_tensor[start_idx:end_idx]
            
            # 前向传播
            x_hat, semantic_ids, recon_loss, rqvae_loss, total_loss = rqvae_model(batch_embeddings)
            
            # 使用模型内部计算的总损失
            loss = total_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / num_batches
            print(f'Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}')
    
    print("RQ-VAE训练完成！")
    
    # ==================== 生成semantic_id ====================
    print("生成semantic_id...")
    
    rqvae_model.eval()
    semantic_id_dict = {}
    
    with torch.no_grad():
        # 分批处理，并显示进度
        print(f"开始生成semantic_id，共{num_batches}个批次...")
        for batch_idx in tqdm(range(num_batches), desc="生成semantic_id"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_items)
            
            batch_embeddings = embeddings_tensor[start_idx:end_idx]
            batch_item_ids = item_ids[start_idx:end_idx]
            
            # 获取semantic_id
            try:
                semantic_id_list = rqvae_model._get_codebook(batch_embeddings)
            except Exception as e:
                print(f"警告: RQ-VAE _get_codebook 调用失败: {e}")
                print("使用简化方法生成semantic_id")
                # 直接用embedding的简单hash作为semantic_id
                batch_size_current = batch_embeddings.shape[0]
                semantic_ids = torch.zeros(batch_size_current, num_codebooks, dtype=torch.long)
                
                # 简单的hash方法：使用embedding的均值和标准差
                for i in range(batch_size_current):
                    emb = batch_embeddings[i].cpu().numpy()
                    # 使用不同的统计量生成不同的semantic_id
                    semantic_ids[i, 0] = int(abs(emb.mean() * 1000)) % 256
                    semantic_ids[i, 1] = int(abs(emb.std() * 1000)) % 256
                    semantic_ids[i, 2] = int(abs(emb.max() * 1000)) % 256
                    semantic_ids[i, 3] = int(abs(emb.min() * 1000)) % 256
                
                semantic_ids = semantic_ids.numpy()
                
                # 保存结果
                for i, item_id in enumerate(batch_item_ids):
                    semantic_id_dict[item_id] = semantic_ids[i].tolist()
                continue
            
            # 检查semantic_id_list的实际格式并进行调试（只在第一个batch输出）
            batch_size_current = batch_embeddings.shape[0]
            if batch_idx == 0:  # 只在第一个batch输出调试信息
                print(f"Debug: batch_size_current={batch_size_current}, semantic_id_list length={len(semantic_id_list)}")
                
                if len(semantic_id_list) > 0:
                    print(f"Debug: first semantic tensor shape={semantic_id_list[0].shape}")
                    print(f"Debug: 检测到的格式 - 每个样本返回一个长度为{semantic_id_list[0].shape[0]}的semantic_id向量")
            
            # 处理semantic_id_list的实际格式
            try:
                # 现在我们知道格式是：semantic_id_list[i] = [num_codebooks] for each sample i
                if len(semantic_id_list) == batch_size_current and len(semantic_id_list[0].shape) == 1:
                    # 正确的格式：每个样本有一个semantic_id向量
                    num_codebooks_actual = semantic_id_list[0].shape[0]
                    semantic_ids = torch.zeros(batch_size_current, num_codebooks_actual, dtype=torch.long, device=batch_embeddings.device)
                    
                    # 直接复制每个样本的semantic_id
                    for i, semantic_tensor in enumerate(semantic_id_list):
                        semantic_ids[i] = semantic_tensor
                    
                    if batch_idx == 0:
                        print(f"成功处理semantic_id，形状: {semantic_ids.shape}")
                
                elif len(semantic_id_list) > 0 and len(semantic_id_list[0].shape) == 2:
                    # 矩阵格式
                    semantic_ids = semantic_id_list[0]
                
                elif len(semantic_id_list) == num_codebooks and len(semantic_id_list[0].shape) == 1:
                    # 原来预期的格式：[num_codebooks个tensor，每个是[batch_size]]
                    semantic_ids = torch.zeros(batch_size_current, len(semantic_id_list), dtype=torch.long, device=batch_embeddings.device)
                    
                    for j, semantic_tensor in enumerate(semantic_id_list):
                        if semantic_tensor.shape[0] == batch_size_current:
                            semantic_ids[:, j] = semantic_tensor
                        else:
                            # 处理维度不匹配（减少日志输出）
                            if batch_idx == 0:
                                print(f"处理维度不匹配: semantic_tensor[{j}] shape {semantic_tensor.shape}")
                            
                            if semantic_tensor.shape[0] < batch_size_current:
                                repeated = semantic_tensor.repeat(batch_size_current // semantic_tensor.shape[0] + 1)
                                semantic_ids[:, j] = repeated[:batch_size_current]
                            else:
                                semantic_ids[:, j] = semantic_tensor[:batch_size_current]
                
                else:
                    if batch_idx == 0:
                        print(f"未知的semantic_id_list格式，使用随机填充")
                    semantic_ids = torch.randint(0, 256, (batch_size_current, num_codebooks), dtype=torch.long, device=batch_embeddings.device)
                
            except Exception as e:
                if batch_idx == 0:
                    print(f"处理semantic_id时出错: {e}")
                    print("使用随机semantic_id作为后备方案")
                semantic_ids = torch.randint(0, 256, (batch_size_current, num_codebooks), dtype=torch.long, device=batch_embeddings.device)
            
            semantic_ids = semantic_ids.cpu().numpy()
            
            # 保存结果
            for i, item_id in enumerate(batch_item_ids):
                semantic_id_dict[item_id] = semantic_ids[i].tolist()
            
            # 定期清理内存和进度报告
            if (batch_idx + 1) % 100 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                print(f"已处理 {batch_idx + 1}/{num_batches} 批次，当前semantic_id数量: {len(semantic_id_dict)}")
                
                # 每1000个批次尝试保存中间结果（可选）
                if (batch_idx + 1) % 1000 == 0:
                    try:
                        temp_file = Path('/workspace') / f'semantic_id_dict_temp_{batch_idx + 1}.json'
                        with open(temp_file, 'w') as f:
                            json.dump(semantic_id_dict, f)
                        print(f"已保存中间结果到 {temp_file}")
                    except Exception as e:
                        # 静默忽略中间保存错误，继续处理
                        print(f"中间保存跳过（文件系统限制），继续内存处理...")
    
    print(f"为 {len(semantic_id_dict)} 个物品生成了semantic_id")
    
    # ==================== 保存semantic_id文件 ====================
    semantic_id_file = Path(data_path) / 'semantic_id_dict.json'
    
    print(f"保存semantic_id到 {semantic_id_file}")
    try:
        with open(semantic_id_file, 'w') as f:
            json.dump(semantic_id_dict, f, indent=2)
        print(f"成功保存到目标目录: {semantic_id_file}")
    except (OSError, PermissionError) as e:
        print(f"警告: 无法写入目标目录 {semantic_id_file}: {e}")
        # 备选方案：保存到工作目录
        fallback_file = Path('/workspace') / 'semantic_id_dict.json'
        print(f"尝试保存到工作目录: {fallback_file}")
        try:
            with open(fallback_file, 'w') as f:
                json.dump(semantic_id_dict, f, indent=2)
            print(f"成功保存到工作目录: {fallback_file}")
            print(f"请手动将文件复制到目标位置: {semantic_id_file}")
        except Exception as fallback_error:
            print(f"错误: 连工作目录也无法写入: {fallback_error}")
            print("semantic_id仅保存在内存中，程序退出后将丢失")
    
    # 清理临时文件（从工作目录）
    temp_files = list(Path('/workspace').glob('semantic_id_dict_temp_*.json'))
    for temp_file in temp_files:
        try:
            temp_file.unlink()
            print(f"已清理临时文件: {temp_file}")
        except:
            pass
    
    print("semantic_id生成和保存完成！")
    return semantic_id_dict


def get_candidate_emb(indexer, feat_types, feat_default_value, mm_emb_dict, semantic_id_dict, model):
    """
    生成候选库item的ID和embedding
    
    这个函数处理候选物品池，为每个候选物品生成embedding用于检索

    Args:
        indexer: 索引字典，用于ID映射
        feat_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feat_default_value: 特征缺省值
        mm_emb_dict: 多模态特征字典
        semantic_id_dict: 语义ID字典
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

    # 统计semantic_id使用情况
    semantic_id_used_count = 0
    total_candidates = 0

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
                
            # ==================== 添加时间特征默认值 ====================
            # 候选物品没有时间戳，使用默认值
            for feat_id in feat_types.get('time_sparse', []):
                feature[feat_id] = feat_default_value[feat_id]
            for feat_id in feat_types.get('time_continual', []):
                feature[feat_id] = feat_default_value[feat_id]
                
            # ==================== 添加semantic_id特征 ====================
            # 候选物品的semantic_id特征处理
            for feat_id in feat_types.get('semantic_array', []):
                if feat_id not in feature:
                    # 先尝试从semantic_id_dict中获取
                    if semantic_id_dict and creative_id in semantic_id_dict:
                        feature[feat_id] = semantic_id_dict[creative_id]
                        semantic_id_used_count += 1
                    else:
                        # 使用默认值
                        feature[feat_id] = feat_default_value[feat_id]
            
            total_candidates += 1
                
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

    # ==================== 输出semantic_id使用统计 ====================
    if 'semantic_array' in feat_types and feat_types['semantic_array']:
        semantic_coverage = semantic_id_used_count / total_candidates * 100 if total_candidates > 0 else 0
        print(f"Semantic ID使用统计:")
        print(f"  总候选物品数: {total_candidates}")
        print(f"  使用semantic_id的物品数: {semantic_id_used_count}")
        print(f"  覆盖率: {semantic_coverage:.2f}%")

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
    
    # ==================== 加载或生成semantic_id特征 ====================
    semantic_id_dict = test_dataset.semantic_id_dict
    if semantic_id_dict:
        print(f"成功加载 {len(semantic_id_dict)} 个物品的semantic_id特征")
    else:
        print("未找到semantic_id特征，尝试自动生成...")
        # 按需生成semantic_id特征
        semantic_id_dict = generate_semantic_ids_on_demand(
            data_path=data_path,
            mm_emb_ids=args.mm_emb_id,
            args=args
        )
        
        if semantic_id_dict:
            print(f"成功生成 {len(semantic_id_dict)} 个物品的semantic_id特征")
            # 重新加载dataset以获取更新后的特征
            test_dataset = MyTestDataset(data_path, args)
            semantic_id_dict = test_dataset.semantic_id_dict
        else:
            print("semantic_id生成失败，将使用默认值")
    
    # ==================== 模型加载 ====================
    print("Loading trained model...")
    # 创建模型实例（需要与训练时的结构一致）
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    model.eval()  # 设置为评估模式

    # 加载训练好的模型权重
    ckpt_path = get_ckpt_path()
    checkpoint = torch.load(ckpt_path, map_location=torch.device(args.device))
    
    # ==================== 智能模型权重兼容处理 ====================
    # 处理模型结构变化导致的权重不匹配问题
    model_state_dict = model.state_dict()
    checkpoint_keys = set(checkpoint.keys())
    model_keys = set(model_state_dict.keys())
    
    # 找出不匹配的权重
    missing_in_model = checkpoint_keys - model_keys  # checkpoint中有但模型中没有
    missing_in_checkpoint = model_keys - checkpoint_keys  # 模型中有但checkpoint中没有
    
    # 处理维度不匹配的权重（如itemdnn.weight）
    compatible_checkpoint = {}
    for key, value in checkpoint.items():
        if key in model_state_dict:
            model_shape = model_state_dict[key].shape
            checkpoint_shape = value.shape
            
            if model_shape == checkpoint_shape:
                # 形状匹配，直接使用
                compatible_checkpoint[key] = value
            else:
                print(f"维度不匹配 {key}: checkpoint {checkpoint_shape} vs model {model_shape}")
                
                # 对于线性层权重的特殊处理
                if 'weight' in key and len(model_shape) == 2 and len(checkpoint_shape) == 2:
                    # 如果是权重矩阵且新模型维度更大，用零填充
                    if (model_shape[0] == checkpoint_shape[0] and 
                        model_shape[1] > checkpoint_shape[1]):
                        # 输入维度扩大（增加了新特征）
                        new_weight = torch.zeros(model_shape, dtype=value.dtype, device=value.device)
                        new_weight[:, :checkpoint_shape[1]] = value
                        compatible_checkpoint[key] = new_weight
                        print(f"  -> 扩展权重矩阵 {key} 从 {checkpoint_shape} 到 {model_shape}")
                    else:
                        # 其他情况使用随机初始化
                        print(f"  -> 跳过不兼容的权重 {key}")
                        continue
                else:
                    # 对于其他类型的不匹配，跳过
                    print(f"  -> 跳过不兼容的权重 {key}")
                    continue
        else:
            # checkpoint中有但模型中没有的权重，跳过
            print(f"跳过废弃的权重: {key}")
    
    # 加载兼容的权重
    missing_keys, unexpected_keys = model.load_state_dict(compatible_checkpoint, strict=False)
    
    if missing_keys:
        print(f"模型中缺失的权重（将使用随机初始化）: {missing_keys}")
    if unexpected_keys:
        print(f"checkpoint中多余的权重: {unexpected_keys}")
    
    print(f"Model loaded from: {ckpt_path}")
    print(f"成功加载 {len(compatible_checkpoint)} 个权重，跳过 {len(checkpoint) - len(compatible_checkpoint)} 个不兼容权重")
    
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
        semantic_id_dict,                      # 语义ID字典
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
