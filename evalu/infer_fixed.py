#!/usr/bin/env python3
"""
修复版推理脚本
专门处理自适应相似度模块权重加载问题
"""

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


def get_ckpt_path():
    """获取训练好的模型权重文件路径"""
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if ckpt_path is None:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)


def get_args():
    """解析评估推理的命令行参数"""
    parser = argparse.ArgumentParser()

    # 基础参数
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)
    parser.add_argument('--hidden_units', default=32, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true', default=True)
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str)
    
    # 优化功能参数（自动检测）
    parser.add_argument('--use_adaptive_similarity', action='store_true')
    parser.add_argument('--similarity_types', nargs='+', default=['dot', 'cosine'])
    parser.add_argument('--use_smart_sampling', action='store_true')
    parser.add_argument('--use_curriculum_learning', action='store_true')

    return parser.parse_args()


def smart_load_model(model, checkpoint_path, device):
    """
    智能加载模型权重，自动处理结构不匹配问题
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 检查是否有自适应相似度模块
    adaptive_keys = [k for k in checkpoint.keys() if k.startswith('adaptive_similarity.')]
    
    if adaptive_keys:
        print(f"检测到自适应相似度权重: {adaptive_keys}")
        
        # 根据权重推断相似度类型
        similarity_types = ['dot']
        if any('beta' in k for k in adaptive_keys):
            similarity_types.append('cosine')
        if any('gamma' in k for k in adaptive_keys):
            similarity_types.append('scaled')
        if any('bilinear' in k for k in adaptive_keys):
            similarity_types.append('bilinear')
            
        print(f"推断的相似度类型: {similarity_types}")
        
        # 添加自适应相似度模块
        from model import AdaptiveSimilarity
        model.adaptive_similarity = AdaptiveSimilarity(
            model.hidden_units, 
            similarity_types=similarity_types
        ).to(device)
        print("已添加自适应相似度模块")
    
    # 使用strict=False加载权重，忽略不匹配的键
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    
    if missing_keys:
        print(f"模型中缺失的权重（将使用随机初始化）: {missing_keys}")
    if unexpected_keys:
        print(f"checkpoint中多余的权重（已忽略）: {unexpected_keys}")
    
    print("✅ 模型权重加载完成")
    return model


def get_candidate_emb(indexer, feat_types, feat_default_value, mm_emb_dict, model):
    """生成候选物品embedding"""
    print("正在生成候选物品embedding...")
    
    all_item_embs = []
    retrieve_id2creative_id = {}
    
    count = 0
    for item_reid, item_id in tqdm(indexer.items(), desc="Processing items"):
        if item_reid == 0:
            continue
            
        # 准备物品特征
        feat = {}
        
        # 填充默认特征
        for feat_type in feat_types['item_sparse']:
            feat[feat_type] = feat_default_value[feat_type]
        for feat_type in feat_types['item_array']:
            feat[feat_type] = feat_default_value[feat_type]
        for feat_type in feat_types['item_continual']:
            feat[feat_type] = feat_default_value[feat_type]
        
        # 多模态特征
        for feat_type in feat_types['item_emb']:
            if item_id in mm_emb_dict[feat_type]:
                feat[feat_type] = mm_emb_dict[feat_type][item_id]
            else:
                feat[feat_type] = feat_default_value[feat_type]
        
        # 生成embedding
        with torch.no_grad():
            item_emb = model.predict_item(feat)
            all_item_embs.append(item_emb.detach().cpu().numpy().astype(np.float32))
        
        retrieve_id2creative_id[count] = item_id
        count += 1
    
    # 保存候选库embedding
    all_item_embs = np.concatenate(all_item_embs, axis=0)
    candidate_path = Path(os.environ.get('EVAL_RESULT_PATH'), 'candidate.fbin')
    save_emb(all_item_embs, candidate_path)
    
    # 保存ID映射
    id_map_path = Path(os.environ.get('EVAL_RESULT_PATH'), 'retrieve_id2creative_id.json')
    with open(id_map_path, 'w') as f:
        json.dump(retrieve_id2creative_id, f)
    
    print(f"✅ 候选库embedding已保存: {all_item_embs.shape}")
    return retrieve_id2creative_id


def infer():
    """主推理函数"""
    print("🚀 开始推理...")
    
    # 参数配置
    args = get_args()
    data_path = os.environ.get('EVAL_DATA_PATH')
    
    # 数据加载
    print("📊 加载测试数据...")
    test_dataset = MyTestDataset(data_path, args)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=0, collate_fn=test_dataset.collate_fn
    )
    
    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    feat_statistics = test_dataset.feat_statistics
    feat_types = test_dataset.feature_types
    
    print(f"用户数: {usernum}, 物品数: {itemnum}")
    
    # 模型加载
    print("🤖 创建并加载模型...")
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    
    # 智能加载权重
    ckpt_path = get_ckpt_path()
    model = smart_load_model(model, ckpt_path, args.device)
    model.eval()
    
    # 生成用户query embedding
    print("👤 生成用户query embedding...")
    all_embs = []
    user_list = []
    
    for step, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        seq, token_type, seq_feat, user_id = batch
        seq = seq.to(args.device)
        
        with torch.no_grad():
            logits = model.predict(seq, seq_feat, token_type)
            
        for i in range(logits.shape[0]):
            emb = logits[i].unsqueeze(0).detach().cpu().numpy().astype(np.float32)
            all_embs.append(emb)
        user_list += user_id
    
    # 生成候选库embedding
    print("🎯 生成候选库embedding...")
    retrieve_id2creative_id = get_candidate_emb(
        test_dataset.indexer['i'],
        test_dataset.feature_types,
        test_dataset.feature_default_value,
        test_dataset.mm_emb_dict,
        model,
    )
    
    # 保存query embedding
    all_embs = np.concatenate(all_embs, axis=0)
    query_path = Path(os.environ.get('EVAL_RESULT_PATH'), 'query.fbin')
    save_emb(all_embs, query_path)
    
    print(f"✅ Query embedding已保存: {all_embs.shape}")
    print(f"✅ 推理完成，共处理 {len(user_list)} 个用户")
    
    return None, user_list


if __name__ == "__main__":
    try:
        infer()
        print("🎉 推理成功完成！")
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
