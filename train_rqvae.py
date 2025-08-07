#!/usr/bin/env python3
"""
RQ-VAE训练脚本：生成物品的semantic_id特征

这个脚本使用物品的多模态embedding训练RQ-VAE模型，
然后为所有物品生成对应的semantic_id序列，
作为新的特征输入到主推荐模型中。

使用方法：
python train_rqvae.py --data_dir <data_path> --output_dir <output_path>
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model_rqvae import RQVAE, kmeans, BalancedKmeans


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='RQ-VAE训练脚本生成semantic_id')
    
    # 数据路径参数
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录路径')
    
    # RQ-VAE模型参数
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, 
                       choices=[str(s) for s in range(81, 87)],
                       help='用于训练RQ-VAE的多模态特征ID列表')
    parser.add_argument('--num_quantizers', default=4, type=int, help='RQ量化器数量')
    parser.add_argument('--codebook_size', default=256, type=int, help='每个codebook的大小')
    parser.add_argument('--commitment_cost', default=0.25, type=float, help='commitment损失权重')
    
    # 训练参数
    parser.add_argument('--batch_size', default=1024, type=int, help='训练批次大小')
    parser.add_argument('--num_epochs', default=50, type=int, help='训练轮数')
    parser.add_argument('--lr', default=1e-3, type=float, help='学习率')
    parser.add_argument('--device', default='cuda', type=str, help='训练设备')
    
    return parser.parse_args()


def load_mm_emb_for_rqvae(mm_path, feat_ids):
    """
    加载多模态特征用于RQ-VAE训练
    
    Args:
        mm_path: 多模态特征路径
        feat_ids: 特征ID列表
        
    Returns:
        item_embeddings: 字典 {item_id: embedding_vector}
        embedding_dim: embedding维度
    """
    from dataset import load_mm_emb
    
    print("加载多模态特征用于RQ-VAE训练...")
    mm_emb_dict = load_mm_emb(mm_path, feat_ids)
    
    # 合并所有特征ID的embeddings
    item_embeddings = {}
    embedding_dim = 0
    
    for feat_id in feat_ids:
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
            # 多个特征取平均
            final_embeddings[item_id] = np.mean(emb_list, axis=0)
    
    print(f"加载了 {len(final_embeddings)} 个物品的多模态特征，维度: {embedding_dim}")
    return final_embeddings, embedding_dim


def prepare_training_data(item_embeddings, embedding_dim):
    """
    准备RQ-VAE训练数据
    
    Args:
        item_embeddings: 物品embeddings字典
        embedding_dim: embedding维度
        
    Returns:
        embeddings: 训练用的embedding矩阵 [num_items, embedding_dim]
        item_ids: 对应的物品ID列表
    """
    print("准备RQ-VAE训练数据...")
    
    item_ids = list(item_embeddings.keys())
    embeddings = np.zeros((len(item_ids), embedding_dim), dtype=np.float32)
    
    for i, item_id in enumerate(item_ids):
        embeddings[i] = item_embeddings[item_id]
    
    print(f"准备了 {len(item_ids)} 个物品的训练数据")
    return embeddings, item_ids


def train_rqvae(embeddings, args):
    """
    训练RQ-VAE模型
    
    Args:
        embeddings: 训练数据 [num_items, embedding_dim]
        args: 训练参数
        
    Returns:
        rqvae_model: 训练好的RQ-VAE模型
    """
    print("开始训练RQ-VAE模型...")
    
    # 数据准备
    embeddings = torch.from_numpy(embeddings).to(args.device)
    num_items, embedding_dim = embeddings.shape
    
    # 初始化RQ-VAE模型
    rqvae_model = RQVAE(
        num_quantizers=args.num_quantizers,
        codebook_size=args.codebook_size,
        embedding_dim=embedding_dim,
        commitment_cost=args.commitment_cost
    ).to(args.device)
    
    # 优化器
    optimizer = torch.optim.Adam(rqvae_model.parameters(), lr=args.lr)
    
    # 训练循环
    rqvae_model.train()
    
    for epoch in range(args.num_epochs):
        total_loss = 0
        total_recon_loss = 0
        total_commitment_loss = 0
        
        # 创建批次
        num_batches = (num_items + args.batch_size - 1) // args.batch_size
        
        for batch_idx in tqdm(range(num_batches), desc=f'Epoch {epoch+1}/{args.num_epochs}'):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, num_items)
            
            batch_embeddings = embeddings[start_idx:end_idx]
            
            # 前向传播
            reconstructed, commitment_loss, semantic_ids = rqvae_model(batch_embeddings)
            
            # 计算重构损失
            recon_loss = F.mse_loss(reconstructed, batch_embeddings)
            
            # 总损失
            loss = recon_loss + commitment_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_commitment_loss += commitment_loss.item()
        
        # 输出训练状态
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_commitment_loss = total_commitment_loss / num_batches
        
        print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, Recon={avg_recon_loss:.4f}, Commit={avg_commitment_loss:.4f}')
    
    print("RQ-VAE训练完成！")
    return rqvae_model


def generate_semantic_ids(rqvae_model, embeddings, item_ids, args):
    """
    使用训练好的RQ-VAE生成semantic_id
    
    Args:
        rqvae_model: 训练好的RQ-VAE模型
        embeddings: 物品embeddings
        item_ids: 物品ID列表
        args: 参数
        
    Returns:
        semantic_id_dict: 字典 {item_id: semantic_id_sequence}
    """
    print("生成物品的semantic_id...")
    
    rqvae_model.eval()
    semantic_id_dict = {}
    
    embeddings = torch.from_numpy(embeddings).to(args.device)
    num_items = embeddings.shape[0]
    
    with torch.no_grad():
        # 分批处理
        num_batches = (num_items + args.batch_size - 1) // args.batch_size
        
        for batch_idx in tqdm(range(num_batches), desc='生成semantic_id'):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, num_items)
            
            batch_embeddings = embeddings[start_idx:end_idx]
            batch_item_ids = item_ids[start_idx:end_idx]
            
            # 获取semantic_id
            _, _, semantic_ids = rqvae_model(batch_embeddings)
            
            # semantic_ids: [batch_size, num_quantizers]
            semantic_ids = semantic_ids.cpu().numpy()
            
            # 保存结果
            for i, item_id in enumerate(batch_item_ids):
                semantic_id_dict[item_id] = semantic_ids[i].tolist()
    
    print(f"为 {len(semantic_id_dict)} 个物品生成了semantic_id")
    return semantic_id_dict


def save_semantic_ids(semantic_id_dict, output_dir):
    """
    保存semantic_id到文件
    
    Args:
        semantic_id_dict: semantic_id字典
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存为JSON文件，方便dataset.py加载
    semantic_id_file = output_dir / 'semantic_id_dict.json'
    
    print(f"保存semantic_id到 {semantic_id_file}")
    with open(semantic_id_file, 'w') as f:
        json.dump(semantic_id_dict, f, indent=2)
    
    print("semantic_id保存完成！")


def main():
    """主函数"""
    args = get_args()
    
    print("="*50)
    print("RQ-VAE训练脚本启动")
    print("="*50)
    
    # 1. 加载多模态特征
    mm_path = Path(args.data_dir) / "creative_emb"
    item_embeddings, embedding_dim = load_mm_emb_for_rqvae(mm_path, args.mm_emb_id)
    
    # 2. 准备训练数据
    embeddings, item_ids = prepare_training_data(item_embeddings, embedding_dim)
    
    # 3. 训练RQ-VAE模型
    rqvae_model = train_rqvae(embeddings, args)
    
    # 4. 生成semantic_id
    semantic_id_dict = generate_semantic_ids(rqvae_model, embeddings, item_ids, args)
    
    # 5. 保存结果
    save_semantic_ids(semantic_id_dict, args.output_dir)
    
    # 6. 保存训练好的模型
    model_path = Path(args.output_dir) / 'rqvae_model.pt'
    torch.save(rqvae_model.state_dict(), model_path)
    print(f"RQ-VAE模型保存到 {model_path}")
    
    print("="*50)
    print("RQ-VAE训练和semantic_id生成完成！")
    print(f"结果保存在: {args.output_dir}")
    print("="*50)


if __name__ == '__main__':
    main()
