import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MyDataset
from model import BaselineModel


def get_warmup_cosine_lr(step, warmup_steps, total_steps, base_lr, min_lr_ratio=0.01):
    """
    计算Warmup + Cosine Annealing学习率调度
    
    Args:
        step: 当前训练步数
        warmup_steps: 预热步数
        total_steps: 总训练步数
        base_lr: 基础学习率
        min_lr_ratio: 最小学习率与基础学习率的比例
        
    Returns:
        lr: 当前步数对应的学习率
    """
    if step < warmup_steps:
        # Warmup阶段：学习率从0线性增长到base_lr
        lr = base_lr * step / warmup_steps
    else:
        # Cosine Annealing阶段：学习率按余弦函数衰减
        cosine_steps = total_steps - warmup_steps
        cosine_step = step - warmup_steps
        min_lr = base_lr * min_lr_ratio
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * cosine_step / cosine_steps))
    
    return lr


def get_args():
    """
    解析命令行参数，配置模型训练的各种超参数
    
    Returns:
        args: 包含所有训练参数的命名空间对象
    """
    parser = argparse.ArgumentParser()

    # 训练参数
    parser.add_argument('--batch_size', default=128, type=int, help='训练批次大小')
    parser.add_argument('--lr', default=0.001, type=float, help='学习率')
    parser.add_argument('--maxlen', default=101, type=int, help='用户序列最大长度')

    # Baseline 模型构建参数
    parser.add_argument('--hidden_units', default=32, type=int, help='隐藏层维度')
    parser.add_argument('--num_blocks', default=1, type=int, help='Transformer块数量')
    parser.add_argument('--num_epochs', default=3, type=int, help='训练轮数')
    parser.add_argument('--num_heads', default=1, type=int, help='多头注意力头数')
    parser.add_argument('--dropout_rate', default=0.2, type=float, help='Dropout概率')
    parser.add_argument('--l2_emb', default=0.0, type=float, help='Embedding L2正则化权重')
    parser.add_argument('--device', default='cuda', type=str, help='训练设备')
    parser.add_argument('--inference_only', action='store_true', help='仅推理模式，不训练')
    parser.add_argument('--state_dict_path', default=None, type=str, help='预训练模型权重路径')
    parser.add_argument('--norm_first', action='store_true', default=True, 
                       help='是否在注意力层前先进行LayerNorm（Pre-LayerNorm结构，默认启用以提升训练稳定性）')

    # ==================== 学习率调度器配置 ====================
    parser.add_argument('--use_lr_scheduler', action='store_true', default=True,
                       help='是否使用学习率调度器（Warmup + Cosine Annealing，默认启用）')
    parser.add_argument('--warmup_steps', default=100, type=int,
                       help='学习率预热步数，在此期间学习率从0线性增长到设定值')
    parser.add_argument('--min_lr_ratio', default=0.01, type=float,
                       help='最小学习率与初始学习率的比例（用于Cosine Annealing）')

    # ==================== 早停机制配置 ====================
    parser.add_argument('--early_stopping', action='store_true', default=True,
                       help='是否启用早停机制（默认启用，防止过拟合）')
    parser.add_argument('--patience', default=5, type=int,
                       help='早停耐心值：验证损失连续多少个epoch不改善后停止训练')
    parser.add_argument('--min_delta', default=1e-4, type=float,
                       help='最小改善阈值：验证损失改善小于此值时认为没有改善')

    # ==================== 优化策略配置 ====================
    parser.add_argument('--use_inbatch_negatives', action='store_true', default=True,
                       help='是否使用In-batch Negatives策略（默认启用，与InfoNCE Loss配合）')
    parser.add_argument('--use_infonce_loss', action='store_true', default=True,
                       help='是否使用InfoNCE Loss（默认启用，对比学习损失函数）')
    parser.add_argument('--temperature', default=0.07, type=float,
                       help='InfoNCE Loss的温度参数，控制分布的尖锐程度')
    parser.add_argument('--max_negatives_per_query', default=100, type=int,
                       help='每个查询的最大负样本数量（用于内存优化）')

    # 多模态特征ID配置
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, 
                       choices=[str(s) for s in range(81, 87)], 
                       help='激活的多模态特征ID列表，81-86对应不同类型的多模态特征')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # ==================== 初始化训练环境 ====================
    # 创建日志和模型保存目录
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    
    # 初始化日志文件和TensorBoard记录器
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    
    # 获取训练数据路径
    data_path = os.environ.get('TRAIN_DATA_PATH')

    # ==================== 数据加载和预处理 ====================
    args = get_args()
    
    # 创建数据集，包含用户序列、物品特征、多模态特征等
    dataset = MyDataset(data_path, args)
    
    # 按9:1比例划分训练集和验证集
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    
    # 创建数据加载器，使用自定义的collate_fn处理变长序列
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn
    )
    
    # 获取用户和物品数量，以及特征统计信息
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    # ==================== 模型初始化 ====================
    # 创建基础推荐模型，包含用户/物品Embedding、多模态特征处理、Transformer等
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

    # 使用Xavier正态分布初始化模型参数
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass  # 跳过无法初始化的参数（如1维张量）

    # 将padding位置的embedding初始化为0
    model.pos_emb.weight.data[0, :] = 0      # 位置embedding的padding位
    model.item_emb.weight.data[0, :] = 0     # 物品embedding的padding位  
    model.user_emb.weight.data[0, :] = 0     # 用户embedding的padding位

    # 将稀疏特征embedding的padding位初始化为0
    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    # ==================== 模型加载（如果指定了预训练权重）====================
    epoch_start_idx = 1  # 开始训练的轮次

    if args.state_dict_path is not None:
        try:
            # 加载预训练模型权重
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            
            # 从文件名中解析上次训练结束的epoch，继续训练
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6 :]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    # ==================== 优化器和损失函数初始化 ====================
    # 使用InfoNCE损失，适用于对比学习任务
    def info_nce_loss(query, pos_key, neg_keys, temperature=0.07):
        """
        InfoNCE损失函数实现
        
        Args:
            query: 查询向量 [N, D]
            pos_key: 正样本向量 [N, D] 
            neg_keys: 负样本向量 [N, K, D] 其中K是负样本数量
            temperature: 温度参数，控制分布的尖锐程度
            
        Returns:
            loss: InfoNCE损失
        """
        # 计算正样本相似度 [N]
        pos_sim = torch.sum(query * pos_key, dim=-1) / temperature
        
        # 计算负样本相似度 [N, K]
        neg_sim = torch.bmm(neg_keys, query.unsqueeze(-1)).squeeze(-1) / temperature
        
        # 拼接正负样本相似度 [N, K+1]
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        
        # 正样本的标签总是0（第一个位置）
        labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.long)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)
        return loss
    
    # 使用Adam优化器，beta参数调整为(0.9, 0.98)以提高训练稳定性
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    # ==================== 学习率调度器初始化 ====================
    if args.use_lr_scheduler:
        # 计算总训练步数（用于Cosine Annealing）
        total_steps = len(train_loader) * args.num_epochs
        print(f"学习率调度器已启用: Warmup({args.warmup_steps} steps) + Cosine Annealing")
        print(f"总训练步数: {total_steps}, 基础学习率: {args.lr}, 最小学习率比例: {args.min_lr_ratio}")
    else:
        print("使用固定学习率:", args.lr)

    # 记录最佳验证和测试指标
    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    global_step = 0
    
    # ==================== 早停机制变量初始化 ====================
    if args.early_stopping:
        best_val_loss = float('inf')        # 最佳验证损失
        patience_counter = 0                # 当前耐心计数器
        early_stop_flag = False             # 早停标志
        print(f"早停机制已启用: patience={args.patience}, min_delta={args.min_delta}")
    else:
        print("早停机制已禁用")
    
    print("Start training")
    
    # ==================== 主训练循环 ====================
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        # ==================== 早停检查 ====================
        if args.early_stopping and 'early_stop_flag' in locals() and early_stop_flag:
            print(f"早停标志已设置，跳过第 {epoch} 个epoch")
            break
            
        model.train()  # 设置为训练模式
        
        # 如果是仅推理模式，跳过训练
        if args.inference_only:
            break
            
        # -------------------- 训练阶段 --------------------
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            # 解包批次数据：序列、正样本、负样本、token类型、动作类型、特征等
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            
            # 将数据移到指定设备（GPU/CPU）
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # ==================== 选择损失函数和负采样策略 ====================
            if args.use_inbatch_negatives and args.use_infonce_loss:
                # 使用In-batch Negatives + InfoNCE Loss
                query_embs, pos_embs_flat, neg_embs, valid_positions = model.forward_inbatch_negatives(
                    seq, pos, token_type, next_token_type, seq_feat, pos_feat, args.max_negatives_per_query
                )
                
                # 如果没有有效位置，跳过这个批次
                if query_embs is None:
                    continue
                    
                # 计算InfoNCE损失
                loss = info_nce_loss(query_embs, pos_embs_flat, neg_embs, temperature=args.temperature)
            else:
                # 使用传统的随机负采样 + BCE Loss
                neg = neg.to(args.device)
                
                # 模型前向传播，计算正样本和负样本的logits
                pos_logits, neg_logits = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                )
                
                # 创建标签：正样本标签为1，负样本标签为0
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                    neg_logits.shape, device=args.device
                )
                
                # 只对item token计算损失（next_token_type == 1表示下一个token是item）
                indices = np.where(next_token_type == 1)
                bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            # 添加item embedding的L2正则化
            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
                
            # 反向传播和参数更新
            loss.backward()
            optimizer.step()
            
            # 记录训练日志
            log_json = json.dumps(
                {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'time': time.time()}
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)

            # 记录到TensorBoard
            writer.add_scalar('Loss/train', loss.item(), global_step)

            global_step += 1
            
            # ==================== 学习率调度 ====================
            if args.use_lr_scheduler:
                # 计算并应用新的学习率
                current_lr = get_warmup_cosine_lr(
                    global_step, args.warmup_steps, total_steps, args.lr, args.min_lr_ratio
                )
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                
                # 记录学习率到TensorBoard
                writer.add_scalar('Learning_Rate', current_lr, global_step)

        # -------------------- 验证阶段 --------------------
        model.eval()  # 设置为评估模式
        valid_loss_sum = 0
        valid_batch_count = 0
        
        # 不计算梯度，节省内存和计算资源
        with torch.no_grad():
            for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
                seq = seq.to(args.device)
                pos = pos.to(args.device)
                
                # ==================== 选择损失函数和负采样策略（验证） ====================
                if args.use_inbatch_negatives and args.use_infonce_loss:
                    # 使用In-batch Negatives + InfoNCE Loss
                    query_embs, pos_embs_flat, neg_embs, valid_positions = model.forward_inbatch_negatives(
                        seq, pos, token_type, next_token_type, seq_feat, pos_feat, args.max_negatives_per_query
                    )
                    
                    # 如果没有有效位置，跳过这个批次
                    if query_embs is None:
                        continue
                        
                    # 计算InfoNCE损失
                    loss = info_nce_loss(query_embs, pos_embs_flat, neg_embs, temperature=args.temperature)
                else:
                    # 使用传统的随机负采样 + BCE Loss
                    neg = neg.to(args.device)
                    
                    # 计算验证损失
                    pos_logits, neg_logits = model(
                        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                    )
                    pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                        neg_logits.shape, device=args.device
                    )
                    
                    # 只对item token计算验证损失
                    indices = np.where(next_token_type == 1)
                    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
                    loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                    loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                valid_loss_sum += loss.item()
                valid_batch_count += 1
                
        # 计算平均验证损失
        if valid_batch_count > 0:
            valid_loss_sum /= valid_batch_count
        else:
            valid_loss_sum = float('inf')  # 如果没有有效批次，设置为无穷大
        writer.add_scalar('Loss/valid', valid_loss_sum, global_step)

        # ==================== 早停检查 ====================
        save_best_model = False  # 是否保存最佳模型
        
        if args.early_stopping:
            # 检查验证损失是否有改善
            if valid_loss_sum < best_val_loss - args.min_delta:
                # 验证损失有改善
                best_val_loss = valid_loss_sum
                patience_counter = 0
                save_best_model = True
                print(f"验证损失改善至 {valid_loss_sum:.6f}，重置早停计数器")
            else:
                # 验证损失没有改善
                patience_counter += 1
                print(f"验证损失未改善 ({valid_loss_sum:.6f} vs {best_val_loss:.6f})，早停计数器: {patience_counter}/{args.patience}")
                
                # 检查是否达到早停条件
                if patience_counter >= args.patience:
                    early_stop_flag = True
                    print(f"触发早停机制！连续 {args.patience} 个epoch验证损失未改善，停止训练")
        else:
            # 如果不使用早停，则每个epoch都保存模型
            save_best_model = True

        # -------------------- 模型保存 --------------------
        if save_best_model:
            # 创建保存目录，包含全局步数和验证损失信息
            save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存模型权重
            torch.save(model.state_dict(), save_dir / "model.pt")
            
            if args.early_stopping:
                print(f"保存最佳模型至 {save_dir}")
        
        # ==================== 早停条件检查 ====================
        if args.early_stopping and early_stop_flag:
            print(f"早停机制生效，在第 {epoch} 个epoch停止训练")
            break

    print("Done")
    writer.close()
    log_file.close()
