import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MyDataset
from model import BaselineModel


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
    parser.add_argument('--norm_first', action='store_true', help='是否在注意力层前先进行LayerNorm')

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
    # 使用二元交叉熵损失，适用于点击率预测任务
    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    
    # 使用Adam优化器，beta参数调整为(0.9, 0.98)以提高训练稳定性
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    # 记录最佳验证和测试指标
    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    global_step = 0
    
    print("Start training")
    
    # ==================== 主训练循环 ====================
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
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
            neg = neg.to(args.device)
            
            # 模型前向传播，计算正样本和负样本的logits
            pos_logits, neg_logits = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
            )
            
            # 创建标签：正样本标签为1，负样本标签为0
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                neg_logits.shape, device=args.device
            )
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 只对item token计算损失（next_token_type == 1表示下一个token是item）
            indices = np.where(next_token_type == 1)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])

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

            # 添加item embedding的L2正则化
            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
                
            # 反向传播和参数更新
            loss.backward()
            optimizer.step()

        # -------------------- 验证阶段 --------------------
        model.eval()  # 设置为评估模式
        valid_loss_sum = 0
        
        # 不计算梯度，节省内存和计算资源
        with torch.no_grad():
            for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
                seq = seq.to(args.device)
                pos = pos.to(args.device)
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
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                valid_loss_sum += loss.item()
                
        # 计算平均验证损失
        valid_loss_sum /= len(valid_loader)
        writer.add_scalar('Loss/valid', valid_loss_sum, global_step)

        # -------------------- 模型保存 --------------------
        # 创建保存目录，包含全局步数和验证损失信息
        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型权重
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()
