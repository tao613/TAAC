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
    解析命令行参数
    
    Returns:
        args: 包含所有训练参数的命名空间对象
    """
    parser = argparse.ArgumentParser()

    # 训练参数
    parser.add_argument('--batch_size', default=128, type=int, help='批次大小')
    parser.add_argument('--lr', default=0.001, type=float, help='学习率')
    parser.add_argument('--maxlen', default=101, type=int, help='序列最大长度')

    # 模型结构参数
    parser.add_argument('--hidden_units', default=32, type=int, help='隐藏层维度')
    parser.add_argument('--num_blocks', default=1, type=int, help='Transformer块数量')
    parser.add_argument('--num_epochs', default=3, type=int, help='训练轮数')
    parser.add_argument('--num_heads', default=1, type=int, help='多头注意力头数')
    parser.add_argument('--dropout_rate', default=0.2, type=float, help='Dropout比例')
    parser.add_argument('--l2_emb', default=0.0, type=float, help='Embedding L2正则化系数')
    parser.add_argument('--device', default='cuda', type=str, help='计算设备')
    parser.add_argument('--inference_only', action='store_true', help='仅推理模式')
    parser.add_argument('--state_dict_path', default=None, type=str, help='预训练模型路径')
    parser.add_argument('--norm_first', action='store_true', help='是否使用Pre-LayerNorm')

    # 多模态特征ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)], help='激活的多模态特征ID')
    
    # 距离计算优化选项
    parser.add_argument('--use_adaptive_similarity', action='store_true', help='是否使用自适应相似度计算')
    parser.add_argument('--similarity_types', nargs='+', default=['dot', 'cosine'], choices=['dot', 'cosine', 'scaled', 'bilinear', 'euclidean'], help='相似度计算类型')
    parser.add_argument('--use_smart_sampling', action='store_true', help='是否使用智能负采样')
    
    # 训练策略优化
    parser.add_argument('--use_curriculum_learning', action='store_true', help='是否使用课程学习')
    parser.add_argument('--curriculum_schedule', default='linear', choices=['linear', 'cosine', 'exponential'], help='课程学习难度调度策略')
    parser.add_argument('--neg_sample_ratio', default=1, type=int, help='负样本采样比例')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # 创建训练日志和TensorBoard日志目录
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    
    # 获取数据路径
    data_path = os.environ.get('TRAIN_DATA_PATH')

    # 解析命令行参数
    args = get_args()
    
    print(f"=== 训练配置 ===")
    print(f"设备: {args.device}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"序列长度: {args.maxlen}")
    print(f"隐藏维度: {args.hidden_units}")
    print(f"Pre-LayerNorm: {args.norm_first}")
    if args.use_adaptive_similarity:
        print(f"自适应相似度计算: {args.similarity_types}")
    if args.use_smart_sampling:
        print(f"智能负采样: 启用")
    if args.use_curriculum_learning:
        print(f"课程学习: 启用 ({args.curriculum_schedule})")
    print(f"===================")
    
    # 初始化数据集和数据加载器
    dataset = MyDataset(data_path, args)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn
    )
    
    # 获取数据集统计信息
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    # 初始化模型
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    
    # 配置模型优化选项
    if args.use_adaptive_similarity and hasattr(model, 'adaptive_similarity'):
        # 重新初始化自适应相似度计算器以使用指定的相似度类型
        from model import AdaptiveSimilarity
        model.adaptive_similarity = AdaptiveSimilarity(
            args.hidden_units, 
            similarity_types=args.similarity_types
        ).to(args.device)
        print(f"自适应相似度计算器已配置: {args.similarity_types}")
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 初始化模型参数
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass  # 对于一维参数（如bias），跳过xavier初始化

    # 将特殊token的embedding设置为0
    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0

    # 将稀疏特征的padding token embedding设置为0
    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    # 设置开始训练的epoch
    epoch_start_idx = 1

    # 如果指定了预训练模型路径，加载模型参数
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            # 从文件名中解析epoch信息
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6 :]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    # 使用原始的BCE损失函数
    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    global_step = 0
    
    print("开始训练")
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        # 更新数据集的epoch信息（用于课程学习）
        dataset.update_epoch(epoch)
        
        model.train()
        if args.inference_only:
            break
            
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            # 解包批次数据（原始方式：包含neg和neg_feat）
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            
            # 前向传播（原始方式）
            pos_logits, neg_logits = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
            )
            
            # 创建标签
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                neg_logits.shape, device=args.device
            )
            
            optimizer.zero_grad()
            
            # 只对item token计算损失
            indices = np.where(next_token_type == 1)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            # 记录训练日志
            log_json = json.dumps(
                {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'time': time.time()}
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            
            # 每100步显示一次详细日志
            if global_step % 100 == 0:
                print(f"Epoch {epoch}, Step {global_step}, Loss: {loss.item():.4f}")
                
                # 记录相似度权重（如果使用自适应相似度）
                if args.use_adaptive_similarity and hasattr(model, 'adaptive_similarity'):
                    alpha = model.adaptive_similarity.alpha.item()
                    beta = model.adaptive_similarity.beta.item()
                    gamma = model.adaptive_similarity.gamma.item()
                    temp = model.adaptive_similarity.temperature.item()
                    writer.add_scalar('Similarity/alpha', alpha, global_step)
                    writer.add_scalar('Similarity/beta', beta, global_step)
                    writer.add_scalar('Similarity/gamma', gamma, global_step)
                    writer.add_scalar('Similarity/temperature', temp, global_step)
                    print(f"  相似度权重 - α:{alpha:.3f}, β:{beta:.3f}, γ:{gamma:.3f}, T:{temp:.3f}")
                
                # 记录课程学习状态
                if args.use_curriculum_learning:
                    curriculum_status = dataset.get_curriculum_status()
                    if curriculum_status['enabled']:
                        difficulty = curriculum_status['difficulty_factor']
                        writer.add_scalar('Curriculum/difficulty_factor', difficulty, global_step)
                        writer.add_scalar('Curriculum/epoch_progress', epoch / args.num_epochs, global_step)
                        print(f"  课程学习难度: {difficulty:.3f} ({curriculum_status['schedule']})")

            writer.add_scalar('Loss/train', loss.item(), global_step)

            global_step += 1

            # L2正则化
            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
                
            loss.backward()
            optimizer.step()

        # 验证阶段
        model.eval()
        valid_loss_sum = 0
        for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)
            
            pos_logits, neg_logits = model(
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
            )
            
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                neg_logits.shape, device=args.device
            )
            
            indices = np.where(next_token_type == 1)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            valid_loss_sum += loss.item()
            
        valid_loss_sum /= len(valid_loader)
        writer.add_scalar('Loss/valid', valid_loss_sum, global_step)

        # 保存模型检查点
        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")
        
        # 保存训练配置
        config_dict = {
            'args': vars(args),
            'model_params': trainable_params,
            'epoch': epoch,
            'global_step': global_step,
            'valid_loss': valid_loss_sum
        }
        
        with open(save_dir / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
            
        print(f"模型已保存: {save_dir}")
        print(f"验证损失: {valid_loss_sum:.4f}")

    print("\n=== 训练完成 ===")
    print(f"总训练步数: {global_step}")
    print(f"最终验证损失: {valid_loss_sum:.4f}")
    
    # 保存最终的相似度权重信息
    if args.use_adaptive_similarity and hasattr(model, 'adaptive_similarity'):
        print("\n=== 最终相似度权重 ===")
        print(f"点积权重 (α): {model.adaptive_similarity.alpha.item():.4f}")
        print(f"余弦权重 (β): {model.adaptive_similarity.beta.item():.4f}")
        print(f"缩放权重 (γ): {model.adaptive_similarity.gamma.item():.4f}")
        print(f"温度参数 (T): {model.adaptive_similarity.temperature.item():.4f}")
    
    # 保存课程学习最终状态
    if args.use_curriculum_learning:
        curriculum_status = dataset.get_curriculum_status()
        if curriculum_status['enabled']:
            print("\n=== 课程学习最终状态 ===")
            print(f"调度策略: {curriculum_status['schedule']}")
            print(f"最终难度因子: {curriculum_status['difficulty_factor']:.4f}")
            print(f"训练进度: {curriculum_status['current_epoch']}/{curriculum_status['total_epochs']}")
    
    writer.close()
    log_file.close()
    print("日志文件已关闭")