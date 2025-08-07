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


def infonce_loss(logits, loss_mask, temperature=0.1):
    """
    InfoNCE损失函数实现，适用于In-batch Negatives策略
    
    Args:
        logits: 相似度矩阵 [batch_size, maxlen, batch_size]
        loss_mask: 损失掩码 [batch_size, maxlen]，标记哪些位置计算损失
        temperature: 温度参数，用于调节softmax的尖锐度
        
    Returns:
        loss: InfoNCE损失值
    """
    batch_size, maxlen, _ = logits.shape
    
    # 应用温度缩放
    logits = logits / temperature
    
    # 创建标签：对角线为正样本，其余为负样本
    # labels[i, j] = i，表示第i个序列的正样本是第i个候选
    labels = torch.arange(batch_size, device=logits.device).unsqueeze(0).expand(maxlen, -1).T  # [batch_size, maxlen]
    
    # 只计算loss_mask为True的位置的损失
    total_loss = 0
    valid_positions = 0
    
    for i in range(batch_size):
        for j in range(maxlen):
            if loss_mask[i, j]:  # 只对item token计算损失
                # 取第i个序列第j个位置对应的logits
                seq_logits = logits[i, j, :]  # [batch_size]
                target = labels[i, j]  # 正样本标签
                
                # 计算cross entropy loss
                loss_val = torch.nn.functional.cross_entropy(seq_logits.unsqueeze(0), target.unsqueeze(0))
                total_loss += loss_val
                valid_positions += 1
    
    if valid_positions > 0:
        return total_loss / valid_positions
    else:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=32, type=int)
    parser.add_argument('--num_blocks', default=1, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', action='store_true')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    # global dataset
    data_path = os.environ.get('TRAIN_DATA_PATH')

    args = get_args()
    dataset = MyDataset(data_path, args)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass

    model.pos_emb.weight.data[0, :] = 0
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0

    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    epoch_start_idx = 1

    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6 :]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    # InfoNCE损失函数用于In-batch Negatives策略
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    global_step = 0
    print("Start training")
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            seq, pos, token_type, next_token_type, next_action_type, seq_feat, pos_feat = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            
            # 使用In-batch Negatives策略
            logits, loss_mask = model(
                seq, pos, token_type, next_token_type, next_action_type, seq_feat, pos_feat
            )
            
            optimizer.zero_grad()
            
            # 使用InfoNCE损失函数
            loss = infonce_loss(logits, loss_mask, temperature=0.1)

            log_json = json.dumps(
                {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'time': time.time()}
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)

            writer.add_scalar('Loss/train', loss.item(), global_step)

            global_step += 1

            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
            loss.backward()
            optimizer.step()

        model.eval()
        valid_loss_sum = 0
        with torch.no_grad():
            for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                seq, pos, token_type, next_token_type, next_action_type, seq_feat, pos_feat = batch
                seq = seq.to(args.device)
                pos = pos.to(args.device)
                
                # 使用In-batch Negatives策略进行验证
                logits, loss_mask = model(
                    seq, pos, token_type, next_token_type, next_action_type, seq_feat, pos_feat
                )
                
                # 使用InfoNCE损失函数计算验证损失
                loss = infonce_loss(logits, loss_mask, temperature=0.1)
                valid_loss_sum += loss.item()
        valid_loss_sum /= len(valid_loader)
        writer.add_scalar('Loss/valid', valid_loss_sum, global_step)

        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.valid_loss={valid_loss_sum:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()
