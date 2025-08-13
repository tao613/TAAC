# InfoNCE Loss 集成总结

## 🎯 修改概述

已成功将InfoNCE loss集成到basemodel中，替换原有的BCE loss。这包括了对训练和推理代码的完整修改。

## 📋 修改的文件

### 1. `model.py`
- **新增温度参数**: `self.temp = getattr(args, 'infonce_temperature', 0.2)`
- **新增方法**: `compute_infonce_loss()` - 核心InfoNCE损失计算
- **新增方法**: `forward_infonce()` - InfoNCE版本的前向传播
- **保持兼容**: 原有的`forward()`方法保持不变，用于BCE loss

### 2. `main.py`
- **新增参数**: 
  - `--use_infonce`: 启用InfoNCE loss的开关
  - `--infonce_temperature`: InfoNCE温度参数 (默认0.2)
- **训练逻辑修改**: 根据`--use_infonce`标志选择使用InfoNCE或BCE loss
- **验证逻辑修改**: 验证过程同样支持两种loss类型

### 3. `eval/infer.py`
- **新增参数**: 添加InfoNCE相关参数以保证模型兼容性
- **参数同步**: 确保推理时的模型结构与训练时一致

### 4. `run.sh`
- **默认使用InfoNCE**: 启用`--use_infonce`和`--infonce_temperature 0.2`
- **提供对比选项**: 保留BCE loss版本的注释命令

## 🚀 使用方法

### 训练 (InfoNCE Loss)
```bash
cd basemodel
./run.sh
```

或者手动运行：
```bash
python -u main.py \
    --batch_size 256 \
    --lr 0.0007 \
    --num_epochs 4 \
    --dropout_rate 0.2 \
    --hidden_units 64 \
    --num_blocks 2 \
    --num_heads 2 \
    --norm_first \
    --maxlen 101 \
    --l2_emb 1e-6 \
    --mm_emb_id 81 \
    --use_infonce \
    --infonce_temperature 0.2
```

### 推理
```bash
cd basemodel
python -u eval/infer.py
```

推理会自动使用与训练匹配的参数配置。

## 🔧 技术细节

### InfoNCE Loss 实现
- **L2归一化**: 对序列表示、正样本和负样本进行L2归一化
- **余弦相似度**: 使用余弦相似度计算正样本logits
- **矩阵乘法**: 高效计算负样本logits
- **温度缩放**: 通过温度参数控制softmax分布的锐度
- **交叉熵**: 最终使用交叉熵损失，正样本标签为0

### 关键参数
- **温度参数**: `0.2` (可调整，较小值使分布更锐化)
- **损失掩码**: 只对item token计算损失
- **TensorBoard**: 记录正样本和负样本logits的统计信息

## 📊 预期效果

InfoNCE loss相比BCE loss的优势：
1. **更好的表示学习**: 通过对比学习提升嵌入质量
2. **更稳定的训练**: 归一化操作提升训练稳定性
3. **更强的泛化能力**: 对比学习机制增强模型泛化
4. **更好的负样本利用**: 充分利用batch内的负样本信息

## ⚠️ 注意事项

1. **参数匹配**: 训练和推理时必须使用相同的模型结构参数
2. **温度调优**: 可以尝试不同的温度值 (0.1-0.5) 来优化性能
3. **学习率**: InfoNCE可能需要稍微调整学习率，当前使用0.0007
4. **兼容性**: 代码完全向后兼容，可以通过去掉`--use_infonce`回到BCE loss

## 🎯 下一步建议

1. **运行基线**: 先用InfoNCE训练一个模型测试性能
2. **参数调优**: 如果效果好，可以尝试调整温度参数
3. **对比实验**: 可以同时训练BCE和InfoNCE版本进行对比
4. **性能监控**: 关注TensorBoard中的loss曲线和logits统计

---

所有修改已完成，现在可以直接使用InfoNCE loss进行训练！
