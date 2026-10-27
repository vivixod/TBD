# 跨患者相似切片 Top4

`train_joint.sh` 默认 `SUPPORT_SAMPLING=similarity`。BTCV2、AMOS、BraTS 的 query/class
选择保持原逻辑；support 从当前 split 的同类别阳性切片中，排除 query 整位患者后，按图像特征
余弦相似度降序取4张不同切片。4张可以属于同一位其他患者，没有患者配额或单患者预选。
PAPILA 保持原来的4位异患者、每人一只眼及固定验证组合。

当前默认训练 100 epochs，初始主学习率保持 `1e-4`，cosine 调度按 100 轮总步数
衰减到 0。SAM 参数组初始学习率保持 `1e-5`；若解冻 layer3，其初始学习率仍为
`1e-6`，各组使用相同的 cosine 衰减比例。可用 `EPOCHS` 环境变量覆盖轮数，默认
日志名带 `_ep100`。已启动的旧进程不自动更改轮数或学习率日程。

相似度采用固定预训练 ResNet50 的 layer2/layer3 特征，各自做1×1和2×2平均池化、分别L2
归一化后拼接再归一化（7680维）。输入为未增强RGB图像缩放至224，采用各数据集已有的
归一化常量。检索编码器始终eval、无需梯度，不随训练模型微调而更新。
检索不读取任何 query mask、GT框、预测Dice，也不运行SAM。support类别候选池沿用现有有标注元数据。

所有 query/class 的排序在训练前一次计算，训练与valid仅查询索引，不在DataLoader中运行编码器
或反复计算全池相似度。训练加载选中图像后照常独立增强和在线编码，检索缓存不替代SAM特征。
同一个query/class的Top4在训练和验证对应split内均固定；随机性仍来自query/class采样和训练增强。
分数相同时按切片名排序；候选不足shot时明确报错，不使用自身、不重复凑数。

## 启动

保持当前冻结layer3、quality融合和两处attention屏蔽：

```bash
GPU_ID=1,2 LAYER3_MODE=frozen SUPPORT_SAMPLING=similarity \
SUPPORT_MASK_POOLING=max BBOX_SUPPORT_ATTN_MASK=1 ICL_SUPPORT_ATTN_MASK=1 \
SUPPORT_FUSION=quality bash train_joint.sh
```

脚本先在`GPU_ID`中的第一张卡准备三个体数据集train/val的检索索引，再启动原来的DDP训练。
默认`SUPPORT_RETRIEVAL_DEVICE=cuda:0`是经过`CUDA_VISIBLE_DEVICES=1,2`映射后的物理GPU1。
首次启动增加一次特征提取和全池排序耗时；后续配置相同会复用，具体时间需在本机完整预计算后测量。
新日志名带`_supportsimilarity`，日志及checkpoint配置记录采样模式和缓存目录，验证频率不变。

也可先单独准备（例如复制cache到另一台服务器后复用）：

```bash
CUDA_VISIBLE_DEVICES=1 python prepare_support_retrieval.py \
  --joint-subdatasets 'btcv2:/data/2025Zhutaotao/BTCV,amos:/data/2025Zhutaotao/AMOS2022_MR,brats:/data/2025Zhutaotao/BraTS2019' \
  --device cuda:0 --output-dir cache_support_retrieval
```

CPU预计算使用`--device cpu`，或启动训练时设置`SUPPORT_RETRIEVAL_DEVICE=cpu`。
`--max-shot`默认4；更高shot需重新准备足够长的排名。缓存写入是原子替换，记录算法版本、
split/类别/候选池/归一化摘要和编码器权重SHA256。准备脚本自动重建不匹配的缓存；
直接调用train时缓存缺失或不匹配会报错。原图像内容被替换后需加`--force`重新准备。

## 对照设置

| 环境变量 | 规则 |
|---|---|
| `SUPPORT_SAMPLING=similarity` | 排除query患者，全池图像相似度Top4 |
| `SUPPORT_SAMPLING=cross-patient-random` | 同一候选约束，均匀随机4张，不需要检索缓存 |
| `SUPPORT_SAMPLING=random NEIGHBOR_PROB=0` | 保留旧nb0随机规则（可能同患者/自身） |
| `SUPPORT_SAMPLING=random NEIGHBOR_PROB=1` | 保留旧近邻规则 |

前两种模式强制neighbor_prob=0；PAPILA在所有模式都保持原采样规则。Python训练接口默认
`--support-sampling random`保持旧调用兼容，shell入口默认similarity以启用新实验。
已有独立test和support评分诊断会读取保存的采样配置；换机器可显式指定新的缓存目录。

这是从较大的有标注候选池中检索4张，候选池标注预算需与baseline一致。候选筛选不跨train/val。
检索改变support抽样的随机数消费，旧随机模式与新模式的query/class序列可能不同；
严格配对对比应固定同一份query/class清单。首次建议同一checkpoint比较随机和检索，收益尚待实验。
