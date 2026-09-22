# 第二版元模型：对角 Q/K/V 与显式关系缩放

本版本实现用户确定的结构：逐通道缩放得到 Q、K；先计算节点相似度，再按结构类型逐元素缩放；经过图 mask 和 Softmax 后聚合缩放后的 V；最后进行残差与 MLP 更新。核心代码为 `src/matrix_dsl/metamodel/diagonal.py`，第一版 `model.py` 保持原样。

## 模型契约

- 输入：现有 `EncodedGraph`，包含节点特征、祖先可达 mask 和依赖距离。
- 单头；每层 `wq / wk / wv` 各为 width 长度的可训练向量，初始为 1。没有完整的 Q/K/V 投影矩阵，也没有 attention output projection。
- 节点对类型：自身、直接前驱、其他祖先。每层只学习三个共享的正值缩放系数，通过 softplus 参数化，初始为 1。
- `pair_scale` 按类型展开为 `[B,N,N]`；不按节点编号存储权重，因此支持不同节点数及重新编号。
- `similarity = QKᵀ / sqrt(width)`；`weighted_similarity = similarity * pair_scale`；先 mask，再按 source 维度 Softmax。
- `H = H + relation @ V`，再做 `H = H + MLP(LayerNorm(H))`。
- 2 层、width 32。输入投影、mean/scaled-sum pooling、任务 embedding、图级 readout 与第一版保持相同，主要比较关系 block 的变化。
- 无额外可训练的节点编号 embedding。节点特征原有的计算深度和 scope 深度继续保留。
- 输出为标准化 score 与完成 logit。两项任务分别拟合预测器；不把这次实验称为跨任务迁移。

`return_relations=True` 返回每层的原始相似度、类型缩放、缩放后分数、实际聚合矩阵、V、聚合消息、通道缩放及 `wq*wk`。矩阵约定为「行接收、列发送」。没有值投影偏置。

分析时注意：独立 Q/K 通道缩放只通过乘积影响点积，不能唯一识别两者各自作用；对称的原始匹配可以被有向 mask/逐行归一化变成不对称的路由矩阵。关系缩放不是概率，对负分数放大会使它更负。注意力权重和消息不是对任务 loss 的因果贡献。

## 冻结的两任务实验

每个任务生成 128 个新随机架构，按 body family 划分 meta-train / meta-validation / candidate，训练种子为 11、23。Adding 的 body family 排除全部 v1 家族；因此不会把已检查过的 v1 候选作为本轮候选。Adding 数据实例沿用 v1，必须视为相同固定任务上的后续架构探索。

所有可训练任务模型统一 width 64、batch 256、400 epochs、每 epoch 1024 个样本（共 1600 次更新），AdamW lr 0.001、weight decay 0、clip 1、FP32、确定性 CUDA。参数量仍不同，单独记录。验证集选择最优标准化 MSE，包括 epoch 0。发散也记入语料和查询预算，不能事后过滤。

Adding 采用已归档的逐位一致数据张量：1024/256/256 个样本，长度 48，两个通道，预测两个标记值的平均值；保留 v1 的七个人工基线。

Jena 采用 Salesforce/GiftEval 的 `jena_weather/10T`，固定版本 `30841734ac5cfddbd0c3bad6d09d2b6b32becbb0`：52704×21，96 步输入预测 48 步。原始 Arrow 和整理后的数组均验证参考 SHA256。189 个缺失值只用训练历史内的线性插值补齐。均值/总体标准差只拟合 `[0,51696)`。

Jena 的 51553 个合法训练窗口按有放回方式采样；验证只有一个 48 步窗口，测试有 20 个目标互不重叠的滚动窗口。后续测试输入可以使用当时已观测到的前段历史；不是递归预测。单一验证窗口使选模具有较高方差。

Jena 候选使用相同随机 body 语法，添加固定 sin/cos 位置编码、可训练的 `MIX` 时间投影（96→48）与 `PROJECT` 通道读出（64→21）。不把序列平均成单个值。基线为 persistence、linear_time、channel MLP、1 层 CNN、共享/独立参数的 4 层 CNN、2 层 4-head Transformer；均有可审查 GraphIR。persistence 不含可训练参数，不进行优化更新。

本轮 Jena 使用 1600 次 FP32 更新；历史上游表格使用 50 次大 batch BF16 更新，不能直接拿历史分数判胜负。这里的所有可训练任务基线与候选使用本轮同一协议。

## 元模型比较与最终测试

每个任务分别拟合 diagonal v2、原图 attention v1、图统计 Ridge，另有随机搜索对照。两种图预测器均用 3 个训练种子、相同 score 标准化、Huber+BCE 目标、验证早停及读出。Ridge 的正则化只按 meta-validation score MSE 选择。

候选预测冻结后，在完整 32 架构候选池上执行 4/8/16 次查询、10 trials；每个 trial 前 2 个随机查询相同。一次架构查询对应两个训练种子的验证结果。所有选择被冻结后，所有 winner 与所有基线用新种子 101/102/103 重训，再评估 test。

报告保留全部 trial，包括失败；明确披露元训练语料成本。相同池和 winner 被复用，不把 trial 当独立统计重复。基线根据验证结果选定，全部基线测试成绩同时披露。

每个 epoch 原子保存模型、优化器、最佳模型、epoch、history、Python/NumPy/Torch CPU/CUDA RNG；CPU 与 Jena GPU 中断恢复均已验证。数据、源代码、架构图和选择文件都有指纹保护。

## 运行

```sh
# 先在原始训练环境中准备固定数据；Adding 指向 v1 的数据归档。
PYTHONPATH=src python -m experiments.relation_v2.data --task adding --root corpora/adding --adding-archive reports/adding-v1/corpus
PYTHONPATH=src python -m experiments.relation_v2.data --task jena --root corpora/jena

PYTHONPATH=src python -m experiments.relation_v2.prepare --root corpora/adding --exclude reports/adding-v1/corpus/manifest.json
PYTHONPATH=src python -m experiments.relation_v2.prepare --root corpora/jena

# 两个任务分别运行以下阶段；多卡 worker 的 shard 互不重复。
PYTHONPATH=src python recover_numeric.py --root corpora/jena --shard 0 --shards 1 --device cuda
PYTHONPATH=src python -m experiments.relation_v2.meta --root corpora/jena
PYTHONPATH=src python -m experiments.relation_v2.search --root corpora/jena
PYTHONPATH=src python -m experiments.relation_v2.inspect --root corpora/jena
PYTHONPATH=src python recover_numeric.py --root corpora/jena --shard 0 --shards 1 --device cuda --final
PYTHONPATH=src python -m experiments.relation_v2.report --root corpora/jena
```

Jena 数据准备需要 `pyarrow`；训练仅依赖 PyTorch 和 NumPy。实际使用环境单独冻结在实验归档的 `environment.txt`。

已完成的 [两任务结果](../reports/relation-v2/RESULTS.md) 保留全部方法和基线。`recover_numeric.py` 将 DSL 已检测出的 NaN/+inf softmax 异常转为训练器可记录的数值发散；其他异常继续抛出，训练数学、预算和失败样本保留规则不变。原实验的适配器指纹见结果归档中的 `recovery.json`。
