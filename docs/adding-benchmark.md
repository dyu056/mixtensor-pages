# Adding v1：从架构图预测走向等预算搜索

本实验验证：元模型对未参与拟合的随机架构排序后，能否在固定的查询次数内，找到比随机搜索和人工基线更好的 Adding 模型。它只覆盖一个固定任务实例，不代表跨任务泛化或总体计算成本优势。

## 数据与上游对齐

来源为 [bangzhehuang/Meta-model](https://github.com/bangzhehuang/Meta-model/tree/a3e784dc54dc01174903394fd0155afb1515ad94)，固定 commit `a3e784dc54dc01174903394fd0155afb1515ad94`。`experiments/adding/data.py` 独立实现相同数据配方；三个划分均已与上游生成函数逐张量核对。

- 序列长度 48，输入 `[batch,48,2]`：数值通道和标记通道。数值均匀分布于 [-1,1]。
- 每条序列标记两个位置，目标为这两个位置数值的**平均值**，输出 `[batch,1]`。
- 第一位置在 0–11，第二位置在 24–47。保留上游实现的实际范围，而不把它误读成均匀覆盖两个半区。
- train / validation / test 数量为 1024 / 256 / 256；data seed 为 `6838890811063975792`，各 split 独立派生种子。
- 仅用训练标签的均值和总体标准差归一化；NMSE 是归一化标签空间的 MSE，原始 MSE 一并保留。

数据相同不代表复现上游模型分数：本实验使用 Mixtensor 的生成语法、FP32 训练和独立的公平比较。历史上游采用其他架构/精度，不作为直接胜负基线。

## 冻结的实验协议

128 个唯一随机架构，隐藏宽度 64，4–12 个 body 原语节点，REPEAT 次数为 1/2/4；支持共享与非共享参数。参数上限 100,000，展开操作上限 128。生成器语法保持现有版本，不依据本次结果修改。

以 body family 划分元模型 train / validation / candidate，比例约 60% / 15% / 25%；同一 body 的循环变体不会跨集合。每个架构用种子 11、23 训练。

另有 7 个显式 DSL 基线：linear、MLP、DeepSets、CNN 单层、共享参数 4 层 CNN、独立参数 4 层 CNN、attention。各模型隐藏宽度相同，**参数量不完全相等**，报告会列出实际参数量。

所有任务模型统一：400 epochs，batch 256，每次 1600 个 optimizer steps；AdamW，学习率 0.001，weight decay 0，梯度裁剪 1，FP32，确定性 CUDA，关闭 TF32。验证集选择最佳 epoch，包括 epoch 0。数据采集不评估 test。

每个 epoch 原子保存 model、optimizer、best model、历史、epoch/global step 和 Python/NumPy/Torch CPU/CUDA RNG。CPU 与 GPU 的 epoch 边界中断恢复已与连续训练逐项核验。使用同一命令即可续跑；被冻结的训练源代码、配置或数据指纹变更会被拒绝。

## 元模型和搜索

使用现有关系感知的图 attention 模型：节点特征、端口关系、祖先可达 mask、距离偏置，以及图级 mean/sum pooling。宽度 32，4 heads，2 layers。双输出分别估计完成概率与 score；score 是零预测验证 NMSE 相对模型验证 NMSE 的对数改善。三个元模型种子组成 ensemble，score 仅按元模型训练集合标准化。

拟合只读取 train 与 validation 架构的训练结果。candidate 只编码结构，不读取 loss。以 train 拟合、validation 选择的图统计 ridge 作为更简单的搜索对照。所有预测写入冻结文件后，才运行搜索。

比较 GNN、ridge 与 random：

- 都从完整 held-out candidate 池搜索，查询预算为 4/8/16 个架构。
- 每个架构查询返回两个种子的平均验证 NMSE，即一个 query 对应两次任务训练。
- 10 个搜索 trial；同一个 trial 的三个方法共享最初两个随机查询。后续 GNN/ridge 按预先冻结的预测 utility 排序，random 随机排序；本版不做在线重训。
- 每个预算只使用已查询的 validation 结果选择 winner，失败也消耗预算。
- 所有 winner 与全部人工基线冻结后，用新种子 101/102/103 从头训练；随后才评估 test。
- 报告全部 trial 的结果，不用 best-of-trials 代表方法。trial 复用候选池与 winner，不能当作完全独立样本进行显著性声明。

等查询比较不包含元模型 warm data 的额外成本，报告单列这部分训练次数。当前 collection 是离线 benchmark 数据集，未查询候选标签不会被 predictor 读取。

## 复现入口

从项目根目录运行（安装 `.[training]`）：

```sh
PYTHONPATH=src python -m experiments.adding.prepare --root work/adding-v1
PYTHONPATH=src python -m experiments.adding.worker --root work/adding-v1 --shard 0 --shards 1 --device cuda
PYTHONPATH=src python -m experiments.adding.meta --root work/adding-v1
PYTHONPATH=src python -m experiments.adding.search --root work/adding-v1
PYTHONPATH=src python -m experiments.adding.worker --root work/adding-v1 --shard 0 --shards 1 --device cuda --final
PYTHONPATH=src python -m experiments.adding.report --root work/adding-v1
```

多卡时每卡一个 worker：设置各自 `CUDA_VISIBLE_DEVICES`，使用不同 `--shard`，`--shards` 保持一致。所有 collection worker 完成后再训练 predictor；所有 final worker 完成后再生成报告。

产物包括 manifest、GraphIR JSON、每次训练完整 loss history/checkpoint、predictor 权重与冻结预测、查询记录与冻结选择、各 seed 的独立最终结果，以及 `report.json` / `REPORT.md`。恢复使用本地可信的 checkpoint；不要加载来源不明的 pickle 文件。

## 本轮数据归档与跨版本复核

正式训练环境为 Linux、PyTorch 2.8.0+cu128、8×A100 40GB，完整依赖位于结果目录的 `environment.txt`。`datasets.npz` 保存本次实际使用的输入和归一化/原始标签，`data-snapshot.json` 记录逐划分与归档文件 SHA256。

本机 PyTorch 2.14.0 重新生成的**原始数据完全一致**，归一化标签最大绝对差为 2.3842e-7。因此，精确续训应使用原训练环境；本机审计用归档张量对照训练指纹，并单列重新生成的数值差异。旧 run 的训练器会拒绝不同的数据指纹，不会静默混用数据。

```sh
PYTHONPATH=src python scripts/audit_adding_run.py --root reports/adding-v1/corpus
python scripts/build_adding_report.py --root reports/adding-v1/corpus --output docs/adding-benchmark/index.html
```

[交互报告](adding-benchmark/index.html) · [原始实验汇总](../reports/adding-v1/corpus/REPORT.md)
