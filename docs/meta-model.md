# 架构图 → 性能预测 → 搜索评测

这一版本验证的是：**元模型指导搜索能否找到比预定义基线更好的任务模型**。预测 loss 的相关性只作为诊断指标，最终比较独立重训后的任务测试 MSE。

## 代码入口

| 文件 | 职责 |
|---|---|
| `src/matrix_dsl/metamodel/encoding.py` | GraphIR → 节点特征、端口关系、祖先 mask、距离 |
| `src/matrix_dsl/metamodel/model.py` | 图 attention 编码器、score 与完成概率双输出 |
| `experiments/metasearch/data.py` | 固定训练 / 验证 / 测试数据，训练标签统计量归一化 |
| `experiments/metasearch/collect.py` | 生成架构、训练记录、失败记录、精确配置缓存 |
| `experiments/metasearch/predict.py` | 家族划分、拟合图元模型与统计特征 ridge、冻结候选预测 |
| `experiments/metasearch/search.py` | 等查询预算搜索回放、冻结入选架构、独立重训 |
| `experiments/metasearch/report.py` | 所有搜索试验、基线、失败、预测质量、计算成本报告 |

## 元模型输入与结构

节点矩阵为 `[节点数, 46]`。包括 22 种节点类型 one-hot（19 原语以及 INPUT / PARAMETER / OUTPUT）、15 个数值特征、6 个激活函数指示值和 3 个归约方式指示值。数值特征包括 shape、参数量、循环次数和参数共享方式；不读取模型名、生成 seed、训练 loss 或 status 作为输入。

关系张量为 `[目标节点, 来源节点, 20]`，记录输入端口及循环体输入 / 输出边界。另有祖先可达 mask 和截断的最短路径距离矩阵。一个节点可以 attention 到自己及其上游祖先，不能访问下游节点；batch padding 独立屏蔽。这个约束表示程序的数据依赖，**不代表 attention 权重具有因果贡献的含义**。

REPEAT / SCAN 的 scope 会展开为带边界连接的图；运行次数和共享方式作为特征，不将每次运行复制成一个子图。这是已有循环图的编码方式；当前候选生成已暂时排除 REPEAT 和 SCAN，仅生成普通 DAG，待实现执行展开后的表征再重新考虑循环搜索。

实验模型：节点特征投影到 32 维 → 两层四头 attention → masked mean / scaled sum 汇聚 → 拼接 16 维任务 embedding → 两个预测头。attention logits 加上端口类型与路径距离 bias。训练三份不同初始化的模型并平均预测；三者的离散程度只是启发式不确定度，未经校准。

## 标签与训练

score 是“零预测器验证 MSE / 任务模型验证 MSE”的自然对数，加极小常数避免除零。越大越好；归一化只使用元训练集合的均值与标准差。失败运行不伪装成一个有限的极差 loss：单独预测训练完成比例，score 回归只使用成功运行。

损失由完成概率 BCE 与 score Huber 两部分组成。AdamW、梯度裁剪；元验证集选择 checkpoint 和停止时间。搜索排序使用完成概率乘以 score 的 sigmoid，兼顾有效训练概率和预期性能。这是一个排序启发式，不等于经过校准的期望 loss。

图之间按规范化的**子图家族**划分为训练 / 验证 / 候选集，所有任务、训练种子和循环深度变体保持同一划分。移除 REPEAT 包装时将 flat `embed` 与 body `state` 统一；划分仅依赖结构。`predictor/split.json` 保存最终划分，优先于初期 collection manifest。

## 固定实验协议

- 三个合成回归任务：集合能量、相邻乘积、首尾乘积。输入 16 个 token，每个 4 维。
- 训练集 2048，验证集 512，最终测试集 512；所有目标只按训练标签统计量归一化。
- 每个架构、任务训练种子 11 / 23；300 AdamW steps，batch 64，width 16，学习率 0.001，weight decay 0.0001，梯度裁剪 1。
- 7 个基线：linear、MLP、DeepSets、单层 CNN、共享四层 CNN、独立四层 CNN、attention。按同一验证协议选出每个任务的比较基线，保留全部基线测试结果。
- 比较图元模型、原语统计特征 ridge、随机搜索。每个 trial 从候选集合抽 24 个架构；共同随机查询前 2 个，再各查询 6 个。每次查询包括两个训练种子。10 个 trial，使用相同候选池。
- 冻结的候选预测只读结构。搜索器仅查询选中的候选验证记录；选择完成后才读取独立测试数据。
- 所有入选架构及 7 个基线按新种子 101 / 102 / 103 重训并测试。完成状态、曲线、配置 hash、数据 hash 和最终权重落盘。
- V1 为 128 个架构，test seed 1731。V1 未超过基线后，预先固定一次 V2 扩量至 512 个架构，协议不变，使用新的 test seed 1732。V2 是后续探索实验，不能把两轮测试看成一次预注册的确认性实验。

## 运行

在仓库根目录运行，Python 3.10+。CPU 实验使用一个线程 / 训练进程；并发 worker 数按机器资源设置。

```sh
python -m pip install -e '.[training,test]'
python -m experiments.metasearch.collect --output reports/my-corpus --count 512 --workers 16
python -m experiments.metasearch.predict --corpus reports/my-corpus
python -m experiments.metasearch.search --corpus reports/my-corpus --test-seed 1732 --workers 16
python -m experiments.metasearch.report --corpus reports/my-corpus
python -m experiments.metasearch.infer --graph reports/my-corpus/graphs/g1000.json --checkpoint-dir reports/my-corpus/predictor --task local_product
python -m pytest -q
```

图元模型的 `.pt` checkpoint 包含 model_config、state_dict 和 score normalization，可用 `GraphScoreModel(**checkpoint['model_config'])` 恢复。它们是推理 checkpoint；没有保存 optimizer 状态来恢复中断的元训练。已完成的任务训练结果可按精确配置恢复使用；冻结后的元预测拒绝覆盖，改动实验须新建 corpus。

## 解释结果的边界

这是内部合成 benchmark，不能宣称在公开 NAS benchmark 或真实应用任务上达到 SOTA。训练 / 元验证架构的预热成本单独计入，查询数一致不等于总计算量相等。trial 重用候选和获胜模型，所以不是独立统计重复，不报告虚假的显著性。

同样的训练步数和宽度不保证参数量、实际计算量相等，原始结果保留这些信息。当前搜索是离线回放固定排序，没有在线更新元模型。首轮失败和后续探索结果都应保留。

下一步应由验证数据定位瓶颈：若候选集合的最佳验证架构仍很差，优先改善生成语法和读出结构；若集合内存在好模型却选不到，改进排序目标与探索策略。不要用已经看过的测试结果选择下一轮架构或超参数。

## 已完成结果

[交互报告](meta-search/index.html) 展示两轮的完整结果。V2 的 g1060 在相邻乘积任务超过 CNN 基线，但图搜索总体尚未胜过随机搜索。详细结论和原始数据见 `reports/meta-search-v2/FINDINGS.md`。
