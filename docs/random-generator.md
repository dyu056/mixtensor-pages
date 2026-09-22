# 随机模型生成器 V0

实现位置：`src/matrix_dsl/generation/`。生成器返回 Graph IR，不动态生成或执行 Python 字符串。当前仅生成保形 DAG；REPEAT 和 SCAN 暂时排除在候选节点之外。DSL 仍保留循环执行支持。任意输入布局、分类任务输出头、GNN 引导采样尚未接入。

## 使用

```python
from matrix_dsl.generation import (
    TaskSpec, GeneratorConfig, RandomGraphGenerator,
    TrainingConfig, ExperimentRunner,
)

task = TaskSpec(tokens=16, features=4, outputs=2)
generator = RandomGraphGenerator(GeneratorConfig(hidden_width=16))
graph = generator.generate(task, seed=42, batch_size=16)
graph.save('architecture.json')
model = graph.compile({'B': 16}, seed=123)
result = ExperimentRunner(TrainingConfig(steps=80)).train(
    graph, task, training_seed=123, checkpoint='checkpoint.pt',
)
```

输入固定为 `(B, tokens, features)`；输出为 `(B, outputs)`。首尾适配为输入 PROJECT、token mean REDUCE、输出 PROJECT 以及 BROADCAST + ADD 偏置。所有适配操作、参数和成本均计入模型。内层状态为 `(B, tokens, hidden_width)`，全程不混合 batch 轴。

`min_nodes` / `max_nodes` 指内层子图的原语调用数，不含适配器、输出头及控制节点。第一版不把它解释为有效非线性深度。可以产生线性或退化网络，这些是合法探索样本，不暗中补非线性。

## 生成规则

从输入建立主路径。每次操作必须读取当前主路径；二元操作的另一输入从历史主路径端点选择，允许跳连、自乘和重复引用。每次融合最多两个输入，但没有声称限制全图分支总数。

8 个提案规则：

- PROJECT：新增一个可学习的特征投影矩阵。
- NONLINEAR：从 relu、gelu、tanh、silu 选择。
- NORM：无 affine 参数的尾轴 LayerNorm。
- ADD / MUL / GATE：选择一个历史同形张量作第二输入。
- SHIFT：token 轴位移 -2、-1、1、2，边界补零。
- attention：显式展开 COMPARE → BROADCAST(scale) → MUL → NORMALIZE → MIX，共 5 个原语。q 与 values 使用当前主路径，k 从历史端点选择；这里不是完整标准 Transformer 模板。

规则在剩余节点预算允许的集合中均匀采样；并不意味着原语或架构均匀分布。可以通过 `operations` 配置子集，且必须保留至少一个单节点规则以填满预算。固定输入/输出骨架和 attention 组合都会形成采样偏好，后续频次分析必须以提案分布为对照。

`repeat_choices` 暂时仅允许 `(1,)`，表示直接 DAG；旧的多次循环配置会明确报错。`GraphFilter` 同时拒绝 REPEAT / SCAN 控制节点。模型规模由 `min_nodes` / `max_nodes` 控制。历史结果中的循环模型仍可读取和执行，但不再作为新生成候选。

## 约束与失败

`GraphFilter` 检查全部子图的契约、未使用定义、独立参数量、最大声明张量元素数、展开操作数和展开原语输出元素累计值。最后一项是保守的原语输出规模代理，**不是**实测显存/内存上限，不涵盖所有后端临时张量、优化器状态或 FLOPs。

候选不符合预算时记录原因并重新提案，最多 `max_attempts` 次。耗尽后抛出 `GenerationError`，保留 `.rejections`。seed 使用局部 Python RNG；参数初始化与数据使用各自的局部 Torch generator。

`architecture_hash` 排除图名和 metadata，包含编号、连接、参数定义和操作配置；用于去掉完全相同的编号结构，不是图同构或计算等价去重。本轮数据仅标注重复，不丢弃重复样本。

## 训练试跑

`ExperimentRunner` 当前只提供 CPU / FP32 的固定合成回归任务：

1. 同一 token 的前两个特征乘积取 tanh，再沿 token 求均值。
2. 相邻 token 的第一个与第二个特征乘积求均值。

因此该 runner 要求 tokens>=2、features>=2、outputs=2；生成器本身允许其他正维度。样本来自独立标准正态分布，train 与 validation 从同一个局部随机流的不同区段生成，没有测试集。所有架构复用相同数据与 minibatch 抽样序列。

默认 256 个训练样本、128 个验证样本、batch=16、Adam 学习率 0.001、梯度裁剪阈值 1、训练 80 步，每 10 步验证一次。不同参数量的模型同样训练步数不代表同样算力预算。

保存初始/最终训练和验证 loss、最佳验证 loss 及步数、每步 minibatch loss、裁剪前梯度范数、首步缺失/零梯度参数、实际步数、耗时和状态。常量基线用训练标签均值预测验证集。最佳验证 loss 允许出现在训练前第 0 步；checkpoint 保存最终模型，而不是最佳验证模型。

状态包括 completed、invalid_graph、nonfinite、out_of_memory、timeout、runtime_error。训练失败的 final loss 保留 null，不填造假的惩罚值。超时在训练步之间协作检查，不能中断一个正在执行的昂贵算子。未测峰值内存。部分结构可能输出常量或梯度为零，这些情况记录下来，不冒充优质模型。

## 第一批数据

```sh
.venv/bin/python scripts/run_random_pilot.py
```

默认生成 seed 0–31，使用共同的训练初始化 seed=123 和数据 seed=2026。每个架构仅训练一次，结果只用于确认链路和研究下一轮配置，不支持可靠的架构排名或普遍设计结论。

- `reports/random-pilot-v0/seed-XXXX/graph.json`：原始模型定义。
- `matrices.json`：邻接、节点类型特征、端口与子图关系。
- `result.json`：完整训练与失败记录。
- `checkpoint.pt`：成功完成的最终权重、优化器状态、抽样 RNG 状态及编译绑定。
- `reports/random-pilot-v0/results.json`、`results.csv`：汇总。
- [结果页面](random-models/results.html)、[所有模型图](random-models/index.html)。

再次运行需通过 `--output` 指定新目录，防止覆盖已有完整实验。图可视化输出可由 `--docs` 指定。checkpoint 含恢复所需状态，但本版没有封装 resume 命令。

本轮实际结果：32 个不同编号结构，28 个完成 80 步训练，4 个非有限数值失败；最佳最终验证 MSE 为 0.04066996，训练均值常量基线为 0.04221117，只有 3 个模型的最终验证 MSE 低于该基线（其中两个差距极小）。不能将完成训练视为表现良好，或将单次胜出视为统计显著。全量测试 202 项通过、1 项无 CUDA 跳过。
