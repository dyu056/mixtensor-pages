# Graph IR v1

目标：同一份 JSON 定义可以校验、实例化成可训练 PyTorch 模型，并导出带端口的图与矩阵，用于后续随机生成及结构分析。

本轮范围：19 个原语；命名输入、参数、节点与输出；显式参数共享；REPEAT / SCAN 嵌套子图；JSON 往返；类型与形状检查；邻接矩阵、节点类型特征及端口边表；示例图可视化。随机生成器、GNN 训练与结构规律挖掘是后续工作。

约定：节点 ID 与输入、参数名互不重复，节点引用采用 `node:port`，普通节点可简写为 `node`（默认 `out`）；子图通过显式输入绑定访问外部参数，不允许隐式捕获。JSON 只保存结构与初始化配置，训练后权重用 `state_dict` 单独保存。

v1 使用具体形状校验；输入形状可写符号，但编译前必须提供符号值。轴标签作为显式元数据保存，不宣称已证明所有轴语义。结构检查不保证训练稳定性。数值相关的索引范围、NaN 与 mask 内容仍由原语在运行时检查。

LOOP v1 状态为单个张量；SCAN 可无 carry。更复杂状态可显式拼接或在后续扩展树结构契约。循环子图只声明输入和计算，所有参数由根图 ParameterDefinition 注册。REPEAT 支持 shared / per_step 参数绑定；SCAN 支持 Sequence / Window、shared / per_step。不展开循环次数来存储图。

验收：JSON 往返保持定义；非法引用/环/端口/形状/循环状态被拒绝；MLP、attention、REPEAT 与两种 SCAN 数值和梯度符合直接 PyTorch 参考；共享参数只注册一次；矩阵保留重复输入端口与子图层级；执行模型 forward 只调用已编译网络。

## Python 接口

```python
import torch
from matrix_dsl.graph_ir import GraphIR, Node, TensorSpec, ParameterSpec

graph = GraphIR(
    inputs={'x': TensorSpec(('B', 4), axes=('batch', 'feature'))},
    parameters={'w': ParameterSpec((4, 8))},
    nodes=[
        Node('projection', 'PROJECT', {'x': 'x', 'weight': 'w'}),
        Node('activation', 'NONLINEAR', {'x': 'projection'}, {'activation': 'relu'}),
    ],
    outputs={'y': 'activation'},
    name='Example',
)
graph.save('example.json')
restored = GraphIR.load('example.json')
checked = restored.validate({'B': 2})
model = restored.compile({'B': 2}, seed=42)
y = model(x=torch.randn(2, 4))['y']
y.sum().backward()
matrices = restored.matrices({'B': 2})
```

执行必须使用命名输入；结果是命名输出的字典。`model.p.tensor('w')` 可读取唯一注册的参数。两次引用同一个参数名就是共享参数，不会复制。编译会复制源定义，后续修改原 GraphIR 不会悄悄改变已经编译的模型。

`ParameterDefinition` 在构建时创建 `nn.Parameter` 或 buffer。`GraphModule.forward` 只调用 `self.network(self.p, inputs)`；后端解释器的 Python 循环负责调度原语，模型没有隐藏的 PyTorch 层。与现有 `dsl_nn.ParameterDefinition` 的追踪式模块转写接口相互独立，旧接口继续可用。

`state_dict` 保存权重；Graph IR JSON 保存结构和初始化。需要一起保存 JSON、符号绑定和 state_dict 才能恢复实验。初始化使用本地 CPU generator，不修改全局随机状态。CPU 是默认位置；模型及输入可以一起迁移设备。同质浮点图可以使用 `.double()` 等转换；混合浮点 dtype 的图应保持声明的各个 dtype。

## 类型与引用

- `TensorSpec(shape, dtype='float32', axes=())`：支持非负维度和符号维度；轴标签数量必须与 rank 一致。
- `ParameterSpec(shape, dtype='float32', init='xavier_uniform', trainable=True, value=None)`：参数形状必须具体。初始化支持 `xavier_uniform`（二维非空矩阵）、`normal`、`zeros`、`ones`、`constant`。常量必须显式提供与 shape 一致的 value；索引及 mask 使用 `trainable=False`、`int64` / `bool`。
- `Node(id, op, inputs, attrs={}, outputs={}, body=None)`：普通原语的端口名与现有 Python 函数形参一致。参数值必须通过 inputs 引用，静态配置放在 attrs。`CONCAT.xs` 使用有序引用列表。
- 普通节点输出固定名为 `out`；`n` 和 `n:out` 等价。REPEAT 输出为 `n:state`；SCAN 输出为 `n:ys`，有状态时还有 `n:carry`。
- 普通节点 `outputs` 可省略，由校验器推断；填写时会对照推断结果检查 shape/dtype。边界输出通过 GraphIR.outputs 声明。
- 节点可按任意顺序提供，执行前拓扑排序；数据流环被拒绝，循环必须使用 LOOP。
- ADD / MUL 遵循当前 DSL 的严格同形契约，必须显式 BROADCAST，不能依赖隐式广播。
- 所有 19 个原语均可通过 Graph IR 调用；当前原语本身的功能范围仍适用。

`validate()` 返回 `.order`、`.specs`、`.outputs`、`.children`、`.warnings`。它不分配实际中间激活；普通操作主要使用 meta tensor 推断，ROUTE / NORMALIZE 使用不依赖值的契约检查。未连接到输出的定义生成 warning，不会被自动删除。它不证明索引值有效、数值稳定、梯度非零、任务语义合理或 batch 隔离；这些仍需生成器规则与 smoke test。

## LOOP 契约

REPEAT 节点：

```python
Node('loop', 'REPEAT',
     inputs={'state': 'x', 'w': 'weight'},
     attrs={'steps': 3, 'max_steps': 8, 'binding': 'shared', 'output': 'last'},
     body=body_graph)
```

body 的输入必须恰好是 `state` 与显式 capture 名（例如 `w`）；输出必须只有 `state`，且状态 shape/dtype 不变。`output='stacked'` 增加最前面的迭代轴。支持零步，仍然检查 body 契约。`binding='per_step'` 时每个 capture 改为引用列表，长度等于实际步数；零步且含 per_step captures 暂不支持。

SCAN 节点：

```python
Node('scan', 'SCAN',
     inputs={'xs': 'x', 'carry': 'initial', 'w': 'weight'},
     attrs={'domain': {'kind': 'sequence', 'axis': 0, 'out_dim': 0},
            'binding': 'shared'},
     body=body_graph)
```

body 输入包含 `fragment`、可选 `carry` 以及 captures；输出为 `y` 和对应的可选 `carry`。窗口扫描使用现有 `Window` 的 JSON 字段，参见 [LOOP API](loop-family.md)。per_step 列表按原始 domain 索引绑定；反向序列扫描仍按原始位置选择参数并返回对齐到原位置的输出。

参数统一注册在根图，body 不允许私有 parameter 表或隐式外部引用。循环可嵌套，上限 32 层；运行成本预算由后续生成器控制。

## 矩阵及分析接口

`graph.matrices(symbols)` 返回 JSON 可序列化的数据：

- 每个 scope 的 `node_ids` 定义行列顺序。顶点包括 INPUT、PARAMETER、原语调用、OUTPUT。
- `adjacency[source][target]` 是引用边的数量。比如 `ADD(x, x)` 对应值 2，而不是丢掉一条端口信息。
- `edges` 保留 source_port、target_port、列表位置 item 与 TensorSpec；矩阵不能替代它。
- `node_features` 是固定 22 列类型 one-hot。列顺序在 `feature_columns` 中。它不是完整的 GNN 特征工程：shape、attrs、参数共享和层级另外保留。
- `hierarchy` 关联父控制节点与 body scope，并保存显式绑定。子图不会被误当作与外层无关的独立模型。
- 参数只出现为对应定义的参数顶点，多条使用边表征共享。子图参数输入通过 hierarchy 追溯根图参数。
- `stored_depth` 是存储图中的依赖深度，不是循环展开深度，也不是与任意重编号无关的唯一节点编号。

`structural_summary(graph, symbols)` 输出原语静态调用频次、所在 scope、存储图深度和独立可训练参数量。这只是结构记录，尚未把频次与训练成败做关联。

`definition_hash` 是包含名字、元数据和节点顺序的精确定义摘要，不是图同构去重算法。后续架构去重需要单独实现，不能把不同 hash 直接视为不同计算结构。

## 示例与复验

六个示例：[MLP](graph-ir/MLP.json)、[SelfAttention](graph-ir/SelfAttention.json)、[SharedRepeat](graph-ir/SharedRepeat.json)、[PerStepRepeat](graph-ir/PerStepRepeat.json)、[SequenceScan](graph-ir/SequenceScan.json)、[WindowCNN](graph-ir/WindowCNN.json)。可直接打开 [交互图与矩阵](graph-ir/index.html)，无需网络依赖。

```sh
.venv/bin/python -m pytest tests/test_graph_ir.py -q
.venv/bin/python scripts/build_graph_ir_examples.py
python3 -m http.server 8765 --directory docs
```

JSON 保留符号 `B`，示例矩阵生成时绑定 `B=2`。窗口示例等价于一个无 bias 的 3×3 Conv2d，序列示例是线性投影后的累积和，并非完整 LSTM。短训练测试仅验证梯度和优化器链路，不代表随机架构搜索或基准排名。
