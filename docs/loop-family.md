# LOOP 控制结构 · Python DSL v0.2

现在共有 **19 个具体原语：17 个计算/结构原语，以及 LOOP 家族的 REPEAT、SCAN 两个控制原语**。SCAN 从原来的时间扫描扩展而来，不重复计数。LOOP 是命名空间和类别，本身不是第三个执行原语。

| 类别 | 子原语 | 循环边界 | 循环体 |
| --- | --- | --- | --- |
| LOOP | REPEAT | `0 <= steps <= max_steps`，恰好执行 steps 次 | `body(state, parameters) -> next_state` |
| LOOP | SCAN | Sequence 长度或 Window 输出网格 | `body(fragment, carry, parameters) -> (output, next_carry)` |

注册表 ID 使用 `REPEAT` / `SCAN`，`get_operation('LOOP.REPEAT')` 和 `get_operation('LOOP.SCAN')` 也可查询。调用可以使用 `d.LOOP.REPEAT` / `d.LOOP.SCAN`，或对应的小写函数。

## REPEAT：重复网络块

```python
import matrix_dsl as d

# x: (B, N, D); weight: (D, D)
def block(state, parameters):
    return d.nonlinear(d.project(state, parameters['weight']))

y = d.LOOP.REPEAT(
    block, x,
    steps=3,
    max_steps=6,
    parameters=d.Shared({'weight': weight}),
    output='last',
)
```

- `state` 可以是张量、嵌套 tuple、具名 dict；每轮结构、形状、dtype、device 必须保持一致。
- `Shared(tree)` 每轮绑定同一组参数；`PerStep((tree0, tree1, ...))` 每轮绑定对应参数，必须恰好提供 steps 组，参数树规格必须一致。这里的参数树叶子是张量或 None。
- `output='last'` 返回最终 state；`'stacked'` 将每轮更新后的 state 沿新增的第 0 轴堆叠，不包含初始 state。
- steps=0 时，last 返回初始 state；stacked 的每个张量叶子具有长度为 0 的新轴。越界次数在执行循环体之前报错。
- 第一版不包含提前退出条件或数据相关的迭代次数。

独立参数的 Transformer 堆叠使用 PerStep；共享参数的 Looped Transformer 使用 Shared。每轮重新执行原语并构建 autograd 路径，没有复制/冻结权重或截断梯度。

## SCAN：空间窗口或序列域

```python
# source: (B, C, H, W)
domain = d.Window(
    axes=(2, 3),
    size=(3, 3),
    stride=(1, 1),
    dilation=(1, 1),
    padding='same',
    output_axis=-1,
)

# fragment: (B, C, 3, 3); parameters['weight']: (C * 9, E)
def kernel(fragment, carry, parameters):
    patch = d.reshape(fragment, (B, C * 9))
    output = d.project(patch, parameters['weight'])
    return output, carry

output, carry = d.LOOP.SCAN(
    kernel, source, None,
    domain=domain,
    parameters=d.Shared({'weight': weight}),
)
# output: (B, E, H, W); carry: None
```

`Window`、`Sequence`、`Shared` 和 `PerStep` 是 ParameterDefinition 的元数据结构，不是额外的计算原语。定义中的 B/C 等尺寸由输入规格提供。

窗口扫描规则：

- `axes` 使用张量轴编号；`size`、`stride`、`dilation` 按该轴顺序对应。
- padding 支持 `valid`、`same`、非负整数或每轴 `(before, after)` 对。第一版只补零。
- fragment 的轴顺序固定为“未扫描的轴，随后为窗口轴”。
- 窗口按输出网格的行优先顺序遍历；`output_axis` 指定输出网格插入单步 output 的位置。默认 -1 将网格轴追加在末尾。
- 所有单步输出必须同形状/dtype/device，最终返回 `(grid_output, final_carry)`。第一版要求非空网格。
- 有 carry 时按扫描顺序更新；无 carry 的 CNN 可以独立计算各窗口。本后端仍以串行实现为主，不承诺卷积内核级性能。

时间扫描使用 `Sequence(axis=0, out_dim=0, reverse=False)`。每轮 fragment 去掉被扫描的轴；reverse=True 逆序更新状态，但输出仍按原序列坐标排列。PerStep 参数按原坐标绑定。当前要求非空序列。

## ParameterDefinition 与 forward

模型定义阶段使用只有形状、dtype、device 的 Symbol。控制原语的 body 被记录为显式子图；forward 执行子图，不执行原始构图函数。索引/配置预处理属于 ParameterDefinition；循环调度和窗口读取属于原语后端。

展开版 forward 直接列出 `d.LOOP.REPEAT` / `d.LOOP.SCAN`，循环体在同一文件的 `node_*_body` 中完整列出。没有使用原生 nn.Module.forward 回退。静态审计和运行时审计都会进入控制原语的 body，防止将未声明的张量计算藏入循环。

图编译器对具体形状专门化；零步 REPEAT 的 PerStep 模式尚需显式参数模板，因此当前图编译器拒绝该组合，Shared 零步可用。没有承诺任意动态形状、torch.compile、分布式或提前停止语义。

## 兼容旧 SCAN

未提供 `domain` 时，保留原接口：`step(state, xt) -> (next_state, output)`，以及 `dim/out_dim/reverse/output_template` 参数。新域接口采用 `step(fragment, carry, parameters) -> (output, next_carry)`，顺序不同，不能混用旧布局选项。

现有 RNN/GRU/LSTM 的审查适配器已改用明确的 Sequence 域；旧 examples/compositions.py 仍可通过兼容接口运行。

## 本轮覆盖验证

运行 `python -m smoke.loop_recap` 重新枚举所有公开 torch.nn 类，并比较新表示的输出及输入/参数梯度。结果见 [nn.Module 重新概括](../reports/nn-loop-recap.md)。逐类展开代码的独立复验命令为 `python -m dsl_nn.validate_expanded`。
