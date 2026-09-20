# Matrix DSL · Python API v0.2

这套库提供 19 个具体原语：17 个计算/结构原语，以及 LOOP 家族的 REPEAT 和 SCAN。旧网页中的 18 个标识均保留。它们可组合、可反向传播，输入输出仍是原生 `torch.Tensor`，可直接放进自己的 `nn.Module.forward`。

## 安装与运行

在 `Meta-model-matrix` 目录下：

```bash
python -m pip install -e '.[test]'
python -m examples.demo
python -m pytest -q
```

已为本项目创建独立的 `.venv`。也可直接使用：

```bash
.venv/bin/python -m examples.demo
.venv/bin/python -m pytest -q
```

Python 3.10+，PyTorch 2.2+。具体已验证版本和设备见 `docs/validation.md`，不把未实测版本或设备视作已验证。

```python
import torch
import matrix_dsl as dsl

x = torch.randn(2, 8, 16)                     # B=2, N=8, D=16
w = torch.nn.Parameter(torch.randn(16, 32))  # D×E
b = torch.nn.Parameter(torch.zeros(32))
y = dsl.project(x, w)                        # (2, 8, 32)
y = dsl.add(y, dsl.broadcast(b, y.shape))
y = dsl.nonlinear(y, activation="gelu")
y.square().mean().backward()
```

## 全局契约

- **张量与维度**：`B...` 表示零个或多个批次轴；`N/M` 是输入/输出对象数；`D/E/H` 是特征维度；`K` 是邻域大小；`T` 是时间长度；`S` 表示任意完整形状。轴从 **0** 开始，负数轴与 PyTorch 一致。
- **行向量约定**：PROJECT 的 `weight` 为 `(D,E)`。从 `nn.Linear` 迁移时传 `layer.weight.T`。函数不会自动添加偏置；用 BROADCAST 和 ADD 表达。
- **参数所有权**：函数不创建、不注册、不初始化可学习参数。调用者把权重放到 `nn.Module` 的 `nn.Parameter`、`ParameterDict` 或 `ParameterList` 中，再传给函数。优化器和 `state_dict` 因而仍由 PyTorch 管理。
- **设备和 dtype**：不自动转设备或转参数类型。参与同一算术操作的输入必须同 device、同 dtype。数值操作要求实数浮点 dense Tensor；结构操作可处理整数/布尔等 dense Tensor。NORMALIZE 为半精度 softmax 使用 float32 累积，再还原输入 dtype。浮点数值精度、溢出和设备支持遵循后端。
- **广播**：ADD、MUL、GATE 不做隐式广播；用 `broadcast` 显式对齐。MIX、COMPARE 允许前导批次轴按 PyTorch 规则广播，并在接口中明确声明。索引张量及 mask 不可学习。
- **梯度**：不使用 `.detach()`、NumPy 转换或原地修改。对可微的数值输入保留计算图；max/ReLU 等在不可微点采用 PyTorch 的梯度约定。RESHAPE/PERMUTE/BROADCAST/identity 可能返回共享内存的视图，调用者不得就地改写广播视图。
- **错误**：形状、参数、轴、索引、遮罩违反契约时抛出 `ValueError`/`TypeError`；其他后端错误仍可能由 PyTorch 抛出。
- **执行范围**：这是 eager 函数库。值域检查含 Python 控制流，在加速器上可能同步；尚未保证 `torch.compile`、export、vmap、分布式张量或稀疏张量支持。SCAN 是串行循环，并不自动并行化。

## 接口总览

`*` 后为仅关键字参数。计算原语返回 Tensor；SCAN 返回 `(outputs, final_state)`；REPEAT 返回状态树或按轮次堆叠的状态树。

| DSL | Python 函数 | 输入 → 输出（默认轴） |
|---|---|---|
| PROJECT | `project(x, weight)` | `(...,D), (D,E) → (...,E)` |
| MIX | `mix(x, weights)` | `(...,N,D), (...,M,N) → (...,M,D)` |
| COMPARE | `compare(q,k,*,mode="dot",weight=None)` | `(...,M,Dq), (...,N,Dk) → (...,M,N)` |
| ROUTE | `route(x,indices,*,dim=-2,padding="error")` | `(...,N,D), (M,K) → (...,M,K,D)` |
| NORMALIZE | `normalize(scores,*,dim=-1,mask=None,empty="error")` | `S → S` |
| AGGREGATE | `aggregate(x,*,reducer="sum",keepdim=False)` | `(...,M,K,D) → (...,M,D)` |
| SHIFT | `shift(x,offset,*,dim=-2,boundary="zero")` | `S → S` |
| PERMUTE | `permute(x,dims)` | 原形状 → 按 `dims` 排列的形状 |
| RESHAPE | `reshape(x,shape)` | 原形状 → 指定形状，元素数不变 |
| REDUCE | `reduce(x,*,dim,reducer="sum",keepdim=False)` | 移除指定轴，或将其保留为 1 |
| BROADCAST | `broadcast(x,shape)` | 原形状 → 显式目标形状 |
| ADD | `add(x,other)` | `S,S → S` |
| MUL | `mul(x,other)` | `S,S → S` |
| GATE | `gate(logits,values)` | `S,S → S` |
| CONCAT | `concat(xs,*,dim=-1)` | 相同非拼接维度 → 拼接轴尺寸相加 |
| NONLINEAR | `nonlinear(x,*,activation="relu")` | `S → S` |
| NORM | `norm(x,*,normalized_shape=None,weight=None,bias=None,eps=1e-5)` | `S → S` |
| SCAN | `scan(step,xs,initial_state,*,dim=0,out_dim=0,reverse=False,output_template=None)` | 序列、状态、单步函数 → 输出序列、最终状态 |

## 1. PROJECT：特征投影

\[
Y=XW,\quad X\in\mathbb R^{B\cdots\times N\times D},\quad W\in\mathbb R^{D\times E},\quad Y\in\mathbb R^{B\cdots\times N\times E}.
\]

`x` 至少一维，最后一轴是被投影的特征；前导轴可以是任意对象/批次结构。`weight` 严格为二维共享权重。输出只改变末轴。

```python
x = torch.randn(2, 5, 4)
wq = torch.randn(4, 3)
q = dsl.project(x, wq)  # (2, 5, 3)，attention 查询投影
```

## 2. MIX：对象混合

\[
Y_{ic}=\sum_j A_{ij}X_{jc},\quad X:(B\cdots,N,D),\quad A:(B\cdots,M,N),\quad Y:(B\cdots,M,D).
\]

函数参数顺序为 `mix(x, weights)`，内部计算 `weights @ x`。允许批次轴广播，例如同一个 `(M,N)` 邻接矩阵用于每个 batch。权重无需非负或和为 1，不隐式归一化。

```python
x = torch.randn(2, 5, 4)
a = torch.eye(5)
y = dsl.mix(x, a)  # (2, 5, 4)，GCN/attention 的传播步骤
```

## 3. COMPARE：成对比较

输入 `q: (...,M,Dq)`、`k: (...,N,Dk)`；输出 `(...,M,N)`。前导批次轴允许广播。三种模式：

\[
S_{ij}=q_i k_j^\top\quad(\mathrm{dot},\ D_q=D_k),
\]

\[
S_{ij}=\sum_c(q_{ic}-k_{jc})^2\quad(\mathrm{squared\_l2},\ D_q=D_k),
\]

\[
S_{ij}=q_i W k_j^\top,\quad W\in\mathbb R^{D_q\times D_k}\quad(\mathrm{bilinear}).
\]

仅 `bilinear` 接收 `weight`。距离值是正距离，不自动取负数或 softmax；如果作为相似度需要显式变换。`squared_l2` 通过显式差分计算，额外内存随对象对数量和特征维度增长。

```python
q, k = torch.randn(2, 5, 3), torch.randn(2, 7, 3)
s = dsl.compare(q, k)  # (2, 5, 7)，attention logits 的未缩放部分
```

## 4. ROUTE：索引/邻域选取

\[
R_{ikc}=X_{I_{ik},c},\quad X:(B\cdots,N,D),\quad I:(M,K),\quad R:(B\cdots,M,K,D).
\]

`indices` 必须是与 `x` 同 device 的 `torch.int64`，至少一维。一般规则是：用 `indices.shape` 替换 `x.shape[dim]`。索引表在所有 batch 间共享；不同 batch 的不同索引表不是这个接口的隐式能力。

索引使用 Python 的 0 基计数。`padding="error"` 只允许 `[0,N)`；`padding="zero"` 额外允许 `-1` 哨兵，选出全零项，其他越界仍报错。重复索引的梯度累加到同一输入位置。

```python
x = torch.randn(2, 4, 3)
idx = torch.tensor([[-1, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, -1]])
patches = dsl.route(x, idx, padding="zero")  # (2, 4, 3, 3)，局部卷积窗口
```

stride、dilation 和图邻域结构由调用者生成索引表表达；二维 CNN 示例演示了这一点。

## 5. NORMALIZE：关系权重归一化

\[
A_{ij}=\frac{\exp(S_{ij}-m_i)}{\sum_{k\in J_i}\exp(S_{ik}-m_i)},\qquad m_i=\max_{k\in J_i}S_{ik}.
\]

`scores` 是实数浮点张量；输出同 shape、dtype、device。`dim` 指定 softmax 轴。`mask` 可省略，或是能广播到输入形状的 bool Tensor，**True 表示允许参与**。输入的负无穷也表示排除。参与位置上的 NaN 和正无穷报错。

`empty="error"` 对全排除切片报错；`empty="zeros"` 对这些切片返回全零，且该切片的 score 梯度为零。后者切片的权重和是 0，不是 1。物理长度为 0 的归一化轴始终报错。

```python
scores = torch.randn(2, 5, 5)
causal = torch.ones(5, 5, dtype=torch.bool).tril()
a = dsl.normalize(scores, mask=causal)  # (2, 5, 5)，因果 attention 权重
```

## 6. AGGREGATE：邻域汇聚

\[
Y_{ic}=\sum_{k=1}^K R_{ikc},\quad R:(B\cdots,M,K,D),\quad Y:(B\cdots,M,D).
\]

`x` 至少三维，倒数第二轴固定为邻域轴。`reducer` 为 `sum`、`mean` 或 `max`；`keepdim=True` 保留长度为 1 的邻域轴。

这是 `reduce(x,dim=-2,...)` 的语义宏。`mean` 会把补零槽位算进分母，因此不等于变长邻域的有效节点平均；变长图需显式权重或独立计数。

```python
messages = torch.randn(2, 5, 3, 4)
y = dsl.aggregate(messages, reducer="sum")  # (2, 5, 4)，GNN 消息聚合
```

## 7. SHIFT：位置平移

\[
Y_i=X_{i-\delta}\quad\text{在合法范围内；否则补零。}
\]

`offset` 是带符号整数，正值向更大索引移动。`dim` 指定移动轴，默认倒数第二轴。`boundary="zero"` 补零；`"circular"` 循环回绕。输出形状不变。空轴返回空张量；零填充下绝对位移大于等于轴长则全零，并保留零梯度计算图。

```python
x = torch.randn(2, 5, 4)
previous = dsl.shift(x, 1)  # (2,5,4)，因果卷积的一个延迟 tap
```

## 8. PERMUTE：轴置换

对 `x` 的全部轴作排列，`dims` 必须是包含每个轴恰好一次的序列，可使用负数轴。

\[
X:(d_0,\ldots,d_{r-1})\longmapsto Y:(d_{\pi(0)},\ldots,d_{\pi(r-1)}).
\]

输出可能不连续，但不改变元素值。它不是对象轴内任意排列；对象重排使用 ROUTE。

```python
x = torch.randn(2, 5, 3, 4)       # B,N,heads,d
heads = dsl.permute(x, (0,2,1,3))  # B,heads,N,d
```

## 9. RESHAPE：形状重解释

\[
X:(d_0,\ldots,d_{r-1})\longmapsto Y:(e_0,\ldots,e_{s-1}),\qquad\prod d_i=\prod e_j.
\]

`shape` 指定目标形状；最多一个 `-1` 表示推导维度。元素数不匹配或零维度下推导存在歧义时失败。按逻辑顺序重排分组；非连续输入可能发生内存复制，不执行转置。

```python
patches = torch.randn(2, 5, 3, 4)
flat = dsl.reshape(patches, (2, 5, 12))  # CNN patch 的 K×D 合并
```

## 10. REDUCE：显式轴归约

\[
Y_{ic}=\sum_k X_{ikc}\quad\text{是沿第二轴求和的示例。}
\]

`dim` 必须显式提供，是单个整数或非空、无重复的整数元组。`reducer` 是 `sum/mean/max`，max 只返回值，不返回位置。`keepdim=False` 移除归约轴；True 把归约轴设为 1。空轴求和为零；空轴 mean/max 报错。

```python
features = torch.randn(2, 5, 4)
pooled = dsl.reduce(features, dim=-2, reducer="sum")  # (2,4)，DeepSets
```

## 11. BROADCAST：广播扩展

\[
b\in\mathbb R^D\longmapsto B\in\mathbb R^{N\times D},\qquad B_{ic}=b_c.
\]

`shape` 的每个维度必须是显式非负整数；不接受 `-1`。按右对齐规则扩展缺失的前导轴和长度为 1 的轴。若希望插入中间轴，应先 RESHAPE。结果可能是 stride 为零的视图，不应就地写入。

```python
b = torch.randn(4)
bias = dsl.broadcast(b, (2,5,4))  # 共享线性层偏置
```

## 12. ADD：逐元素相加

\[
Y=X+Z,\quad X,Z,Y\in\mathbb R^S.
\]

`x` 与 `other` 必须完全同形状、同 dtype、同 device。不会自动把向量加到矩阵上；这种情况先 BROADCAST。用于 residual 或偏置。

```python
x, residual = torch.randn(2,5,4), torch.randn(2,5,4)
y = dsl.add(x, residual)
```

## 13. MUL：逐元素相乘

\[
Y=X\odot Z,\quad X,Z,Y\in\mathbb R^S.
\]

约束与 ADD 相同。标量也必须先转为同 device、dtype 的 Tensor 并显式广播。

```python
scores = torch.randn(2,5,5)
scale = dsl.broadcast(scores.new_tensor(0.5), scores.shape)
scaled = dsl.mul(scores, scale)  # attention 缩放
```

## 14. GATE：门控宏

\[
Y=\sigma(G)\odot V,\qquad\sigma(g)=\frac{1}{1+e^{-g}}.
\]

`logits` 是**未激活**的门值；`values` 是内容；二者完全同形状、dtype、device。这个宏展开为 `NONLINEAR(sigmoid) + MUL`。已激活的 LSTM 门直接传给 MUL，不要再次 GATE。

```python
logits, content = torch.randn(2,5,4), torch.randn(2,5,4)
y = dsl.gate(logits, content)  # GLU 的门控部分
```

## 15. CONCAT：沿轴拼接

\[
X:(B\cdots,N,D_1),\quad Z:(B\cdots,N,D_2)\longmapsto [X\;Z]:(B\cdots,N,D_1+D_2).
\]

`xs` 为非空 Tensor 序列，默认沿最后一轴拼接。要求 rank、非拼接轴形状、dtype、device 一致，不自动广播。

```python
xt, ht = torch.randn(2,4), torch.randn(2,3)
joined = dsl.concat((xt,ht))  # (2,7)，LSTM 的输入和旧状态
```

## 16. NONLINEAR：逐元素激活

\[
Y_i=\phi(X_i).
\]

支持 `relu`、`gelu`（erf 版本）、`silu`、`sigmoid`、`tanh` 和 `identity`，输出形状不变。softmax 是跨元素归一化，归属于 NORMALIZE。所有实现均不原地修改输入。

```python
x = torch.randn(2,5,4)
y = dsl.nonlinear(x, activation="gelu")  # Transformer FFN
```

## 17. NORM：特征标准化

当前版本明确为 **LayerNorm**：

\[
Y=\gamma\odot\frac{X-\mu}{\sqrt{v+\epsilon}}+\beta.
\]

`normalized_shape` 表示参与均值和方差计算的**尾部轴**，默认最后一轴。`weight` 是缩放参数、`bias` 是平移参数，必须精确匹配 `normalized_shape`；二者可省略。统计方差使用总体方差，即 `correction=0`。`eps` 为有限正数。输出与输入形状相同。

```python
x = torch.randn(2,5,4)
gamma, beta = torch.ones(4), torch.zeros(4)
y = dsl.norm(x, weight=gamma, bias=beta)  # Transformer LayerNorm
```

NORM 不是向量长度归一化，也不是 softmax。本版本不实现 BatchNorm/RMSNorm 的不同统计及状态规则，以免共享名字隐藏不同数学含义。

## 18. SCAN：旧时序接口（兼容保留）

\[
(s_t,y_t)=F(s_{t-1},x_t),\quad t=0,\ldots,T-1.
\]

- `step(state,x_t)` 返回二元组 **`(new_state,y_t)`**。
- `xs` 是包含时间轴的输入张量；`dim=0` 是输入时间轴，单步移除此轴。
- `initial_state` 是 Tensor，或非空的嵌套 Tensor 元组。LSTM 用 `(h0,c0)`。状态树结构、每个叶子的形状、dtype、device 必须在整个循环中保持不变。
- 每个 `y_t` 必须是固定 shape、dtype、device 的 Tensor。`out_dim=0` 指定在单步输出中插入时间轴的位置，与输入 `dim` 分开定义。
- 返回 **`(outputs,final_state)`**。函数不返回所有中间状态；如需要，把它们编码到 `y_t` 中。
- `reverse=True` 按最后一个输入到第一个输入执行，但输出序列仍与原输入位置对齐。最终状态是处理完索引 0 的状态。
- 空序列无法通过执行第一步推断输出，需要 `output_template`，即具有单步输出 shape/dtype/device 的 Tensor；此时返回空输出和原初始状态，不执行 `step`。非空时若提供模板，也会验证每一步输出与模板相同。
- `step` 必须不原地修改输入和状态；库不复制每步所有状态来防御用户回调中的原地修改。完整时间梯度被保留，因此长序列的反向传播内存仍随时间增长。

```python
xs = torch.randn(2,5,4)    # B,T,D
initial = torch.zeros(2,4)

def step(state, xt):
    new_state = dsl.add(state, xt)
    return new_state, new_state

ys, final = dsl.scan(step, xs, initial, dim=1, out_dim=1)
# ys: (2,5,4)，final: (2,4)
```

`examples/compositions.py` 中的 LSTM 使用门控状态元组；SSM 使用单状态，二者都通过同一个 SCAN 执行。

## 模型组合与扩展边界

`examples/compositions.py` 提供 7 个可执行函数：

| 例子 | 输入 → 输出 | 范围 |
|---|---|---|
| `mlp` | `(...,D) → (...,E)` | 两层仿射投影，中间 ReLU |
| `cnn2d` | `(B,C,H,W) → (B,O,Hout,Wout)` | ROUTE 展开二维窗口；groups=1、dilation=1，整数 stride/padding，无末尾激活 |
| `self_attention` | `(...,N,D) → (...,N,E)` | 单头、有可选 bool mask，无 dropout/输出投影 |
| `gcn` | `(...,N,D) → (...,N,E)` | 输入已归一化的邻接矩阵，无偏置 |
| `deepsets` | `(...,N,D) → rho 的输出` | 共享逐对象 phi、集合轴求和、rho |
| `lstm` | `(T,B,D) → (T,B,H),(hT,cT)` | 单层单向；门顺序 i,f,g,o，与 nn.LSTM 一致 |
| `ssm` | `(T,...,D) → (T,...,E),hT` | 已离散化的线性时不变 SSM |

网页有时采用 `keepdim=True` 表示集合汇聚后的单个对象；Python 示例默认移除集合轴。需要 `(B,1,H)` 时显式传 `keepdim=True`，含义一致。

使用 `OPERATIONS` 可以把网页/未来图节点里的大写 op 名称映射到真实函数：

```python
spec = dsl.get_operation("PROJECT")
y = spec.function(x, w)
print(spec.inputs, spec.output, spec.kind)
```

注册表对外只读，包含名称、类别、函数、输入输出描述，以及 `primitive/macro/control` 标识。它还不是随机架构生成器：暂不包含图调度、参数自动生成、符号形状推导、图序列化或自动成本估计。当前网页的描述性 JSON 并不能直接当作可执行模型。后续这些能力可建立在本函数契约之上。

## PyTorch 官方接口参考

- [torch.index_select](https://docs.pytorch.org/docs/stable/generated/torch.index_select.html)：ROUTE 的底层索引操作。
- [functional.softmax](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)：NORMALIZE 的底层归一化操作。
- [functional.layer_norm](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.layer_norm.html)：NORM 的底层特征标准化。

## 19. LOOP 家族：REPEAT 与扩展 SCAN

REPEAT 提供有上限的重复执行，SCAN 同时支持 Sequence 和 Window 域。两者的固定格式、参数共享、carry、输出组织以及旧接口迁移见 [loop-family.md](loop-family.md)。
