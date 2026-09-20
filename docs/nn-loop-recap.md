# 加入 LOOP 后的 nn.Module 重新概括

结论：新增有上限的 REPEAT，并将已有 SCAN 扩展为空间窗口和时间序列两个域。它们同属 LOOP 类别。具体原语为 **17 个计算/结构原语 + 2 个控制原语 = 19 个**。本轮主要改善模型结构表达，模块数值覆盖率没有因增加控制结构而自动提高。

环境：PyTorch 2.14.0，Matrix DSL 0.2.0，CPU / float64，随机种子 20260920。

## 一、重新执行的覆盖结果

| 证据 | 结果 |
| --- | --- |
| 公开 torch.nn Module 类 | 163 类全部有记录 |
| 已有完整数值候选的基准类 | 125 类在指定配置下前向、输入/参数梯度通过 |
| 尚未完成分解 | 30 类；原生参考前向执行通过，但不算 DSL 等价 |
| 结构/抽象类 | 8 类，不是独立数值运算 |
| 含训练模式、额外配置和循环组合 | 191 项：131 等价、13 仅训练无状态计算、39 未完成、8 非数值 |
| 独立展开的 Python 文件 | 131 项等价、13 项训练无状态计算；运行时禁止原生 forward 回退 |
| 单元测试 | 130 通过，1 因无 CUDA 跳过 |

与加入 LOOP 前相比，公开类计数仍为 125 / 30 / 8。131 项等价比原来的 127 项多出的 4 项，是新增控制结构组合探针，不是新增 4 个公开 nn.Module 类。

## 二、模型结构如何概括

| 模型类别 | 表示 | 本轮实际改写 |
| --- | --- | --- |
| Linear / MLP | PROJECT → NONLINEAR → PROJECT | 保持直接原语网络 |
| 普通 CNN | LOOP.SCAN(Window, convolution kernel) | Conv1d/2d/3d 及 Lazy 版本，共 6 类 |
| 转置卷积 | ROUTE + PROJECT + CONCAT 等 | 保留现有几何分解，不把反向散射误称为普通窗口读取 |
| 循环 CNN | LOOP.REPEAT(卷积块)，块内部 LOOP.SCAN | 共享参数重复 3 次，max_steps=6 |
| 单步 RNN / GRU / LSTM cell | PROJECT + ADD + MUL/GATE + NONLINEAR | 单步子图无需循环 |
| RNN / GRU / LSTM 序列 | LOOP.SCAN(Sequence, cell, carry) | 3 类适配器改用显式 Sequence 域 |
| 单层 Transformer / attention | COMPARE + NORMALIZE + MIX，配合投影、残差、NORM | 作为循环体的显式原语子图 |
| 多层 Transformer | LOOP.REPEAT(block, PerStep(parameters)) | TransformerEncoder、Decoder、完整 Transformer，共 3 类 |
| Looped Transformer | LOOP.REPEAT(block, Shared(parameters)) | 同一组权重重复 3 次，max_steps=6 |
| Pool / 图消息汇聚 / 集合归约 | ROUTE + REDUCE / AGGREGATE / MIX | 循环不是所有汇聚操作的必需表示 |
| 归一化、激活、损失、布局操作 | 直接原语组合 | 结构重复外提后，内部数值计算仍需要这些原语 |

原语图中的循环体只保存一次，steps/domain 决定执行次数。Shared 与 PerStep 都保留可训练参数引用；没有按迭代次数复制权重快照。Window 定义窗口、步长、膨胀、补零与输出网格。

## 三、新增控制结构证据

| 用例 | 结果 | 显式控制节点（含嵌套 body） |
| --- | --- | --- |
| [TransformerEncoder:repeat3](code/variants/TransformerEncoder__repeat3.py) | 指定配置数值/梯度等价 | {'REPEAT': 1} |
| [TransformerDecoder:repeat3](code/variants/TransformerDecoder__repeat3.py) | 指定配置数值/梯度等价 | {'REPEAT': 1} |
| [LoopedTransformer:shared3](code/variants/LoopedTransformer__shared3.py) | 指定配置数值/梯度等价 | {'REPEAT': 1} |
| [LoopedCNN:shared3](code/variants/LoopedCNN__shared3.py) | 指定配置数值/梯度等价 | {'REPEAT': 1, 'SCAN': 1} |

额外测试覆盖：REPEAT 的上限、零轮与输出堆叠；共享/独立参数梯度；Window 的非对称 padding、stride、dilation、same；Sequence 的逆序输出与 carry；REPEAT 嵌套 SCAN；故意将 torch.sin 藏入循环体时审计必须失败。

## 四、仍然缺少什么

- 随机性：Dropout/RReLU 的训练分支、FractionalMaxPool 的随机区域，不会因为加入循环就自动获得 RNG 语义。
- 数值模式：部分激活与损失仍缺 log、exp、倒数、幂/根、阈值或选择的具体分解。循环可以重复已定义的计算，但不能将未实现的算子直接当作已实现。
- 状态副作用：13 个归一化训练探针的输出和梯度通过，running_mean/running_var 等持久更新仍未实现；不计完整训练模块等价。
- 特殊输出与执行语义：整数 argmax、稀疏/分布式行为、动态形状及所有参数选项尚未完整覆盖。
- 运行开销：窗口 SCAN 的后端目前逐窗口执行，表达清晰不等于优于 PyTorch 专用内核；本轮不是性能基准。

## 五、全部 163 个公开类

每一行只表示列出的 fixture。未实现列表不是数学上不可表达性的证明。

| 类 | 概括 | 结果 |
| --- | --- | --- |
| [AdaptiveAvgPool1d](code/AdaptiveAvgPool1d.py) | 窗口/集合汇聚：ROUTE + REDUCE | 指定配置数值/梯度等价 |
| [AdaptiveAvgPool2d](code/AdaptiveAvgPool2d.py) | 窗口/集合汇聚：ROUTE + REDUCE | 指定配置数值/梯度等价 |
| [AdaptiveAvgPool3d](code/AdaptiveAvgPool3d.py) | 窗口/集合汇聚：ROUTE + REDUCE | 指定配置数值/梯度等价 |
| [AdaptiveLogSoftmaxWithLoss](code/unsupported/AdaptiveLogSoftmaxWithLoss.py) | 未完成的算子或效应契约 | 尚无完整分解 |
| [AdaptiveMaxPool1d](code/AdaptiveMaxPool1d.py) | 窗口/集合汇聚：ROUTE + REDUCE | 指定配置数值/梯度等价 |
| [AdaptiveMaxPool2d](code/AdaptiveMaxPool2d.py) | 窗口/集合汇聚：ROUTE + REDUCE | 指定配置数值/梯度等价 |
| [AdaptiveMaxPool3d](code/AdaptiveMaxPool3d.py) | 窗口/集合汇聚：ROUTE + REDUCE | 指定配置数值/梯度等价 |
| [AlphaDropout](code/AlphaDropout.py) | 仅推理分支；训练随机性未覆盖 | 指定配置数值/梯度等价 |
| [AvgPool1d](code/AvgPool1d.py) | 窗口/集合汇聚：ROUTE + REDUCE | 指定配置数值/梯度等价 |
| [AvgPool2d](code/AvgPool2d.py) | 窗口/集合汇聚：ROUTE + REDUCE | 指定配置数值/梯度等价 |
| [AvgPool3d](code/AvgPool3d.py) | 窗口/集合汇聚：ROUTE + REDUCE | 指定配置数值/梯度等价 |
| [BCELoss](code/unsupported/BCELoss.py) | 未完成的算子或效应契约 | 尚无完整分解 |
| [BCEWithLogitsLoss](code/unsupported/BCEWithLogitsLoss.py) | 未完成的算子或效应契约 | 尚无完整分解 |
| [BatchNorm1d](code/BatchNorm1d.py) | 归一化：NORM 与轴变换 | 指定配置数值/梯度等价 |
| [BatchNorm2d](code/BatchNorm2d.py) | 归一化：NORM 与轴变换 | 指定配置数值/梯度等价 |
| [BatchNorm3d](code/BatchNorm3d.py) | 归一化：NORM 与轴变换 | 指定配置数值/梯度等价 |
| [Bilinear](code/Bilinear.py) | 直接计算原语组合 | 指定配置数值/梯度等价 |
| [CELU](code/unsupported/CELU.py) | 未完成的算子或效应契约 | 尚无完整分解 |
| [CTCLoss](code/unsupported/CTCLoss.py) | 未完成的算子或效应契约 | 尚无完整分解 |
| [ChannelShuffle](code/ChannelShuffle.py) | 布局 / 索引 / 重采样 | 指定配置数值/梯度等价 |
| [CircularPad1d](code/CircularPad1d.py) | 布局 / 索引 / 重采样 | 指定配置数值/梯度等价 |
| [CircularPad2d](code/CircularPad2d.py) | 布局 / 索引 / 重采样 | 指定配置数值/梯度等价 |
| [CircularPad3d](code/CircularPad3d.py) | 布局 / 索引 / 重采样 | 指定配置数值/梯度等价 |
| [ConstantPad1d](code/ConstantPad1d.py) | 布局 / 索引 / 重采样 | 指定配置数值/梯度等价 |
| [ConstantPad2d](code/ConstantPad2d.py) | 布局 / 索引 / 重采样 | 指定配置数值/梯度等价 |
| [ConstantPad3d](code/ConstantPad3d.py) | 布局 / 索引 / 重采样 | 指定配置数值/梯度等价 |
| [Container](code/structural/Container.py) | 宿主结构 / 参数容器 | 结构/抽象类 |
| [Conv1d](code/Conv1d.py) | 窗口扫描：LOOP.SCAN | 指定配置数值/梯度等价 |
| [Conv2d](code/Conv2d.py) | 窗口扫描：LOOP.SCAN | 指定配置数值/梯度等价 |
| [Conv3d](code/Conv3d.py) | 窗口扫描：LOOP.SCAN | 指定配置数值/梯度等价 |
| [ConvTranspose1d](code/ConvTranspose1d.py) | 转置卷积：ROUTE + PROJECT | 指定配置数值/梯度等价 |
| [ConvTranspose2d](code/ConvTranspose2d.py) | 转置卷积：ROUTE + PROJECT | 指定配置数值/梯度等价 |
| [ConvTranspose3d](code/ConvTranspose3d.py) | 转置卷积：ROUTE + PROJECT | 指定配置数值/梯度等价 |
| [CosineEmbeddingLoss](code/unsupported/CosineEmbeddingLoss.py) | 未完成的算子或效应契约 | 尚无完整分解 |
| [CosineSimilarity](code/unsupported/CosineSimilarity.py) | 未完成的算子或效应契约 | 尚无完整分解 |
| [CrossEntropyLoss](code/unsupported/CrossEntropyLoss.py) | 未完成的算子或效应契约 | 尚无完整分解 |
| [CrossMapLRN2d](code/unsupported/CrossMapLRN2d.py) | 未完成的算子或效应契约 | 尚无完整分解 |
| [DataParallel](code/DataParallel.py) | 直接计算原语组合 | 指定配置数值/梯度等价 |
| [Dropout](code/Dropout.py) | 仅推理分支；训练随机性未覆盖 | 指定配置数值/梯度等价 |
| [Dropout1d](code/Dropout1d.py) | 仅推理分支；训练随机性未覆盖 | 指定配置数值/梯度等价 |
| [Dropout2d](code/Dropout2d.py) | 仅推理分支；训练随机性未覆盖 | 指定配置数值/梯度等价 |
| [Dropout3d](code/Dropout3d.py) | 仅推理分支；训练随机性未覆盖 | 指定配置数值/梯度等价 |
| [ELU](code/unsupported/ELU.py) | 未完成的算子或效应契约 | 尚无完整分解 |
| [Embedding](code/Embedding.py) | 查表：ROUTE / AGGREGATE | 指定配置数值/梯度等价 |
| [EmbeddingBag](code/EmbeddingBag.py) | 查表：ROUTE / AGGREGATE | 指定配置数值/梯度等价 |
| [FeatureAlphaDropout](code/FeatureAlphaDropout.py) | 仅推理分支；训练随机性未覆盖 | 指定配置数值/梯度等价 |
| [Flatten](code/Flatten.py) | 布局 / 索引 / 重采样 | 指定配置数值/梯度等价 |
| [Fold](code/Fold.py) | 布局 / 索引 / 重采样 | 指定配置数值/梯度等价 |
| [FractionalMaxPool2d](code/unsupported/FractionalMaxPool2d.py) | 未完成的算子或效应契约 | 尚无完整分解 |
| [FractionalMaxPool3d](code/unsupported/FractionalMaxPool3d.py) | 未完成的算子或效应契约 | 尚无完整分解 |
| [GELU](code/GELU.py) | 直接计算原语组合 | 指定配置数值/梯度等价 |
| [GLU](code/GLU.py) | 直接计算原语组合 | 指定配置数值/梯度等价 |
| [GRU](code/GRU.py) | 时序扫描：LOOP.SCAN | 指定配置数值/梯度等价 |
| [GRUCell](code/GRUCell.py) | 单步状态更新：投影 + 门控 | 指定配置数值/梯度等价 |
| [GaussianNLLLoss](code/unsupported/GaussianNLLLoss.py) | 未完成的算子或效应契约 | 尚无完整分解 |
| [GroupNorm](code/GroupNorm.py) | 归一化：NORM 与轴变换 | 指定配置数值/梯度等价 |
| [Hardshrink](code/unsupported/Hardshrink.py) | 未完成的算子或效应契约 | 尚无完整分解 |
| [Hardsigmoid](code/Hardsigmoid.py) | 直接计算原语组合 | 指定配置数值/梯度等价 |
| [Hardswish](code/Hardswish.py) | 直接计算原语组合 | 指定配置数值/梯度等价 |
| [Hardtanh](code/Hardtanh.py) | 直接计算原语组合 | 指定配置数值/梯度等价 |
| [HingeEmbeddingLoss](code/HingeEmbeddingLoss.py) | 损失：逐元素运算 + REDUCE / ROUTE | 指定配置数值/梯度等价 |
| [HuberLoss](code/HuberLoss.py) | 损失：逐元素运算 + REDUCE / ROUTE | 指定配置数值/梯度等价 |
| [Identity](code/Identity.py) | 直接计算原语组合 | 指定配置数值/梯度等价 |
| [InstanceNorm1d](code/InstanceNorm1d.py) | 归一化：NORM 与轴变换 | 指定配置数值/梯度等价 |
| [InstanceNorm2d](code/InstanceNorm2d.py) | 归一化：NORM 与轴变换 | 指定配置数值/梯度等价 |
| [InstanceNorm3d](code/InstanceNorm3d.py) | 归一化：NORM 与轴变换 | 指定配置数值/梯度等价 |
| [KLDivLoss](code/unsupported/KLDivLoss.py) | 未完成的算子或效应契约 | 尚无完整分解 |
| [L1Loss](code/L1Loss.py) | 损失：逐元素运算 + REDUCE / ROUTE | 指定配置数值/梯度等价 |
| [LPPool1d](code/LPPool1d.py) | 窗口/集合汇聚：ROUTE + REDUCE | 指定配置数值/梯度等价 |
| [LPPool2d](code/LPPool2d.py) | 窗口/集合汇聚：ROUTE + REDUCE | 指定配置数值/梯度等价 |
| [LPPool3d](code/LPPool3d.py) | 窗口/集合汇聚：ROUTE + REDUCE | 指定配置数值/梯度等价 |
| [LSTM](code/LSTM.py) | 时序扫描：LOOP.SCAN | 指定配置数值/梯度等价 |
| [LSTMCell](code/LSTMCell.py) | 单步状态更新：投影 + 门控 | 指定配置数值/梯度等价 |
| [LayerNorm](code/LayerNorm.py) | 归一化：NORM 与轴变换 | 指定配置数值/梯度等价 |
| [LazyBatchNorm1d](code/LazyBatchNorm1d.py) | 归一化：NORM 与轴变换 | 指定配置数值/梯度等价 |
| [LazyBatchNorm2d](code/LazyBatchNorm2d.py) | 归一化：NORM 与轴变换 | 指定配置数值/梯度等价 |
| [LazyBatchNorm3d](code/LazyBatchNorm3d.py) | 归一化：NORM 与轴变换 | 指定配置数值/梯度等价 |
| [LazyConv1d](code/LazyConv1d.py) | 窗口扫描：LOOP.SCAN | 指定配置数值/梯度等价 |
| [LazyConv2d](code/LazyConv2d.py) | 窗口扫描：LOOP.SCAN | 指定配置数值/梯度等价 |
| [LazyConv3d](code/LazyConv3d.py) | 窗口扫描：LOOP.SCAN | 指定配置数值/梯度等价 |
| [LazyConvTranspose1d](code/LazyConvTranspose1d.py) | 转置卷积：ROUTE + PROJECT | 指定配置数值/梯度等价 |
| [LazyConvTranspose2d](code/LazyConvTranspose2d.py) | 转置卷积：ROUTE + PROJECT | 指定配置数值/梯度等价 |
| [LazyConvTranspose3d](code/LazyConvTranspose3d.py) | 转置卷积：ROUTE + PROJECT | 指定配置数值/梯度等价 |
| [LazyInstanceNorm1d](code/LazyInstanceNorm1d.py) | 归一化：NORM 与轴变换 | 指定配置数值/梯度等价 |
| [LazyInstanceNorm2d](code/LazyInstanceNorm2d.py) | 归一化：NORM 与轴变换 | 指定配置数值/梯度等价 |
| [LazyInstanceNorm3d](code/LazyInstanceNorm3d.py) | 归一化：NORM 与轴变换 | 指定配置数值/梯度等价 |
| [LazyLinear](code/LazyLinear.py) | 直接计算原语组合 | 指定配置数值/梯度等价 |
| [LeakyReLU](code/LeakyReLU.py) | 直接计算原语组合 | 指定配置数值/梯度等价 |
| [Linear](code/Linear.py) | 直接计算原语组合 | 指定配置数值/梯度等价 |
| [LinearCrossEntropyLoss](code/unsupported/LinearCrossEntropyLoss.py) | 未完成的算子或效应契约 | 尚无完整分解 |
| [LocalResponseNorm](code/unsupported/LocalResponseNorm.py) | 未完成的算子或效应契约 | 尚无完整分解 |
| [LogSigmoid](code/unsupported/LogSigmoid.py) | 未完成的算子或效应契约 | 尚无完整分解 |
| [LogSoftmax](code/unsupported/LogSoftmax.py) | 未完成的算子或效应契约 | 尚无完整分解 |
| [MSELoss](code/MSELoss.py) | 损失：逐元素运算 + REDUCE / ROUTE | 指定配置数值/梯度等价 |
| [MarginRankingLoss](code/MarginRankingLoss.py) | 损失：逐元素运算 + REDUCE / ROUTE | 指定配置数值/梯度等价 |
| [MaxPool1d](code/MaxPool1d.py) | 窗口/集合汇聚：ROUTE + REDUCE | 指定配置数值/梯度等价 |
| [MaxPool2d](code/MaxPool2d.py) | 窗口/集合汇聚：ROUTE + REDUCE | 指定配置数值/梯度等价 |
| [MaxPool3d](code/MaxPool3d.py) | 窗口/集合汇聚：ROUTE + REDUCE | 指定配置数值/梯度等价 |
| [MaxUnpool1d](code/MaxUnpool1d.py) | 直接计算原语组合 | 指定配置数值/梯度等价 |
| [MaxUnpool2d](code/MaxUnpool2d.py) | 直接计算原语组合 | 指定配置数值/梯度等价 |
| [MaxUnpool3d](code/MaxUnpool3d.py) | 直接计算原语组合 | 指定配置数值/梯度等价 |
| [Mish](code/unsupported/Mish.py) | 未完成的算子或效应契约 | 尚无完整分解 |
| [Module](code/structural/Module.py) | 宿主结构 / 参数容器 | 结构/抽象类 |
| [ModuleDict](code/structural/ModuleDict.py) | 宿主结构 / 参数容器 | 结构/抽象类 |
| [ModuleList](code/structural/ModuleList.py) | 宿主结构 / 参数容器 | 结构/抽象类 |
| [MultiLabelMarginLoss](code/MultiLabelMarginLoss.py) | 损失：逐元素运算 + REDUCE / ROUTE | 指定配置数值/梯度等价 |
| [MultiLabelSoftMarginLoss](code/unsupported/MultiLabelSoftMarginLoss.py) | 未完成的算子或效应契约 | 尚无完整分解 |
| [MultiMarginLoss](code/MultiMarginLoss.py) | 损失：逐元素运算 + REDUCE / ROUTE | 指定配置数值/梯度等价 |
| [MultiheadAttention](code/MultiheadAttention.py) | 关系计算：COMPARE + NORMALIZE + MIX | 指定配置数值/梯度等价 |
| [NLLLoss](code/NLLLoss.py) | 损失：逐元素运算 + REDUCE / ROUTE | 指定配置数值/梯度等价 |
| [NLLLoss2d](code/NLLLoss2d.py) | 损失：逐元素运算 + REDUCE / ROUTE | 指定配置数值/梯度等价 |
| [PReLU](code/PReLU.py) | 直接计算原语组合 | 指定配置数值/梯度等价 |
| [PairwiseDistance](code/unsupported/PairwiseDistance.py) | 未完成的算子或效应契约 | 尚无完整分解 |
| [ParameterDict](code/structural/ParameterDict.py) | 宿主结构 / 参数容器 | 结构/抽象类 |
| [ParameterList](code/structural/ParameterList.py) | 宿主结构 / 参数容器 | 结构/抽象类 |
| [PixelShuffle](code/PixelShuffle.py) | 布局 / 索引 / 重采样 | 指定配置数值/梯度等价 |
| [PixelUnshuffle](code/PixelUnshuffle.py) | 直接计算原语组合 | 指定配置数值/梯度等价 |
| [PoissonNLLLoss](code/unsupported/PoissonNLLLoss.py) | 未完成的算子或效应契约 | 尚无完整分解 |
| [RMSNorm](code/RMSNorm.py) | 归一化：NORM 与轴变换 | 指定配置数值/梯度等价 |
| [RNN](code/RNN.py) | 时序扫描：LOOP.SCAN | 指定配置数值/梯度等价 |
| [RNNBase](code/structural/RNNBase.py) | 宿主结构 / 参数容器 | 结构/抽象类 |
| [RNNCell](code/RNNCell.py) | 单步状态更新：投影 + 门控 | 指定配置数值/梯度等价 |
| [RNNCellBase](code/structural/RNNCellBase.py) | 宿主结构 / 参数容器 | 结构/抽象类 |
| [RReLU](code/RReLU.py) | 仅推理分支；训练随机性未覆盖 | 指定配置数值/梯度等价 |
| [ReLU](code/ReLU.py) | 直接计算原语组合 | 指定配置数值/梯度等价 |
| [ReLU6](code/ReLU6.py) | 直接计算原语组合 | 指定配置数值/梯度等价 |
| [ReflectionPad1d](code/ReflectionPad1d.py) | 布局 / 索引 / 重采样 | 指定配置数值/梯度等价 |
| [ReflectionPad2d](code/ReflectionPad2d.py) | 布局 / 索引 / 重采样 | 指定配置数值/梯度等价 |
| [ReflectionPad3d](code/ReflectionPad3d.py) | 布局 / 索引 / 重采样 | 指定配置数值/梯度等价 |
| [ReplicationPad1d](code/ReplicationPad1d.py) | 布局 / 索引 / 重采样 | 指定配置数值/梯度等价 |
| [ReplicationPad2d](code/ReplicationPad2d.py) | 布局 / 索引 / 重采样 | 指定配置数值/梯度等价 |
| [ReplicationPad3d](code/ReplicationPad3d.py) | 布局 / 索引 / 重采样 | 指定配置数值/梯度等价 |
| [SELU](code/unsupported/SELU.py) | 未完成的算子或效应契约 | 尚无完整分解 |
| [Sequential](code/Sequential.py) | 直接计算原语组合 | 指定配置数值/梯度等价 |
| [SiLU](code/SiLU.py) | 直接计算原语组合 | 指定配置数值/梯度等价 |
| [Sigmoid](code/Sigmoid.py) | 直接计算原语组合 | 指定配置数值/梯度等价 |
| [SmoothL1Loss](code/SmoothL1Loss.py) | 损失：逐元素运算 + REDUCE / ROUTE | 指定配置数值/梯度等价 |
| [SoftMarginLoss](code/unsupported/SoftMarginLoss.py) | 未完成的算子或效应契约 | 尚无完整分解 |
| [Softmax](code/Softmax.py) | 直接计算原语组合 | 指定配置数值/梯度等价 |
| [Softmax2d](code/Softmax2d.py) | 直接计算原语组合 | 指定配置数值/梯度等价 |
| [Softmin](code/Softmin.py) | 直接计算原语组合 | 指定配置数值/梯度等价 |
| [Softplus](code/unsupported/Softplus.py) | 未完成的算子或效应契约 | 尚无完整分解 |
| [Softshrink](code/Softshrink.py) | 直接计算原语组合 | 指定配置数值/梯度等价 |
| [Softsign](code/unsupported/Softsign.py) | 未完成的算子或效应契约 | 尚无完整分解 |
| [SyncBatchNorm](code/SyncBatchNorm.py) | 归一化：NORM 与轴变换 | 指定配置数值/梯度等价 |
| [Tanh](code/Tanh.py) | 直接计算原语组合 | 指定配置数值/梯度等价 |
| [Tanhshrink](code/Tanhshrink.py) | 直接计算原语组合 | 指定配置数值/梯度等价 |
| [Threshold](code/unsupported/Threshold.py) | 未完成的算子或效应契约 | 尚无完整分解 |
| [Transformer](code/Transformer.py) | 层级重复：LOOP.REPEAT | 指定配置数值/梯度等价 |
| [TransformerDecoder](code/TransformerDecoder.py) | 层级重复：LOOP.REPEAT | 指定配置数值/梯度等价 |
| [TransformerDecoderLayer](code/TransformerDecoderLayer.py) | 单层块：注意力 + 残差 + NORM | 指定配置数值/梯度等价 |
| [TransformerEncoder](code/TransformerEncoder.py) | 层级重复：LOOP.REPEAT | 指定配置数值/梯度等价 |
| [TransformerEncoderLayer](code/TransformerEncoderLayer.py) | 单层块：注意力 + 残差 + NORM | 指定配置数值/梯度等价 |
| [TripletMarginLoss](code/unsupported/TripletMarginLoss.py) | 未完成的算子或效应契约 | 尚无完整分解 |
| [TripletMarginWithDistanceLoss](code/unsupported/TripletMarginWithDistanceLoss.py) | 未完成的算子或效应契约 | 尚无完整分解 |
| [Unflatten](code/Unflatten.py) | 直接计算原语组合 | 指定配置数值/梯度等价 |
| [Unfold](code/Unfold.py) | 直接计算原语组合 | 指定配置数值/梯度等价 |
| [Upsample](code/Upsample.py) | 布局 / 索引 / 重采样 | 指定配置数值/梯度等价 |
| [UpsamplingBilinear2d](code/UpsamplingBilinear2d.py) | 布局 / 索引 / 重采样 | 指定配置数值/梯度等价 |
| [UpsamplingNearest2d](code/UpsamplingNearest2d.py) | 布局 / 索引 / 重采样 | 指定配置数值/梯度等价 |
| [ZeroPad1d](code/ZeroPad1d.py) | 布局 / 索引 / 重采样 | 指定配置数值/梯度等价 |
| [ZeroPad2d](code/ZeroPad2d.py) | 布局 / 索引 / 重采样 | 指定配置数值/梯度等价 |
| [ZeroPad3d](code/ZeroPad3d.py) | 布局 / 索引 / 重采样 | 指定配置数值/梯度等价 |

## 六、复验入口

```sh
.venv/bin/python -m smoke.loop_recap
.venv/bin/python -m dsl_nn.validate_expanded
.venv/bin/python -m pytest -q
```

完整数值、梯度误差和实际原语调用：[JSON](nn-loop-smoke.json)。逐类机器清单：[CSV](nn-loop-coverage.csv)。固定格式：[LOOP API](loop-family.md)。展开源码：[审查索引](index.html)。

范围限于当前安装版本的 torch.nn 公开命名空间，以及上述 4 个明确构造的组合；不等同于整个 PyTorch 生态或所有模型配置。
