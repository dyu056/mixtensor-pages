# 验证记录

2026-09-20，本地实际运行结果。

- 环境：Python 3.12.7，PyTorch 2.14.0，macOS arm64。
- `python -m pytest -q`：**79 passed, 1 skipped**，约 38 秒。
- CUDA 测试因本机不可用而跳过；Apple MPS 冒烟测试通过。未运行远程 GPU 测试。
- `python -m examples.demo`：7 类模型均运行成功；attention 参数获得梯度。
- Wheel 构建成功，已在新 Python 进程中从 wheel 导入而非源目录导入，并检查 PROJECT 前向/反向及 18 操作注册表。

覆盖内容：

- 与 PyTorch Conv2d、scaled_dot_product_attention、LSTM 对比前向结果及输入/权重/初始状态梯度。
- SSM 与矩阵幂闭式展开对比数值和梯度。
- DeepSets 对对象排列不变，GCN 对对象及图的同步排列等变。
- PROJECT、MIX、COMPARE、ROUTE、NORMALIZE、AGGREGATE、SHIFT、逐元素运算、激活、NORM、SCAN 的双精度有限差分梯度检查。
- 轴转换、广播的梯度累加、重复索引、补零、全屏蔽 softmax、空序列、逆向 SCAN、元组状态和输入不被修改。
- 32 组无效形状/参数/轴/边界输入明确报错。
- 注册表的 18 个名称与本目录 `data.js` 的网页原语一致。

环境中未安装 NumPy，PyTorch 初始化打印了一条 NumPy bridge 不可用的提示；本库未使用 NumPy，以上测试及示例均成功。这不是 NumPy 互操作能力的验证。

此记录仅代表上述版本与测试覆盖；不声称所有 PyTorch 版本、所有设备、compile/export 或所有模型变体都已验证。
