# Code Module Definition

前端入口：`index.html`。所有页面、样式、脚本、KaTeX 和代码快照均位于本文件夹。

## 本地打开

可以直接用浏览器打开 `index.html`，也可以在项目根目录运行：

```sh
python3 -m http.server 8765 --directory docs
```

访问 http://localhost:8765/ 。部署静态网站时，将 `docs/` 设为发布目录即可。

## 页面

- `nn-modules.html`：所有当前公开 nn.Module 的原语平替总览，包含原语组合、配置、输入输出、数值验证和缺口；可切换显示变体。

- `index.html`：模块定义源码浏览器，包含搜索、验证状态筛选、代码标签页、复制及下载。
- `primitives.html`：19 个原语，包括 LOOP.REPEAT 和 LOOP.SCAN。
- `models.html`：模型的原语组合图。

## 同步代码

真实源代码仍在项目的 `dsl_nn/expanded/`。修改或重新生成模块后，在项目根目录运行：

```sh
python3 scripts/build_docs.py
```

先运行 `python -m dsl_nn.validate_expanded` 更新展开源码的验证记录，再同步网页。

生成器仅解析源码，不执行模块；从实际代码、manifest、registry 和 smoke 报告生成 `module-data.js`，并将源码复制到 `code/`。页面显示原始行号和完整文件 SHA 摘要，便于核对。

`code/` 是供浏览器审查和下载的快照；单独下载其中一个文件不等于安装可运行的 Python 库，运行仍需完整项目及依赖。

## 覆盖口径

当前目录包含 163 个公开类条目和 19 个配置或组合变体。125 个公开类在列出的配置下通过数值对比；30 个尚未实现；8 个属于结构或抽象类。未实现类明确显示 NotImplementedError，不将其计为实现覆盖。输出形状来自参考模型的 smoke 记录。

将新模块加入上游 manifest 并重新生成数据，即可扩展代码浏览器；UI 与生成数据分离，便于后续接入架构生成器。

## Graph IR

独立示例入口：[Graph IR 图与矩阵](graph-ir/index.html)。包含 MLP、attention、两种参数共享方式的 REPEAT，以及序列 / 窗口 SCAN。节点可点击查看端口，循环体可展开，并能下载完整 JSON。

接口说明见 [Graph IR v1](graph-ir.md)。在项目环境中运行 `python scripts/build_graph_ir_examples.py` 更新图定义、矩阵和页面。

## 随机生成器 V0

[接口说明](random-generator.md) · [第一批训练结果](random-models/results.html) · [随机模型图与矩阵](random-models/index.html)。

## 图元模型与搜索评测

[交互实验报告](meta-search/index.html) · [实现与实验协议](meta-model.md)：架构图编码、score / 完成概率预测、数据采集、等预算搜索和独立测试。

## Adding benchmark

[任务、数据与评测协议](adding-benchmark.md)：上游 Adding 数据配方、元模型拟合隔离、GNN/ridge/random 等查询比较和最终独立测试。

已完成的 [Adding v1 交互报告](adding-benchmark/index.html) 包含 4/8/16 次查询预算、全部基线、独立重训曲线和查询明细。

## 结构编码元模型 V2.1

[架构与代码接口](relation-meta-v2-1.md)：拓扑对齐、稀疏依赖编码、对角 Q/K/V 关系建模、中间矩阵导出与 no-sparse 对照。

[原 V2 协议](relation-meta-v2.md) 与 [Adding / Jena benchmark 结果](../reports/relation-v2/RESULTS.md) 单独保留；V2.1 目前完成正确性 smoke tests，尚未给出任务性能结论。

## 在线版本

[GitHub Pages](https://dyu056.github.io/mixtensor-pages/) · [nn.Module 原语平替](https://dyu056.github.io/mixtensor-pages/nn-modules.html)。网页发布到独立公开仓库 `dyu056/mixtensor-pages` 的 `main` 分支 `/docs` 目录。
