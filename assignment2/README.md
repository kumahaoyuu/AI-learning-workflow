# Assignment 2 手写数字分类

完整 MNIST 测试集结果为 **Naive Bayes 91.60%、小型贝叶斯网络 92.38%**。正式报告请阅读 [report.md](report.md)，数据处理、两个模型的方法、实验比较和误差分析已经整合在同一篇报告中。

本目录同时包含模型代码、训练入口和复核记录。`history/` 中的旧报告仅供追溯，不是正式报告的分章内容，阅读当前报告无需逐个打开旧文件。

## 安装和复现

需要 Python 3.10 或更新版本。将本目录作为课程仓库的 `assignment2/`，课程原始数据放在平级 `assignment2-data/mnist/`，不要提交原始 MNIST。

```console
python -m pip install -r requirements.txt
python main.py --data-dir ../assignment2-data/mnist
python verify_results.py
```

当前 Codex 工作目录中，数据在 `work/data/assignment2-data/mnist`，可从工作目录运行以下命令。

```console
python outputs/assignment2-target90/main.py --data-dir work/data/assignment2-data/mnist
python outputs/assignment2-target90/verify_results.py
```

两个模型类严格保持 `__init__(self, alpha: float = 1.0)` 默认值。验证选择的平滑系数为 `0.1`，因此 `main.py` 默认显式传入 `0.1`，报告对应这一配置。类构造函数省略参数时仍为 `1.0`。

训练入口会先核对冻结核心源码与验证记录，再校验四个数据文件、使用完整 60,000 张训练图像拟合，并评估全部 10,000 张测试图像。可用 `--output-dir independent-run` 将重跑结果另存。`report.md` 是当前配置的整理稿，不会被程序自动改写。

## 本版改动

- NB 采用逐图矩去倾斜、包围框双线性拉伸至 24×24、在 28×28 画布上对齐灰度质心，再进行八档像素灰度离散化。核心仍为 784 个离散像素的朴素贝叶斯。
- BN 采用九个区域的梯度主轴角、去倾斜包围框宽高比和上下孔洞比例。共十二个特征、每项五个状态，使用八条手工特征边；Y 为根、总共十三个节点、每项至多两个父节点，无隐藏变量。
- 两个核心模型都只导入 NumPy，自行实现平滑、计数和对数概率预测。

BN 的方向特征用 24×24 双线性采样图像的九个 8×8 区域；孔洞在去倾斜的 28×28 阈值图上计算，再映射到 21×21 掩码。两者的定义在报告和代码中分别说明。

## 文件与结果

| 文件 | 作用 |
|---|---|
| `naive_bayes.py` | `NaiveBayesClassifier`，只依赖 NumPy |
| `bayes_net.py` | `BayesNetClassifier` 和 `extract_features(X)`，只依赖 NumPy |
| `main.py` | 完整训练、测试、图像与元数据生成 |
| `mnist_loader.py` | 原样保留的课程读取器 |
| `report.md` | 整合后的完整实验报告，包含三阶段对照 |
| `REVIEW.md` | Gemini / 人工复核入口 |
| `selected_config.json` | 最终测试前冻结的配置和模型源码校验值 |
| `optimization_protocol.json`、`experiments/` | 本轮候选范围及可复现的候选实现 |
| `results/optimization/selection.json` | 208 组验证记录、选中方案和同特征消融 |
| `results/predictions.csv` | 10,000 行原始测试顺序的标签和两个预测 |
| `results/metrics.json` | 四项指标、每类指标、混淆矩阵和耗时 |
| `results/run_metadata.json` | 环境、数据/代码校验值、实际配置与数据概况 |
| `figures/` | 两张混淆矩阵、两组错误样例和完整 DAG，PNG/SVG |
| `history/previous/`、`history/initial/` | 前两版完整快照 |

`verify_results.py` 不需要 MNIST 原始文件，会从 CSV 独立复算全部指标和混淆矩阵、验证历史快照与冻结源码、运行 23 项检查，结果保存到 `results/audit.json` 和 `results/test_results.txt`。重跑会改写测试日志，交付清单中的对应哈希会随运行时间变化。

## 训练集内部选型

固定种子 `20261019`，从训练集每类抽取 1,000 张作为验证集，余下 50,000 张拟合候选。NB 48 组，BN 两轮共 160 组，总计 208 组，按验证 Macro F1 排名、Accuracy 为第二标准。BN 全程使用同一手工 DAG，不搜索或学习边。

可重跑全部训练集内部候选，并另存输出。

```console
python optimize_target90.py --data-dir ../assignment2-data/mnist --output-dir selection-recheck
```

此过程只读取训练图像和训练标签，会缓存中间特征，耗时和磁盘占用高于最终训练。默认缓存目录和上述另存目录均已在 `.gitignore` 中忽略。前版测试结果已被查看，本轮明确保留这一事实；本轮最终测试未用于继续选择配置。

## 统一接口示例

```python
from naive_bayes import NaiveBayesClassifier
from bayes_net import BayesNetClassifier, extract_features

nb = NaiveBayesClassifier(alpha=0.1)
assert nb.fit(X_train, y_train) is nb
prediction = nb.predict(X_test)

bn = BayesNetClassifier(alpha=0.1)
assert bn.fit(X_train, y_train) is bn
prediction_bn = bn.predict(X_test)
discrete_features = extract_features(X_test)
```

`X_train` 和 `X_test` 是原始 0 至 255 的 `(n_samples,784)` 数组。预测返回整数 `(n_samples,)`，BN 提取函数返回整数 `(n_samples,12)`、各值在 0 至 4。两个类还提供归一化的 `predict_proba`。空训练数据或非法输入会明确报错，允许空预测批次。

数据、清单和读取器来自课程数据包，数据来源按课程清单记录为 [Google CVDF MNIST 镜像](https://storage.googleapis.com/cvdf-datasets/mnist/)。三个版本均由 Codex 辅助搭建，保留源码和实验记录供作者审阅；个人经历与反思由作者理解和复现后补充。
