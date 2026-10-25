# Assignment 2 可运行基线

本目录包含手写 Bernoulli Naive Bayes、小型手工贝叶斯网络、完整训练评估入口、报告初稿和检查用例。两个模型文件均可被助教单独加载，只依赖 NumPy。

## 安装与运行

使用 Python 3.10 或更新版本。在课程仓库中按以下方式放置文件，数据目录与作业目录平级。

```text
course-repository/
├── assignment2-data/
│   └── mnist/                   # 从课程数据包解压，勿提交到 Git
└── assignment2/
    ├── naive_bayes.py
    ├── bayes_net.py
    ├── main.py
    ├── mnist_loader.py          # 课程提供的读取器，原样保留
    ├── requirements.txt
    ├── report.md
    ├── tests/test_models.py
    ├── figures/
    └── results/
```

在 `assignment2/` 中执行以下命令。Windows PowerShell 和其他终端均可使用。

```console
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python main.py --data-dir ../assignment2-data/mnist
```

在本次 Codex 工作目录中，已解压的数据位置为 `work/data/assignment2-data/mnist`。从工作目录运行时可使用以下命令。

```console
python outputs/assignment2/main.py --data-dir work/data/assignment2-data/mnist
```

`--alpha` 默认为 `1.0`。首次报告使用这一默认值，未使用测试集选择超参数、分箱阈值或网络结构。若后续需要调参，应从训练集划出验证集，确定配置后再做最终测试。

程序会检查四个数据文件的 SHA-256、加载完整训练集和测试集、分别训练两个模型，并生成下列结果。所有实验不包含随机过程。

- `results/metrics.json` 保存四项指标、每类指标、混淆矩阵、训练/预测耗时和易混淆数字对。
- `results/predictions.csv` 保存 10,000 张测试图像的原始顺序索引、真实标签及两个预测标签，可独立重算指标。
- `results/run_metadata.json` 保存数据与源代码的 SHA-256、环境版本、配置和数据概况。
- `results/confusion_*.csv` 的行表示真实类别，列表示预测类别，类别顺序均为 0 至 9；文件无表头。
- `results/summary.md` 为本次运行自动生成的指标摘要。
- `figures/` 包含混淆矩阵、各模型前六张误分类图像，以及包含全部 20 条边的 DAG。每图提供 PNG 和 SVG。

`main.py` 会覆盖本次输出目录的结果和图像。可用 `--output-dir PATH` 为新配置另存一份实验。`report.md` 为人工组织的本次默认配置报告初稿，不会在重新运行时自动改写；更换配置后须同步更新报告里的数值和解释。

## 模型接口

```python
from naive_bayes import NaiveBayesClassifier
from bayes_net import BayesNetClassifier, extract_features

# X_train / X_test: (n_samples, 784)，原始 0 至 255 像素
model = NaiveBayesClassifier(alpha=1.0)
model.fit(X_train, y_train)  # 返回 self
y_pred = model.predict(X_test)  # 整数数组，shape 为 (n_samples,)

network = BayesNetClassifier(alpha=1.0)
network.fit(X_train, y_train)
network_prediction = network.predict(X_test)
features = extract_features(X_test)  # (n_samples, 12)，整数状态 0 至 4
```

两个类还提供可选的 `predict_proba(X)`，返回归一化的十类概率。输入空预测批次是允许的；空训练集、非法标签、错误形状及范围外像素会明确报错。

## 数据与来源

数据文件、清单及 `mnist_loader.py` 来自随作业提供的 `assignment2-data.zip`。读取器在本目录中逐字节保留，两个模型及评估代码为本次搭建编写。MNIST 数据来源按课程清单记录为 [Google CVDF MNIST 镜像](https://storage.googleapis.com/cvdf-datasets/mnist/)。程序中固定的校验值来自课程清单。

本基线和报告初稿由 Codex 辅助搭建，保留完整源码与逐样本预测供作者审阅。个人反思应在理解实现和核对结果后由作者补充。

## 提交前

将本目录作为课程仓库的 `assignment2/`，完成作者复核后按课程要求提交。课程截止日期为 2026 年 10 月 25 日，以截止前最后一次 push 为准。当前交付仅为本地文件；原始 MNIST 数据和 Python 虚拟环境均在目录之外。

Gemini 或人工检查可先阅读 `REVIEW.md`，再查看两个模型、检查用例和逐样本预测。
