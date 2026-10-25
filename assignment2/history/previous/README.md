# Assignment 2 优化版

本目录包含去倾斜与八档灰度的 Naive Bayes、小型手工贝叶斯网络、完整训练评估入口、优化报告和检查用例。两个模型文件均可被助教单独加载，只依赖 NumPy。完整测试集准确率分别为 87.75% 和 85.16%。

初版报告及其原图、代码、结果完整保存在 [history/initial/report.md](history/initial/report.md)，所有原文件校验值保持一致。初版准确率分别为 84.33% 和 55.91%。根目录的 `report_initial.md` 仅作为初版导航，原始报告正文位于历史目录。

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
    ├── optimize.py              # 仅读取训练集的 54 组候选比较
    ├── selected_config.json     # 最终测试前冻结的配置及核心源码校验值
    ├── verify_results.py        # 无需原始数据的结果复算与检查
    ├── tests/test_models.py
    ├── figures/
    ├── results/
    └── history/initial/         # 初版完整快照
```

在 `assignment2/` 中执行以下命令。Windows PowerShell 和其他终端均可使用。

```console
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python main.py --data-dir ../assignment2-data/mnist
python verify_results.py
```

在本次 Codex 工作目录中，已解压的数据位置为 `work/data/assignment2-data/mnist`。从工作目录运行时可使用以下命令。

```console
python outputs/assignment2/main.py --data-dir work/data/assignment2-data/mnist
```

两个模型类的构造函数默认值始终为 `alpha=1.0`，严格保留课程接口。新版 `main.py` 的 `--alpha` 默认值为验证选中的 `0.1`，由入口显式传给模型；报告结果对应 `0.1`。省略类构造参数的调用仍使用 `1.0`。

本轮从原始训练集每类固定抽出 1,000 张组成验证集，种子为 `20261019`，其余 50,000 张用于候选训练。54 组候选按验证 Macro F1 排序，全部 BN 使用同一手工 DAG；不存在结构搜索。`optimize.py` 不读取原始测试集。选型后先复核验证结果、冻结配置，再使用完整训练集拟合和测试。

初版测试结果此前已经被查看，本版报告明确保留这一事实。本轮最终测试结果未用于继续挑选方案。

程序会先核对冻结的核心源码与选型记录，再检查四个数据文件的 SHA-256、加载完整训练集和测试集、分别训练两个模型，并生成下列结果。验证划分使用固定随机种子，最终拟合和预测没有随机过程。

- `results/metrics.json` 保存四项指标、每类指标、混淆矩阵、训练/预测耗时和易混淆数字对。
- `results/predictions.csv` 保存 10,000 张测试图像的原始顺序索引、真实标签及两个预测标签，可独立重算指标。
- `results/run_metadata.json` 保存数据与源代码的 SHA-256、环境版本、配置和数据概况。
- `results/optimization/selection.json` 保存全部 54 组验证结果、选中配置与同特征无依赖边的消融结果。该消融模型仅用于分析，提交的 BN 保留八条特征间边。
- `results/confusion_*.csv` 的行表示真实类别，列表示预测类别，类别顺序均为 0 至 9；文件无表头。
- `results/summary.md` 为本次运行自动生成的指标摘要。
- `figures/` 包含混淆矩阵、各模型前六张误分类图像，以及包含全部 20 条边的 DAG。每图提供 PNG 和 SVG。

`main.py` 会覆盖指定输出目录的结果和图像。可用 `--output-dir PATH` 另存实验；`history/initial/` 不会被覆盖。`report.md` 为本次冻结配置的报告初稿，不会在运行时自动改写；修改配置后须同步核对报告。

可用以下命令重新检查训练集内部的候选结果，并将它们另存，避免覆盖原选型记录。

```console
python optimize.py --data-dir ../assignment2-data/mnist --output-dir selection-recheck
```

`python verify_results.py` 无需原始数据，会独立从 CSV 复算四项指标和混淆矩阵，核对原版文件、冻结源码及实验校验值，并运行 21 项检查。结果写入 `results/audit.json` 和 `results/test_results.txt`。若修改核心模型，须重新复核验证结果并更新冻结记录，不能继续把旧实验当作新源码的结果。

## 模型接口

```python
from naive_bayes import NaiveBayesClassifier
from bayes_net import BayesNetClassifier, extract_features

# X_train / X_test: (n_samples, 784)，原始 0 至 255 像素
model = NaiveBayesClassifier(alpha=0.1)
model.fit(X_train, y_train)  # 返回 self
y_pred = model.predict(X_test)  # 整数数组，shape 为 (n_samples,)

network = BayesNetClassifier(alpha=0.1)
network.fit(X_train, y_train)
network_prediction = network.predict(X_test)
features = extract_features(X_test)  # (n_samples, 12)，整数状态 0 至 4
```

两个类还提供可选的 `predict_proba(X)`，返回归一化的十类概率。输入空预测批次是允许的；空训练集、非法标签、错误形状及范围外像素会明确报错。

## 数据与来源

数据文件、清单及 `mnist_loader.py` 来自随作业提供的 `assignment2-data.zip`。读取器在本目录中逐字节保留，两个模型及评估代码为本次搭建编写。MNIST 数据来源按课程清单记录为 [Google CVDF MNIST 镜像](https://storage.googleapis.com/cvdf-datasets/mnist/)。程序中固定的校验值来自课程清单。

初版与优化版由 Codex 辅助搭建，保留完整源码、选型过程和逐样本预测供作者审阅。个人反思应在理解实现和核对结果后由作者补充。

## 提交前

将本目录作为课程仓库的 `assignment2/`，完成作者复核后按课程要求提交。课程截止日期为 2026 年 10 月 25 日，以截止前最后一次 push 为准。当前交付仅为本地文件；原始 MNIST 数据和 Python 虚拟环境均在目录之外。

Gemini 或人工检查可先阅读 `REVIEW.md`，再查看两个模型、检查用例和逐样本预测。
