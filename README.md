# Project 2: CIFAR-10 Classification and Batch Normalization

本仓库包含课程 Project 2 的 PyTorch 代码。代码用于完成两个任务：

1. 在 CIFAR-10 上训练和比较多个分类网络。
2. 比较 VGG-A 与 VGG-A+BatchNorm，并绘制 loss landscape 和梯度统计图。

数据集、训练结果和模型权重不会随代码上传到 GitHub。运行脚本时，CIFAR-10 会由 `torchvision` 自动下载到 `data/`，训练输出会保存到 `reports/`。

## 目录结构

```text
.
├── VGG_Loss_Landscape.py
├── data
│   ├── __init__.py
│   └── loaders.py
├── models
│   ├── __init__.py
│   └── vgg.py
├── requirements.txt
├── run_experiments.py
├── train_cifar.py
├── utils
│   ├── __init__.py
│   ├── losses.py
│   └── nn.py
└── visualize_model.py
```

主要文件说明：

- `train_cifar.py`：训练单个 CIFAR-10 分类模型。
- `run_experiments.py`：运行主分类任务的多组对比实验。
- `VGG_Loss_Landscape.py`：运行 VGG-A 与 VGG-A+BN 的比较实验。
- `visualize_model.py`：根据训练好的模型权重生成滤波器和预测样例图。
- `models/vgg.py`：包含 VGG-A、VGG-A+BN、CIFARConvNet 和 CIFARResidualNet。
- `data/loaders.py`：包含 CIFAR-10 数据加载、标准化和数据增强。
- `utils/losses.py`：包含交叉熵、label smoothing 和 focal loss。
- `utils/nn.py`：包含设备选择、随机种子、准确率、保存 checkpoint 等工具函数。

## 环境配置

建议使用 Python 3.9 到 3.11。先创建并激活虚拟环境，然后安装依赖：

```bash
pip install -r requirements.txt
```

如果使用 NVIDIA GPU，PyTorch 会使用 CUDA。  
如果使用 Apple Silicon 机器，并且 `torch.backends.mps.is_available()` 为 `True`，代码可以使用 MPS。

检查 PyTorch 设备：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.backends.mps.is_available())"
```

## 快速测试

正式训练前建议先用小数据集测试代码流程：

```bash
python train_cifar.py --epochs 1 --n-train 1024 --n-test 256 --workers 0
python VGG_Loss_Landscape.py --epochs 1 --n-train 1024 --n-test 256 --workers 0
```

如果使用 Apple Silicon GPU，可以显式指定：

```bash
python train_cifar.py --epochs 1 --n-train 1024 --n-test 256 --workers 0 --device mps
```

如果使用 NVIDIA GPU，可以显式指定：

```bash
python train_cifar.py --epochs 1 --n-train 1024 --n-test 256 --workers 2 --device cuda
```

## 训练单个模型

训练默认的 CIFARConvNet：

```bash
python train_cifar.py --model cifar_conv --epochs 20 --optimizer adamw --loss cross_entropy --augment
```

使用 Adam 优化器训练：

```bash
python train_cifar.py --model cifar_conv --epochs 20 --optimizer adam --loss cross_entropy --augment
```

使用不同激活函数：

```bash
python train_cifar.py --model cifar_conv --epochs 20 --activation gelu --optimizer adamw --augment
```

调整网络宽度：

```bash
python train_cifar.py --model cifar_conv --epochs 20 --width-multiplier 1.5 --optimizer adamw --augment
```

输出会保存在：

```text
reports/cifar/
```

每次运行会生成：

- `config.json`
- `metrics.csv`
- `summary.json`
- `training_curves.png`
- `best_model.pt`
- `last_model.pt`

## 运行主分类对比实验

主分类任务比较以下因素：

- 不同卷积通道数；
- 不同损失函数；
- 不同激活函数；
- 不同优化器；
- 不同正则化；
- 残差连接。

运行命令：

```bash
python run_experiments.py --epochs 20 --augment
```

Apple Silicon 机器：

```bash
python run_experiments.py --epochs 20 --augment --device mps --workers 0
```

NVIDIA GPU：

```bash
python run_experiments.py --epochs 20 --augment --device cuda --workers 2
```

输出会保存在：

```text
reports/cifar_experiments/
```

总表为：

```text
reports/cifar_experiments/experiment_summary.csv
```

## 运行 BatchNorm 对比实验

该脚本会分别训练 VGG-A 和 VGG-A+BN，并在多个学习率下记录训练 loss、测试准确率、梯度范数和相邻梯度差异。

```bash
python VGG_Loss_Landscape.py --epochs 20 --learning-rates 0.001,0.002,0.0001,0.0005
```

Apple Silicon 机器：

```bash
python VGG_Loss_Landscape.py --epochs 20 --learning-rates 0.001,0.002,0.0001,0.0005 --device mps --workers 0
```

NVIDIA GPU：

```bash
python VGG_Loss_Landscape.py --epochs 20 --learning-rates 0.001,0.002,0.0001,0.0005 --device cuda --workers 2
```

输出会保存在：

```text
reports/bn/
```

主要输出文件包括：

- `bn_summary.csv`
- `bn_epoch_metrics.csv`
- `bn_histories.json`
- `bn_accuracy.png`
- `loss_landscape_compare.png`
- `gradient_statistics.png`
- `vgg_a_lr_*.pt`
- `vgg_a_bn_lr_*.pt`

## 生成模型可视化

训练结束后，可以用 `visualize_model.py` 生成第一层卷积滤波器和测试样本预测图。

示例：

```bash
python visualize_model.py --checkpoint reports/cifar_experiments/optimizer_adam/best_model.pt
```

输出会保存在：

```text
reports/visualizations/
```

## 本次实验结果摘要

在本地实验中，主分类实验的最优模型为 `cifar_conv` 配合 Adam 优化器，测试准确率为 `90.83%`，测试错误率为 `9.17%`。

BatchNorm 对比实验中，VGG-A+BN 在学习率 `0.002` 下取得最佳测试准确率 `83.19%`；无 BatchNorm 的 VGG-A 最佳测试准确率为 `78.38%`。

这些数值依赖随机种子、硬件、PyTorch 版本和训练配置。重新运行时可能存在小幅差异。

## 不应上传到 GitHub 的文件

以下文件或目录由 `.gitignore` 排除：

- `data/cifar-10-batches-py/`
- `data/cifar-10-python.tar.gz`
- `reports/`
- `*.pt`
- `*.pth`
- `__pycache__/`
- `.DS_Store`

如果课程提交要求提供模型权重，需要将权重上传到 Google Drive、OneDrive、百度网盘或其他网盘，并在报告中提供下载链接。

## 参考资料

- CIFAR-10: <https://www.cs.toronto.edu/~kriz/cifar.html>
- PyTorch CIFAR-10 tutorial: <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>
- Batch Normalization paper: <https://arxiv.org/abs/1502.03167>
- How Does Batch Normalization Help Optimization?: <https://arxiv.org/abs/1805.11604>
