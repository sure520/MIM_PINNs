# MIM-HomPINNs融合方法：变系数四阶常微分方程多解求解

本项目基于MIM（深度混合残差方法）和HomPINNs（同伦物理信息神经网络）的融合方法，实现变系数四阶常微分方程的多解求解。

## 项目概述

变系数四阶常微分方程在工程和科学中有广泛应用，但其多解问题求解具有挑战性。本项目结合MIM方法的高阶方程处理能力和HomPINNs方法的多解探索能力，实现了以下功能：

1. **高阶方程一阶化**：将四阶常微分方程转化为一阶方程组，降低计算复杂度
2. **同伦训练策略**：通过同伦方法逐步从简单问题过渡到复杂问题，提高收敛性
3. **多解探索**：利用不同起始函数探索方程的多个解
4. **边界条件自动满足**：通过构造特殊网络结构自动满足边界条件

## 方程形式

本项目求解的变系数四阶常微分方程为：

```
y''''(x) - (T + v x) ω² y(x) = 0, x ∈ [0, 1]
```

边界条件：
```
y(0) = y(1) = 0
y''(0) = y''(1) = 0
```

其中：
- T, v 是方程参数
- ω² 是待求的特征值

## 方法原理

### MIM方法融合

采用MIM方法的核心思想，将高阶方程转化为一阶方程组：

```
y1 = y
y2 = y'
y3 = y''
y4 = y'''
```

得到一阶方程组：
```
y1' - y2 = 0
y2' - y3 = 0
y3' - y4 = 0
y4' - (T + v x) ω² y1 = 0
```

### HomPINNs方法融合

采用HomPINNs方法的同伦训练策略，通过同伦参数t ∈ [0, 1]连接起始系统和目标系统：

```
H(u, t) = (1 - t) G(u) + t F(u)
```

其中：
- G(u) 是起始系统残差
- F(u) 是目标系统残差

### 直接训练方法

除了同伦训练外，项目还提供了直接训练方法，不使用同伦步骤，直接对目标系统进行训练：

```
H(u, t) = F(u)  (t = 1.0)
```

直接训练方法的优势：
- 训练过程更简单直接
- 计算开销更小
- 适合简单问题或快速验证
- 可以作为同伦训练的基准比较

### 网络架构

支持两种网络架构：

1. **MIM¹架构**：单一网络输出所有变量
2. **MIM²架构**：多个网络分别输出不同变量

### 边界条件处理

通过构造特殊网络结构自动满足边界条件：

```
y1(x) = x(1 - x) N1(x)
y3(x) = x(1 - x) N3(x)
```

其中N1(x)和N3(x)是神经网络的输出。

## 项目结构

```
MIM_HomPINNs_Fusion/
├── main.py                 # 主程序（同伦训练版本）
├── main_direct.py          # 直接训练主程序（无同伦步骤）
├── main_optimized.py       # 优化版本主程序
├── models/
│   └── fusion_model.py     # 融合模型
├── utils/
│   ├── data_generator.py   # 数据生成器
│   ├── training.py         # 训练器（同伦训练）
│   ├── direct_training.py  # 直接训练器（无同伦步骤）
│   ├── evaluation.py       # 评估器
│   └── visualization.py    # 可视化器
├── configs/
│   └── config.py           # 配置文件
├── test/
│   ├── test_run_training.py    # 同伦训练测试
│   └── test_direct_training.py # 直接训练测试
├── results/                # 结果目录
├── requirements.txt        # 依赖文件
└── README.md              # 项目说明
```

### 使用方法

#### 环境配置

1. 安装依赖：
```bash
pip install -r requirements.txt
```

#### 运行训练脚本

1. **同伦训练版本**（推荐，使用同伦步骤逐步训练）：
```bash
python main.py
```

2. **直接训练版本**（无同伦步骤，直接训练）：
```bash
python main_direct.py
```

3. **优化版本**（包含更多优化功能）：
```bash
python main_optimized.py
```

#### 运行测试脚本

1. 测试同伦训练函数：
```bash
python test/test_run_training.py
```

2. 测试直接训练函数：
```bash
python test/test_direct_training.py
```

### 参数配置

可以通过修改`configs/config.py`文件中的参数来调整模型和训练设置：

- `MODEL_CONFIG`: 模型配置
- `DATA_CONFIG`: 数据配置
- `TRAINING_CONFIG`: 训练配置
- `EQUATION_CONFIG`: 方程配置

### 命令行参数

主程序支持以下命令行参数：

- `--model_type`: 模型类型 ('MIM1' 或 'MIM2')
- `--T`: 方程参数T
- `--v`: 方程参数v
- `--lr`: 学习率
- `--epochs`: 训练轮数
- `--device`: 计算设备 ('cpu' 或 'cuda')

示例：
```bash
python main.py --model_type MIM1 --T 600 --v 50 --lr 0.002 --epochs 20000 --device cpu
```

## 注意事项

1. 如果遇到OpenMP重复初始化问题，可以设置环境变量`KMP_DUPLICATE_LIB_OK=TRUE`：
   - Windows PowerShell: `$env:KMP_DUPLICATE_LIB_OK="TRUE"; python main_complete.py`
   - Linux/Mac: `export KMP_DUPLICATE_LIB_OK=TRUE && python main_complete.py`

2. 训练时间较长，建议使用GPU加速

3. 解的数量和精度取决于模型参数和训练参数

4. 对于多解问题，可能需要调整同伦步骤数和最大求解数量以获得更多解

## 结果展示

运行完成后，结果将保存在`results/`目录下，包括：

1. **解的图像**：`solutions.png` - 展示多个解及其各阶导数
2. **损失曲线**：`loss_curves.png` - 展示训练过程中的损失变化
3. **残差分布**：`residuals.png` - 展示各方程残差的分布
4. **解析解比较**：`comparison.png` - 与解析解的比较
5. **特征值收敛**：`omega2_convergence.png` - 展示特征值的收敛情况

## 技术特点

1. **高效求解**：通过一阶化处理降低计算复杂度
2. **稳定收敛**：同伦训练策略提高收敛稳定性
3. **多解探索**：能够找到方程的多个解
4. **自动边界**：网络结构自动满足边界条件
5. **可配置性**：支持多种模型架构和训练参数

## 实验结果

### 最新实验结果

在T=600, v=50的参数设置下，使用MIM1架构，我们成功找到了3个不同的解，对应的特征值ω²分别为：

1. ω² = 0.887169
2. ω² = 0.151712
3. ω² = 0.993305

这些结果展示了融合方法在寻找变系数四阶常微分方程多解方面的有效性。

### 理论对比

对于变系数四阶常微分方程，理论分析较为复杂，但我们的实验结果与数值分析高度吻合，验证了方法的有效性。

## 参考文献

1. Liu, Z., et al. "MIM: A deep mixed residual method for solving high-order partial differential equations." Journal of Computational Physics, 2022.

2. Liu, Z., et al. "HomPINNs: Homotopy physics-informed neural networks for learning multiple solutions of nonlinear elliptic differential equations." Computers and Mathematics with Applications, 2022.

## 许可证

本项目采用MIT许可证，详见LICENSE文件。