"""
优化的MIM-HomPINNs融合项目配置文件
基于MIM和HomPINNs论文最佳实践的超参数配置
"""

# 优化的模型配置
OPTIMIZED_MODEL_CONFIG = {
    'model_type': 'MIM1',           # 单DNN架构，参数少，收敛快
    'resnet_layers': 4,             # ResNet层数（平衡表达能力和训练效率）
    'resnet_neurons': 50,           # 神经元数（足够表达复杂解结构）
    'activation': 'tanh',            # 激活函数（适合PDE求解）
}

# 优化的训练配置
OPTIMIZED_TRAINING_CONFIG = {
    'lr': 0.001,                    # 学习率（比默认0.002更稳定）
    'epochs': 20000,                # 每个同伦步骤训练轮数（确保充分收敛）
    'n_homotopy_steps': 11,         # 同伦步骤数（t=0→1，步长0.1）
    'decay_rate': 0.95,             # 学习率衰减率（更平缓的衰减）
    'alpha': 10,                    # 边界惩罚系数（保持强约束）
    'omega2_init': 1.0,             # 特征值初始猜测
<<<<<<< HEAD
    'homotopy_init_ks': [1, 2, 3, 4, 5],  # 多起始函数探索
=======
    'homotopy_init_ks': [3, 4, 5],  # 多起始函数探索
>>>>>>> 427c4e7a161cedc628bf7a527615f8d9abe2a2af
    'solution_threshold': 0.05,     # 解差异阈值（更严格的解筛选）
    'max_solutions': 10,            # 最大解数量（充分探索解空间）
    'verbose': 1,                   # 日志打印频率
}

# 优化的数据配置
OPTIMIZED_DATA_CONFIG = {
    'N_f': 5000,                    # 配置点数量（平衡精度和效率）
    'N_b': 200,                     # 边界点数量（增强边界约束）
    'N_test': 1000,                 # 测试点数量
    'domain': [0, 1],               # 定义域
}

# 方程配置（保持不变）
EQUATION_CONFIG = {
    'T': 600,                       # 方程参数T
    'v': 50,                        # 方程参数v
}

# 设备配置
DEVICE_CONFIG = {
    'device': 'cuda',                # 'cpu' 或 'cuda'
}

# 保存配置
SAVE_CONFIG = {
<<<<<<< HEAD
    'save_dir': 'optimized_results', # 保存目录
=======
    'save_dir': 'results', # 保存目录
>>>>>>> 427c4e7a161cedc628bf7a527615f8d9abe2a2af
    'save_models': True,            # 是否保存模型
    'save_plots': True,             # 是否保存图像
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'figsize': (12, 8),             # 图像大小
    'dpi': 300,                     # 图像分辨率
}

# 非零解惩罚项配置（新增）
NONZERO_PENALTY_CONFIG = {
    'beta': 1e-4,                   # 惩罚权重（避免过度干扰物理约束）
    'epsilon': 1e-6,                # 数值稳定项
}

def get_optimized_config():
    """
    获取优化的完整配置
    Returns:
        config: 包含所有优化配置的字典
    """
    config = {
        'model': OPTIMIZED_MODEL_CONFIG,
        'data': OPTIMIZED_DATA_CONFIG,
        'training': OPTIMIZED_TRAINING_CONFIG,
        'equation': EQUATION_CONFIG,
        'device': DEVICE_CONFIG,
        'save': SAVE_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
        'nonzero_penalty': NONZERO_PENALTY_CONFIG
    }
    return config

def get_quick_config():
    """
    获取快速测试配置（用于调试和快速验证）
    Returns:
        config: 快速测试配置
    """
    config = get_optimized_config()
    
    # 修改为快速配置
    config['training']['epochs'] = 1000        # 减少训练轮数
    config['training']['n_homotopy_steps'] = 3 # 减少同伦步骤
    config['data']['N_f'] = 1000               # 减少配置点
    config['data']['N_b'] = 50                 # 减少边界点
    config['training']['verbose'] = 2          # 详细日志
    
    return config

def get_high_precision_config():
    """
    获取高精度配置（用于最终结果生成）
    Returns:
        config: 高精度配置
    """
    config = get_optimized_config()
    
    # 修改为高精度配置
    config['training']['epochs'] = 50000       # 增加训练轮数
    config['data']['N_f'] = 10000              # 增加配置点
    config['data']['N_b'] = 500                # 增加边界点
    config['training']['solution_threshold'] = 0.01  # 更严格的解筛选
    
    return config

# 配置说明文档
CONFIG_DOCS = """
MIM-HomPINNs融合项目优化配置说明
================================

配置类型：
1. get_optimized_config() - 平衡配置（推荐使用）
2. get_quick_config() - 快速测试配置
3. get_high_precision_config() - 高精度配置

关键优化点：
- 学习率：0.001（更稳定收敛）
- 同伦步骤：11步（精细路径跟踪）
- 数据采样：5000配置点 + 200边界点
- 多解探索：5个起始函数，最大10个解
- 非零解惩罚：beta=1e-4, epsilon=1e-6

使用示例：
```python
from configs.optimized_config import get_optimized_config

config = get_optimized_config()
model = MIMHomPINNFusion(config['model'])
data_gen = DataGenerator(config['data'])
trainer = Trainer(model, data_gen, **config['training'])
```
"""

if __name__ == "__main__":
    # 测试配置加载
    config = get_optimized_config()
    print("优化配置加载成功！")
    print(f"模型类型: {config['model']['model_type']}")
    print(f"学习率: {config['training']['lr']}")
    print(f"训练轮数: {config['training']['epochs']}")
    print(f"同伦步骤: {config['training']['n_homotopy_steps']}")
    print(f"配置点数量: {config['data']['N_f']}")
    print(f"起始函数k值: {config['training']['homotopy_init_ks']}")