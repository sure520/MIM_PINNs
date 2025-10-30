"""
配置文件
"""

# 模型配置
MODEL_CONFIG = {
    'model_type': 'MIM1',  # 'MIM1' 或 'MIM2'
    'resnet_layers': 4,    # ResNet层数
    'resnet_neurons': 50,  # ResNet神经元数
    'activation': 'tanh',  # 激活函数: 'tanh', 'relu', 'relu_squared'
}

# 数据配置
DATA_CONFIG = {
    'N_f': 10000,      # 配置点数量
    'N_b': 100,        # 边界点数量
    'N_test': 1000,    # 测试点数量
    'domain': [0, 1],  # 定义域
}

# 训练配置
TRAINING_CONFIG = {
    'lr': 0.002,              # 学习率
    'epochs': 20000,          # 每个同伦步骤的训练轮数
    'n_homotopy_steps': 11,   # 同伦步骤数
    'decay_rate': 0.9,        # 学习率衰减率
    'alpha': 10,              # 边界惩罚系数
    'omega2_init': 1.0,       # 特征值初始猜测
    'homotopy_init_ks': [1, 2, 3, 4, 5],  # 起始函数的k值列表
    'solution_threshold': 0.1,  # 解差异阈值
    'max_solutions': 5,       # 最大解数量
    'verbose': 1,             # 日志打印频率
}

# 方程配置
EQUATION_CONFIG = {
    'T': 600,  # 方程参数T
    'v': 50,   # 方程参数v
}

# 设备配置
DEVICE_CONFIG = {
    'device': 'cpu',  # 'cpu' 或 'cuda'
}

# 保存配置
SAVE_CONFIG = {
    'save_dir': 'results',  # 保存目录
    'save_models': True,    # 是否保存模型
    'save_plots': True,     # 是否保存图像
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'figsize': (12, 8),  # 图像大小
    'dpi': 300,          # 图像分辨率
}


def get_config():
    """
    获取完整配置
    Returns:
        config: 包含所有配置的字典
    """
    config = {
        'model': MODEL_CONFIG,
        'data': DATA_CONFIG,
        'training': TRAINING_CONFIG,
        'equation': EQUATION_CONFIG,
        'device': DEVICE_CONFIG,
        'save': SAVE_CONFIG,
        'visualization': VISUALIZATION_CONFIG
    }
    return config


def update_config(args):
    """
    根据命令行参数更新配置
    Args:
        args: 命令行参数
    Returns:
        config: 更新后的配置
    """
    config = get_config()
    
    # 更新模型配置
    if hasattr(args, 'width'):
        config['model']['resnet_neurons'] = args.width
    if hasattr(args, 'depth'):
        config['model']['resnet_layers'] = args.depth
    if hasattr(args, 'model_type'):
        config['model']['model_type'] = args.model_type
    
    # 更新训练配置
    if hasattr(args, 'lr'):
        config['training']['lr'] = args.lr
    if hasattr(args, 'epochs'):
        config['training']['epochs'] = args.epochs
    if hasattr(args, 'n_homotopy_steps'):
        config['training']['n_homotopy_steps'] = args.n_homotopy_steps
    if hasattr(args, 'decay_rate'):
        config['training']['decay_rate'] = args.decay_rate
    if hasattr(args, 'alpha'):
        config['training']['alpha'] = args.alpha
    if hasattr(args, 'omega2_init'):
        config['training']['omega2_init'] = args.omega2_init
    if hasattr(args, 'homotopy_init_k'):
        config['training']['homotopy_init_k'] = args.homotopy_init_k
    if hasattr(args, 'max_solutions'):
        config['training']['max_solutions'] = args.max_solutions
    if hasattr(args, 'verbose'):
        config['training']['verbose'] = args.verbose
    
    # 更新数据配置
    if hasattr(args, 'N_g'):
        config['data']['N_f'] = args.N_g
    if hasattr(args, 'N_b'):
        config['data']['N_b'] = args.N_b
    if hasattr(args, 'N_test'):
        config['data']['N_test'] = args.N_test
    
    # 更新方程配置
    if hasattr(args, 'T'):
        config['equation']['T'] = args.T
    if hasattr(args, 'v'):
        config['equation']['v'] = args.v
    
    # 更新设备配置
    if hasattr(args, 'device'):
        config['device']['device'] = args.device
    
    # 更新保存配置
    if hasattr(args, 'save_dir'):
        config['save']['save_dir'] = args.save_dir
    
    return config