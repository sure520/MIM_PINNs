"""
无需同伦步骤的直接训练器配置文件
提供多种预设配置，支持灵活的配置参数调整
"""

# 直接训练器优化配置
DIRECT_TRAINER_CONFIG = {
    'training': {
        'lr': 0.001,                    # 学习率
        'epochs': 20000,                # 训练轮数
        'optimizer': 'adam',            # 优化器类型
        'lr_scheduler': 'step',         # 学习率调度器
        'lr_decay_rate': 0.95,          # 学习率衰减率
        'lr_decay_steps': 5000,         # 学习率衰减步数
        'omega2_init': 1.0,             # 特征值初始猜测
        'alpha': 10.0,                  # 边界惩罚系数
        'beta': 1e-4,                   # 非零解惩罚系数
        'verbose': 1,                   # 日志打印频率
        'save_interval': 1000,          # 保存间隔
        'early_stopping': True,         # 是否启用早停
        'patience': 1000,               # 早停耐心值
        'min_delta': 1e-6               # 早停最小变化量
    },
    'data': {
        'N_f': 5000,                    # 配置点数量
        'N_b': 200,                     # 边界点数量
        'N_test': 1000,                 # 测试点数量
        'domain': [0, 1]                # 定义域
    },
    'equation': {
        'T': 600,                       # 方程参数T
        'v': 50                         # 方程参数v
    }
}

# 快速测试配置
QUICK_CONFIG = {
    'training': {
        'lr': 0.001,
        'epochs': 1000,                 # 减少训练轮数
        'optimizer': 'adam',
        'lr_scheduler': 'none',         # 不使用学习率调度
        'omega2_init': 1.0,
        'alpha': 10.0,
        'beta': 1e-4,
        'verbose': 2,                   # 更详细的日志
        'save_interval': 200,
        'early_stopping': True,
        'patience': 200,
        'min_delta': 1e-4
    },
    'data': {
        'N_f': 1000,                    # 减少配置点
        'N_b': 50,                      # 减少边界点
        'N_test': 500,
        'domain': [0, 1]
    },
    'equation': {
        'T': 600,
        'v': 50
    }
}

# 高精度配置
HIGH_PRECISION_CONFIG = {
    'training': {
        'lr': 0.0005,                   # 更小的学习率
        'epochs': 50000,                # 更多训练轮数
        'optimizer': 'adam',
        'lr_scheduler': 'plateau',      # 使用自适应学习率调度
        'lr_decay_rate': 0.9,
        'omega2_init': 1.0,
        'alpha': 10.0,
        'beta': 1e-5,                   # 更小的非零解惩罚
        'verbose': 1,
        'save_interval': 2000,
        'early_stopping': True,
        'patience': 2000,
        'min_delta': 1e-7               # 更严格的早停条件
    },
    'data': {
        'N_f': 10000,                   # 更多配置点
        'N_b': 500,                      # 更多边界点
        'N_test': 2000,
        'domain': [0, 1]
    },
    'equation': {
        'T': 600,
        'v': 50
    }
}

# 不同优化器配置
OPTIMIZER_CONFIGS = {
    'adam': {
        'training': {
            'optimizer': 'adam',
            'lr': 0.001
        }
    },
    'sgd': {
        'training': {
            'optimizer': 'sgd',
            'lr': 0.01,                  # SGD需要更大的学习率
            'momentum': 0.9
        }
    },
    'rmsprop': {
        'training': {
            'optimizer': 'rmsprop',
            'lr': 0.001
        }
    }
}

# 不同学习率调度器配置
SCHEDULER_CONFIGS = {
    'step': {
        'training': {
            'lr_scheduler': 'step',
            'lr_decay_rate': 0.95,
            'lr_decay_steps': 5000
        }
    },
    'exponential': {
        'training': {
            'lr_scheduler': 'exponential',
            'lr_decay_rate': 0.99        # 更平缓的衰减
        }
    },
    'plateau': {
        'training': {
            'lr_scheduler': 'plateau',
            'lr_decay_rate': 0.5         # 损失平台时大幅降低学习率
        }
    },
    'none': {
        'training': {
            'lr_scheduler': 'none'       # 不使用调度器
        }
    }
}


def get_direct_config(config_type='balanced'):
    """
    获取直接训练器配置
    
    Args:
        config_type: 配置类型，可选 'balanced', 'quick', 'high_precision'
        
    Returns:
        config: 配置字典
    """
    if config_type == 'balanced':
        return DIRECT_TRAINER_CONFIG.copy()
    elif config_type == 'quick':
        return QUICK_CONFIG.copy()
    elif config_type == 'high_precision':
        return HIGH_PRECISION_CONFIG.copy()
    else:
        raise ValueError(f"不支持的配置类型: {config_type}")


def get_custom_config(base_config='balanced', **kwargs):
    """
    获取自定义配置
    
    Args:
        base_config: 基础配置类型
        **kwargs: 自定义参数
        
    Returns:
        config: 自定义配置字典
    """
    config = get_direct_config(base_config)
    
    # 深度更新配置
    for key, value in kwargs.items():
        if '.' in key:
            # 处理嵌套键，如 'training.lr'
            keys = key.split('.')
            current_dict = config
            for k in keys[:-1]:
                if k not in current_dict:
                    current_dict[k] = {}
                current_dict = current_dict[k]
            current_dict[keys[-1]] = value
        else:
            # 处理顶层键
            config[key] = value
    
    return config


def get_optimizer_config(optimizer_type='adam', base_config='balanced'):
    """
    获取特定优化器的配置
    
    Args:
        optimizer_type: 优化器类型
        base_config: 基础配置类型
        
    Returns:
        config: 配置字典
    """
    config = get_direct_config(base_config)
    
    if optimizer_type in OPTIMIZER_CONFIGS:
        # 深度合并优化器配置
        _deep_merge(config, OPTIMIZER_CONFIGS[optimizer_type])
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    return config


def get_scheduler_config(scheduler_type='step', base_config='balanced'):
    """
    获取特定学习率调度器的配置
    
    Args:
        scheduler_type: 调度器类型
        base_config: 基础配置类型
        
    Returns:
        config: 配置字典
    """
    config = get_direct_config(base_config)
    
    if scheduler_type in SCHEDULER_CONFIGS:
        # 深度合并调度器配置
        _deep_merge(config, SCHEDULER_CONFIGS[scheduler_type])
    else:
        raise ValueError(f"不支持的调度器类型: {scheduler_type}")
    
    return config


def _deep_merge(target, source):
    """深度合并两个字典"""
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def validate_config(config):
    """
    验证配置的有效性
    
    Args:
        config: 配置字典
        
    Returns:
        is_valid: 配置是否有效
        errors: 错误信息列表
    """
    errors = []
    
    # 检查必需字段
    required_fields = ['training', 'data', 'equation']
    for field in required_fields:
        if field not in config:
            errors.append(f"缺少必需字段: {field}")
    
    # 检查训练配置
    if 'training' in config:
        training = config['training']
        
        if training.get('lr', 0) <= 0:
            errors.append("学习率必须大于0")
        
        if training.get('epochs', 0) <= 0:
            errors.append("训练轮数必须大于0")
        
        if training.get('alpha', 0) < 0:
            errors.append("边界惩罚系数不能为负数")
        
        if training.get('beta', 0) < 0:
            errors.append("非零解惩罚系数不能为负数")
    
    # 检查数据配置
    if 'data' in config:
        data = config['data']
        
        if data.get('N_f', 0) <= 0:
            errors.append("配置点数量必须大于0")
        
        if data.get('N_b', 0) <= 0:
            errors.append("边界点数量必须大于0")
        
        if data.get('N_test', 0) <= 0:
            errors.append("测试点数量必须大于0")
    
    return len(errors) == 0, errors


def print_config_summary(config, title="配置摘要"):
    """
    打印配置摘要
    
    Args:
        config: 配置字典
        title: 标题
    """
    print(f"\n=== {title} ===")
    
    if 'training' in config:
        training = config['training']
        print("训练配置:")
        print(f"  学习率: {training.get('lr', 'N/A')}")
        print(f"  训练轮数: {training.get('epochs', 'N/A')}")
        print(f"  优化器: {training.get('optimizer', 'N/A')}")
        print(f"  学习率调度器: {training.get('lr_scheduler', 'N/A')}")
        print(f"  边界惩罚系数: {training.get('alpha', 'N/A')}")
        print(f"  非零解惩罚系数: {training.get('beta', 'N/A')}")
    
    if 'data' in config:
        data = config['data']
        print("数据配置:")
        print(f"  配置点数量: {data.get('N_f', 'N/A')}")
        print(f"  边界点数量: {data.get('N_b', 'N/A')}")
        print(f"  测试点数量: {data.get('N_test', 'N/A')}")
        print(f"  定义域: {data.get('domain', 'N/A')}")
    
    if 'equation' in config:
        equation = config['equation']
        print("方程配置:")
        print(f"  参数T: {equation.get('T', 'N/A')}")
        print(f"  参数v: {equation.get('v', 'N/A')}")
    
    print("=" * 50)


# 配置说明文档
CONFIG_DOCS = """
无需同伦步骤的直接训练器配置说明
==================================

配置类型：
1. balanced - 平衡配置（推荐使用）
2. quick - 快速测试配置
3. high_precision - 高精度配置

关键参数说明：
- lr: 学习率，控制参数更新步长
- epochs: 训练轮数，决定训练时长
- optimizer: 优化器类型（adam/sgd/rmsprop）
- lr_scheduler: 学习率调度器（step/exponential/plateau/none）
- alpha: 边界惩罚系数，控制边界条件的重要性
- beta: 非零解惩罚系数，避免得到零解
- N_f: 配置点数量，影响PDE残差计算精度
- N_b: 边界点数量，影响边界条件约束强度

使用示例：
```python
from configs.direct_config import get_direct_config
from utils.direct_trainer import DirectTrainer

# 获取平衡配置
config = get_direct_config('balanced')

# 自定义配置
custom_config = get_custom_config(
    base_config='balanced',
    training_lr=0.0005,
    training_epochs=30000,
    data_N_f=8000
)

# 创建训练器
trainer = DirectTrainer(model, data_gen, config=custom_config)
```
"""


if __name__ == "__main__":
    # 测试配置加载
    config = get_direct_config('balanced')
    print_config_summary(config, "平衡配置")
    
    # 验证配置
    is_valid, errors = validate_config(config)
    if is_valid:
        print("配置验证通过！")
    else:
        print("配置验证失败:")
        for error in errors:
            print(f"  - {error}")
    
    # 测试自定义配置
    custom_config = get_custom_config(
        base_config='balanced',
        training_lr=0.0005,
        training_epochs=30000
    )
    print_config_summary(custom_config, "自定义配置")