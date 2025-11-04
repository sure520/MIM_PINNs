"""
直接训练器测试脚本
测试DirectTrainer类的完整功能
"""

import torch
import numpy as np
import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.fusion_model import MIMHomPINNFusion
from data.data_generator import DataGenerator
from utils.direct_trainer import DirectTrainer, create_direct_trainer


def test_direct_trainer():
    """测试直接训练器基本功能"""
    print("=== 测试直接训练器 ===")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试配置
    config = {
        'training': {
            'epochs': 100,
            'lr': 0.001,
            'optimizer': 'adam',
            'early_stopping': True,
            'patience': 20,
            'min_delta': 1e-6,
            'save_interval': 10,
            'alpha': 1.0,
            'beta': 1.0,
            'omega2_init': 1.0
        },
        'data': {
            'N_f': 1000,
            'N_b': 200,
            'N_test': 500,
            'domain': [0.0, 1.0]
        },
        'equation': {
            'T': 600,
            'v': 50
        }
    }
    
    # 创建模型
    model = MIMHomPINNFusion(
        width=30,
        depth=2,
        model_type='MIM1',
        device=device
    )
    
    # 创建数据生成器
    data_gen = DataGenerator(
        domain=config['data']['domain'],
        n_domain=config['data']['N_f'],
        n_boundary=config['data']['N_b'],
        n_test=config['data']['N_test']
    )
    
    # 创建训练器
    trainer = DirectTrainer(
        model=model,
        data_gen=data_gen,
        config=config,
        device=device,
        save_dir='test_results'
    )
    
    print("训练器创建成功")
    
    # 测试便捷函数
    trainer2 = create_direct_trainer(
        model=model,
        data_gen=data_gen,
        config=config,
        device=device,
        save_dir='test_results2'
    )
    print("便捷函数创建训练器成功")
    
    # 测试训练过程
    print("开始训练...")
    trainer.train()
    print("训练完成")
    
    # 测试评估功能
    print("开始评估...")
    eval_results = trainer.evaluate()
    print(f"评估结果: {eval_results}")
    
    # 验证训练历史
    print(f"训练历史长度: {len(trainer.history['total_loss'])}")
    print(f"最终损失: {trainer.history['total_loss'][-1] if trainer.history['total_loss'] else 'N/A'}")
    
    return True


def test_config_validation():
    """测试配置验证功能"""
    print("\n=== 测试配置验证 ===")
    
    device = torch.device('cpu')
    
    # 测试不完整配置
    partial_config = {
        'training': {
            'epochs': 50,
            'lr': 0.001,
            'omega2_init': 1.0
        },
        'data': {
            'N_f': 100,
            'N_b': 20,
            'N_test': 50,
            'domain': [0.0, 1.0]
        },
        'equation': {
            'T': 600,
            'v': 50
        }
    }
    
    model = MIMHomPINNFusion(width=20, depth=2, model_type='MIM1', device=device)
    data_gen = DataGenerator(domain=partial_config['data']['domain'], n_domain=partial_config['data']['N_f'], n_boundary=partial_config['data']['N_b'], n_test=partial_config['data']['N_test'])
    
    try:
        trainer = DirectTrainer(model, data_gen, partial_config, device, 'test_partial')
        print("不完整配置处理成功")
    except Exception as e:
        print(f"不完整配置处理失败: {e}")
        return False
    
    # 测试空配置
    try:
        trainer = DirectTrainer(model, data_gen, None, device, 'test_empty')
        print("空配置处理成功")
    except Exception as e:
        print(f"空配置处理失败: {e}")
        return False
    
    return True


def test_training_components():
    """测试训练组件功能"""
    print("\n=== 测试训练组件 ===")
    
    device = torch.device('cpu')
    
    config = {
        'training': {
            'epochs': 10,
            'lr': 0.01,
            'optimizer': 'adam',
            'early_stopping': False,
            'save_interval': 5,
            'alpha': 1.0,
            'beta': 1.0,
            'omega2_init': 1.0
        },
        'data': {
            'N_f': 100,
            'N_b': 20,
            'N_test': 50,
            'domain': [0.0, 1.0]
        },
        'equation': {
            'T': 600,
            'v': 50
        }
    }
    
    model = MIMHomPINNFusion(width=20, depth=2, model_type='MIM1', device=device)
    data_gen = DataGenerator(domain=config['data']['domain'], n_domain=config['data']['N_f'], n_boundary=config['data']['N_b'], n_test=config['data']['N_test'])
    
    trainer = DirectTrainer(model, data_gen, config, device, 'test_components')
    
    # 测试损失计算
    try:
        total_loss, pde_loss, bc_loss, nonzero_loss = trainer.compute_loss(trainer.x, trainer.x_b)
        print(f"损失计算成功: Total={total_loss.item():.4f}, PDE={pde_loss.item():.4f}, BC={bc_loss.item():.4f}")
    except Exception as e:
        print(f"损失计算失败: {e}")
        return False
    
    # 测试单步训练
    try:
        total_loss, pde_loss, bc_loss, nonzero_loss = trainer._train_step()
        print(f"单步训练成功: Total={total_loss.item():.4f}")
    except Exception as e:
        print(f"单步训练失败: {e}")
        return False
    
    return True


def main():
    """主测试函数"""
    print("开始测试直接训练器...")
    
    # 运行所有测试
    tests = [
        test_direct_trainer,
        test_config_validation,
        test_training_components
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"测试 {test_func.__name__} 失败: {e}")
            results.append((test_func.__name__, False))
    
    # 输出测试结果
    print("\n=== 测试结果汇总 ===")
    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 所有测试通过！直接训练器功能正常。")
    else:
        print("\n⚠️ 部分测试失败，请检查代码实现。")
    
    return all_passed


if __name__ == "__main__":
    main()