#!/usr/bin/env python3
"""
验证所有导入和基本功能是否正常
"""

import torch
import numpy as np
import os
import sys

def validate_imports():
    """验证所有导入是否正常"""
    print("=== 验证导入和基本功能 ===")
    
    # 添加项目路径
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # 测试模型导入
        from models.fusion_model import MIMHomPINNFusion
        print("✓ 模型导入成功")
        
        # 测试数据生成器导入
        from data.data_generator import DataGenerator
        print("✓ 数据生成器导入成功")
        
        # 测试训练器导入
        from utils.direct_trainer import DirectTrainer, create_direct_trainer
        print("✓ 训练器导入成功")
        
        # 测试模型创建
        device = torch.device('cpu')
        model = MIMHomPINNFusion(width=20, depth=2, model_type='MIM1', device=device)
        print("✓ 模型创建成功")
        
        # 测试数据生成器创建
        data_gen = DataGenerator([0, 1], 100, 20, 50)
        print("✓ 数据生成器创建成功")
        
        # 测试配置创建
        config = {
            'training': {
                'epochs': 10,
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
        print("✓ 配置创建成功")
        
        # 测试训练器创建
        trainer = DirectTrainer(model, data_gen, config, device, 'test_validate')
        print("✓ 训练器创建成功")
        
        # 测试便捷函数
        trainer2 = create_direct_trainer(model, data_gen, config, device, 'test_validate2')
        print("✓ 便捷函数创建成功")
        
        # 测试损失计算
        total_loss, pde_loss, bc_loss, nonzero_loss = trainer.compute_loss(trainer.x, trainer.x_b)
        print(f"✓ 损失计算成功: Total={total_loss.item():.4f}")
        
        print("\n🎉 所有验证通过！代码结构正确。")
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 功能验证失败: {e}")
        return False

if __name__ == "__main__":
    success = validate_imports()
    sys.exit(0 if success else 1)