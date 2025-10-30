import os
import torch
import numpy as np

def test_basic_functionality():
    """测试基本功能"""
    print("="*50)
    print("MIM-HomPINNs融合项目基本功能测试")
    print("="*50)
    
    # 测试导入
    print("\n测试导入...")
    try:
        from configs.config import get_config
        print("  ✓ 配置导入成功")
    except Exception as e:
        print(f"  ✗ 配置导入失败: {e}")
        return False
    
    try:
        from utils.data_generator import DataGenerator
        print("  ✓ 数据生成器导入成功")
    except Exception as e:
        print(f"  ✗ 数据生成器导入失败: {e}")
        return False
    
    try:
        from models.fusion_model import MIMHomPINNFusion
        print("  ✓ 模型导入成功")
    except Exception as e:
        print(f"  ✗ 模型导入失败: {e}")
        return False
    
    # 测试配置
    print("\n测试配置...")
    try:
        config = get_config()
        print("  ✓ 配置加载成功")
    except Exception as e:
        print(f"  ✗ 配置加载失败: {e}")
        return False
    
    # 测试数据生成器
    print("\n测试数据生成器...")
    try:
        data_gen = DataGenerator(config['data'])
        x = data_gen.generate_collocation_points(N_g=10)
        x_b = data_gen.generate_boundary_points(N_b=4)
        x_test = data_gen.generate_test_points(N_test=20)
        print("  ✓ 数据生成成功")
    except Exception as e:
        print(f"  ✗ 数据生成失败: {e}")
        return False
    
    # 测试MIM1模型
    print("\n测试MIM1模型...")
    try:
        # 直接传递设备字符串而不是整个配置
        device_str = config['device']['device'] if isinstance(config['device'], dict) else config['device']
        
        # 使用正确的参数创建模型
        model1 = MIMHomPINNFusion(
            width=config['model']['resnet_neurons'],  # 使用resnet_neurons作为width
            depth=config['model']['resnet_layers'],   # 使用resnet_layers作为depth
            model_type='MIM1',
            device=device_str
        )
        print("  ✓ MIM1模型创建成功")
        
        # 测试前向传播
        y1, y2, y3, y4, omega2 = model1(x)
        print(f"  ✓ MIM1前向传播成功: y1.shape={y1.shape}, omega2.shape={omega2.shape}")
        
        # 测试残差计算
        T, v = 600, 50
        R1, R2, R3, R4, y1, y2, y3, y4, omega2_val = model1.compute_residuals(x, T, v)
        print(f"  ✓ MIM1残差计算成功: R1.shape={R1.shape}")
        
        # 测试边界残差计算
        R_b, y1_b, y3_b = model1.compute_boundary_residuals(x_b)
        print(f"  ✓ MIM1边界残差计算成功: R_b.shape={R_b.shape}")
        
        # 测试同伦损失计算
        loss, F, G, R_b, L_nonzero = model1.compute_homotopy_loss(
            x, x_b, t=0.5, T=T, v=v, omega2=100.0, alpha=10, k=1
        )
        print(f"  ✓ MIM1同伦损失计算成功: loss={loss.item():.6f}")
        
    except Exception as e:
        print(f"  ✗ MIM1模型测试失败: {e}")
        return False
    
    print("\n" + "="*50)
    print("✓ 所有基本功能测试通过! 项目核心组件工作正常。")
    print("="*50)
    return True


if __name__ == "__main__":
    test_basic_functionality()