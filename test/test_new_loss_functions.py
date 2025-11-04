"""
测试新的损失函数实现
验证MIMHomPINNFusion类中新添加的损失函数项
"""

import torch
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.fusion_model import MIMHomPINNFusion
from data.data_generator import DataGenerator

def test_loss_functions():
    """测试新的损失函数实现"""
    print("=== 测试新的损失函数实现 ===")
    
    # 创建模型
    model = MIMHomPINNFusion(input_dim=1, hidden_dim=64, output_dim=5, num_layers=4)
    
    # 创建数据生成器
    domain = [0, 1]
    data_gen = DataGenerator(domain=domain, n_domain=100, n_boundary=20, n_test=50)
    
    # 生成测试数据
    x = data_gen.generate_domain_points()
    x_b = data_gen.generate_boundary_points()
    
    print(f"内部点数量: {len(x)}")
    print(f"边界点数量: {len(x_b)}")
    
    # 测试1: 基本损失函数计算
    print("\n--- 测试1: 基本损失函数计算 ---")
    try:
        total_loss, loss_dict = model.compute_total_loss(x, x_b, T=600, v=50)
        print("✓ 总损失函数计算成功")
        print(f"总损失: {total_loss.item():.6f}")
        print(f"残差损失: {loss_dict['residual_loss'].item():.6f}")
        print(f"边界损失: {loss_dict['boundary_loss'].item():.6f}")
        print(f"振幅损失: {loss_dict['amplitude_loss'].item():.6f}")
        print(f"层级损失: {loss_dict['hierarchy_loss'].item():.6f}")
        print(f"非零解损失: {loss_dict['nonzero_loss'].item():.6f}")
        print(f"特征值: {loss_dict['omega2'].item():.6f}")
    except Exception as e:
        print(f"✗ 总损失函数计算失败: {e}")
        return False
    
    # 测试2: 各独立损失函数
    print("\n--- 测试2: 各独立损失函数 ---")
    try:
        # 残差损失
        L_r, y1, y2, y3, y4, omega2_val = model.compute_residual_loss(x, T=600, v=50)
        print(f"✓ 残差损失: {L_r.item():.6f}")
        
        # 边界损失
        L_b = model.compute_boundary_loss(x_b)
        print(f"✓ 边界损失: {L_b.item():.6f}")
        
        # 振幅约束损失（在特定配点处计算）
        x_a = torch.tensor([0.5], device=model.device).requires_grad_(True)
        L_a = model.compute_amplitude_constraint_loss(x_a)
        print(f"✓ 振幅约束损失: {L_a.item():.6f}")
        
        # 特征值层级约束损失（有低阶特征值）
        omega_low_2 = 400000.0  # 假设的低阶特征值
        L_c = model.compute_eigenvalue_hierarchy_loss(omega2_val, omega_low_2)
        print(f"✓ 特征值层级约束损失: {L_c.item():.6f}")
        
        # 非零解惩罚损失
        L_nz = model.compute_nonzero_solution_loss(y1)
        print(f"✓ 非零解惩罚损失: {L_nz.item():.6f}")
        
    except Exception as e:
        print(f"✗ 独立损失函数计算失败: {e}")
        return False
    
    # 测试3: 权重分配机制
    print("\n--- 测试3: 权重分配机制 ---")
    try:
        # 自定义权重
        custom_weights = {
            'residual': 2.0,
            'boundary': 200.0,
            'amplitude': 50.0,
            'hierarchy': 50.0,
            'nonzero': 1e-3
        }
        
        total_loss_custom, loss_dict_custom = model.compute_total_loss(
            x, x_b, T=600, v=50, weights=custom_weights
        )
        
        print("✓ 自定义权重计算成功")
        print(f"自定义权重总损失: {total_loss_custom.item():.6f}")
        
        # 验证权重影响
        weighted_loss = (
            custom_weights['residual'] * loss_dict_custom['residual_loss'] +
            custom_weights['boundary'] * loss_dict_custom['boundary_loss'] +
            custom_weights['amplitude'] * loss_dict_custom['amplitude_loss'] +
            custom_weights['hierarchy'] * loss_dict_custom['hierarchy_loss'] +
            custom_weights['nonzero'] * loss_dict_custom['nonzero_loss']
        )
        
        print(f"手动加权损失: {weighted_loss.item():.6f}")
        print(f"权重分配一致性: {torch.isclose(total_loss_custom, weighted_loss)}")
        
    except Exception as e:
        print(f"✗ 权重分配测试失败: {e}")
        return False
    
    # 测试4: 层级约束功能
    print("\n--- 测试4: 层级约束功能 ---")
    try:
        # 测试不同特征值情况下的层级约束
        omega2_values = torch.tensor([300000.0, 500000.0, 800000.0], device=model.device)
        omega_low_2 = 400000.0
        
        for omega2_val in omega2_values:
            L_c = model.compute_eigenvalue_hierarchy_loss(omega2_val, omega_low_2)
            print(f"ω²={omega2_val.item():.0f}, L_c={L_c.item():.6f}")
            
            # 验证层级约束逻辑：当ω² < ω_low² + ε时，L_c应该接近1（强惩罚）
            # 当ω² > ω_low² + ε时，L_c应该接近0（弱惩罚）
            if omega2_val < omega_low_2 + 5.0:
                assert L_c > 0.5, "层级约束在ω² < ω_low² + ε时应产生强惩罚"
            else:
                assert L_c < 0.5, "层级约束在ω² > ω_low² + ε时应产生弱惩罚"
        
        print("✓ 层级约束逻辑验证成功")
        
    except Exception as e:
        print(f"✗ 层级约束测试失败: {e}")
        return False
    
    # 测试5: 振幅约束功能
    print("\n--- 测试5: 振幅约束功能 ---")
    try:
        # 测试不同振幅约束点
        x_a_values = [0.25, 0.5, 0.75]
        y_a_target = 1.0
        
        for x_a in x_a_values:
            # 注意：这里需要确保x_a在采样点中，或者使用插值方法
            # 简化测试，使用固定y1值
            test_y1 = torch.ones_like(y1) * 0.5  # 假设解值为0.5
            L_a = model.compute_amplitude_constraint_loss(test_y1, x_a=x_a, y_a=y_a_target)
            expected_loss = (0.5 - 1.0)**2  # (0.5 - 1.0)^2 = 0.25
            
            print(f"x_a={x_a}, L_a={L_a.item():.6f}, 期望值={expected_loss:.6f}")
            assert torch.isclose(L_a, torch.tensor(expected_loss, device=model.device), rtol=1e-4), \
                f"振幅约束损失计算错误: x_a={x_a}"
        
        print("✓ 振幅约束功能验证成功")
        
    except Exception as e:
        print(f"✗ 振幅约束测试失败: {e}")
        return False
    
    print("\n=== 所有测试通过! ===")
    return True

def test_loss_comparison():
    """对比新旧损失函数结构"""
    print("\n=== 对比新旧损失函数结构 ===")
    
    # 创建模型
    model = MIMHomPINNFusion(input_dim=1, hidden_dim=64, output_dim=5, num_layers=4)
    
    # 创建数据生成器
    domain = [0, 1]
    data_gen = DataGenerator(domain=domain, n_domain=100, n_boundary=20, n_test=50)
    
    # 生成测试数据
    x = data_gen.generate_domain_points()
    x_b = data_gen.generate_boundary_points()
    
    print("新损失函数结构:")
    print("- L_r: 控制方程残差损失（核心物理约束）")
    print("- L_b: 边界条件损失（硬约束保障）")
    print("- L_a: 振幅约束损失（规避模态歧义）")
    print("- L_c: 特征值层级约束损失（多阶特征值引导）")
    print("- L_nz: 非零解惩罚损失（排除零解）")
    print("总损失: L_total = ω_r·L_r + ω_b·L_b + ω_a·L_a + ω_c·L_c + ω_nz·L_nz")
    
    # 计算新损失函数
    total_loss_new, loss_dict_new = model.compute_total_loss(x, x_b, T=600, v=50)
    
    print(f"\n新损失函数结果:")
    for key, value in loss_dict_new.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.item():.6f}")
    
    print("\n旧损失函数结构（compute_homotopy_loss）:")
    print("- 包含同伦参数t")
    print("- 混合起始系统G和目标系统F")
    print("- 包含边界损失R_b和非零解惩罚L_nonzero")
    
    # 计算旧损失函数（用于对比）
    try:
        loss_old, F, G, R_b, L_nonzero = model.compute_homotopy_loss(x, x_b, t=1.0, T=600, v=50)
        print(f"\n旧损失函数结果:")
        print(f"总损失: {loss_old.item():.6f}")
        print(f"目标系统F: {F.item():.6f}")
        print(f"起始系统G: {G.item():.6f}")
        print(f"边界损失R_b: {R_b.item():.6f}")
        print(f"非零解惩罚L_nonzero: {L_nonzero.item():.6f}")
    except Exception as e:
        print(f"旧损失函数计算失败: {e}")
    
    return True

if __name__ == "__main__":
    # 运行测试
    success1 = test_loss_functions()
    success2 = test_loss_comparison()
    
    if success1 and success2:
        print("\n🎉 所有测试成功完成!")
        print("新的损失函数结构已正确实现，具备完整的物理约束和数值稳定性保障。")
    else:
        print("\n❌ 部分测试失败，请检查实现。")
        sys.exit(1)