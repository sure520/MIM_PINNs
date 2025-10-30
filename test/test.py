#!/usr/bin/env python3
"""
测试脚本：验证MIM-HomPINNs融合项目的各个组件是否正常工作

这个脚本用于测试项目中的各个模块，确保它们能够正常导入和运行。
"""

import os
import sys
import torch
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试模块导入"""
    print("测试模块导入...")
    try:
        from configs.config import get_config, update_config
        print("  ✓ 配置模块导入成功")
    except ImportError as e:
        print(f"  ✗ 配置模块导入失败: {e}")
        return False
    
    try:
        from utils.data_generator import DataGenerator
        print("  ✓ 数据生成器模块导入成功")
    except ImportError as e:
        print(f"  ✗ 数据生成器模块导入失败: {e}")
        return False
    
    try:
        from models.fusion_model import MIMHomPINNFusion
        print("  ✓ 模型模块导入成功")
    except ImportError as e:
        print(f"  ✗ 模型模块导入失败: {e}")
        return False
    
    try:
        from utils.training import Trainer
        print("  ✓ 训练器模块导入成功")
    except ImportError as e:
        print(f"  ✗ 训练器模块导入失败: {e}")
        return False
    
    try:
        from utils.evaluation import Evaluator
        print("  ✓ 评估器模块导入成功")
    except ImportError as e:
        print(f"  ✗ 评估器模块导入失败: {e}")
        return False
    
    try:
        from utils.visualization import Visualizer
        print("  ✓ 可视化模块导入成功")
    except ImportError as e:
        print(f"  ✗ 可视化模块导入失败: {e}")
        return False
    
    return True


def test_config():
    """测试配置模块"""
    print("\n测试配置模块...")
    try:
        from configs.config import get_config, update_config
        
        # 获取默认配置
        config = get_config()
        print("  ✓ 默认配置获取成功")
        
        # 更新配置
        update_config(config, model_type='MIM1', T=600, v=50)
        print("  ✓ 配置更新成功")
        
        # 验证配置内容
        assert config['model']['type'] == 'MIM1'
        assert config['equation']['T'] == 600
        assert config['equation']['v'] == 50
        print("  ✓ 配置内容验证成功")
        
        return True
    except Exception as e:
        print(f"  ✗ 配置模块测试失败: {e}")
        return False


def test_data_generator():
    """测试数据生成器"""
    print("\n测试数据生成器...")
    try:
        from configs.config import get_config
        from utils.data_generator import DataGenerator
        
        config = get_config()
        data_gen = DataGenerator(config['data'])
        print("  ✓ 数据生成器创建成功")
        
        # 生成配置点
        x_collocation = data_gen.generate_collocation_points()
        assert x_collocation.shape[0] == config['data']['n_collocation']
        assert x_collocation.shape[1] == 1
        print("  ✓ 配置点生成成功")
        
        # 生成边界点
        x_boundary = data_gen.generate_boundary_points()
        assert x_boundary.shape[0] == 2  # 两个边界点
        assert x_boundary.shape[1] == 1
        print("  ✓ 边界点生成成功")
        
        # 生成测试点
        x_test = data_gen.generate_test_points()
        assert x_test.shape[0] == config['data']['n_test']
        assert x_test.shape[1] == 1
        print("  ✓ 测试点生成成功")
        
        # 生成起始函数
        x = torch.linspace(0, 1, 100).unsqueeze(1)
        y_start, omega2_start = data_gen.generate_starting_function(x, k=1)
        assert y_start.shape == x.shape
        assert isinstance(omega2_start, float)
        print("  ✓ 起始函数生成成功")
        
        return True
    except Exception as e:
        print(f"  ✗ 数据生成器测试失败: {e}")
        return False


def test_model():
    """测试模型"""
    print("\n测试模型...")
    try:
        from configs.config import get_config
        from models.fusion_model import MIMHomPINNFusion
        
        config = get_config()
        
        # 测试MIM1模型
        config['model']['type'] = 'MIM1'
        model1 = MIMHomPINNFusion(config['model'])
        print("  ✓ MIM1模型创建成功")
        
        # 测试前向传播
        x = torch.linspace(0, 1, 10).unsqueeze(1)
        y1, y2, y3, y4, omega2 = model1(x)
        assert y1.shape == x.shape
        assert y2.shape == x.shape
        assert y3.shape == x.shape
        assert y4.shape == x.shape
        assert isinstance(omega2, torch.Tensor)
        print("  ✓ MIM1前向传播成功")
        
        # 测试残差计算
        T, v = 600, 50
        R1, R2, R3, R4, y1, y2, y3, y4, omega2 = model1.compute_residuals(x, T, v, omega2)
        assert R1.shape == x.shape
        assert R2.shape == x.shape
        assert R3.shape == x.shape
        assert R4.shape == x.shape
        print("  ✓ MIM1残差计算成功")
        
        # 测试MIM2模型
        config['model']['type'] = 'MIM2'
        model2 = MIMHomPINNFusion(config['model'])
        print("  ✓ MIM2模型创建成功")
        
        # 测试前向传播
        y1, y2, y3, y4, omega2 = model2(x)
        assert y1.shape == x.shape
        assert y2.shape == x.shape
        assert y3.shape == x.shape
        assert y4.shape == x.shape
        assert isinstance(omega2, torch.Tensor)
        print("  ✓ MIM2前向传播成功")
        
        return True
    except Exception as e:
        print(f"  ✗ 模型测试失败: {e}")
        return False


def test_training():
    """测试训练器"""
    print("\n测试训练器...")
    try:
        from configs.config import get_config
        from utils.data_generator import DataGenerator
        from models.fusion_model import MIMHomPINNFusion
        from utils.training import Trainer
        
        config = get_config()
        config['training']['epochs'] = 10  # 减少训练轮数以加快测试速度
        config['training']['n_homotopy_steps'] = 2  # 减少同伦步骤数
        
        data_gen = DataGenerator(config['data'])
        model = MIMHomPINNFusion(config['model'])
        trainer = Trainer(model, data_gen, **config['training'])
        print("  ✓ 训练器创建成功")
        
        # 测试权重初始化
        model.apply(trainer._weights_init)
        print("  ✓ 权重初始化成功")
        
        # 测试训练步骤
        x_collocation = data_gen.generate_collocation_points()
        x_boundary = data_gen.generate_boundary_points()
        
        # 使用模型计算损失
        loss, F, G, R_b, L_nonzero = model.compute_homotopy_loss(
            x_collocation, x_boundary, t=0.5, T=600, v=50, 
            omega2=trainer.omega2, alpha=trainer.alpha, k=1
        )
        assert isinstance(loss, torch.Tensor)
        print("  ✓ 损失计算成功")
        
        return True
    except Exception as e:
        print(f"  ✗ 训练器测试失败: {e}")
        return False


def test_evaluation():
    """测试评估器"""
    print("\n测试评估器...")
    try:
        from configs.config import get_config
        from utils.data_generator import DataGenerator
        from models.fusion_model import MIMHomPINNFusion
        from utils.evaluation import Evaluator
        
        config = get_config()
        data_gen = DataGenerator(config['data'])
        model = MIMHomPINNFusion(config['model'])
        evaluator = Evaluator(model, data_gen)
        print("  ✓ 评估器创建成功")
        
        # 测试评估
        omega2 = 100.0
        residual_error, boundary_error = evaluator.evaluate(model, omega2, T=600, v=50)
        assert isinstance(residual_error, float)
        assert isinstance(boundary_error, float)
        print("  ✓ 评估功能成功")
        
        # 测试解范数计算
        solution_norm = evaluator.compute_solution_norm(model)
        assert isinstance(solution_norm, float)
        print("  ✓ 解范数计算成功")
        
        # 测试导数范数计算
        norms = evaluator.compute_derivative_norms(model)
        assert len(norms) == 4
        assert all(isinstance(norm, float) for norm in norms)
        print("  ✓ 导数范数计算成功")
        
        return True
    except Exception as e:
        print(f"  ✗ 评估器测试失败: {e}")
        return False


def test_visualization():
    """测试可视化模块"""
    print("\n测试可视化模块...")
    try:
        from utils.visualization import Visualizer
        
        # 创建临时保存目录
        save_dir = 'temp_test_results'
        os.makedirs(save_dir, exist_ok=True)
        
        visualizer = Visualizer(save_dir)
        print("  ✓ 可视化器创建成功")
        
        # 清理临时目录
        import shutil
        shutil.rmtree(save_dir)
        print("  ✓ 临时目录清理成功")
        
        return True
    except Exception as e:
        print(f"  ✗ 可视化模块测试失败: {e}")
        return False


def main():
    """主函数"""
    print("="*50)
    print("MIM-HomPINNs融合项目组件测试")
    print("="*50)
    
    # 运行所有测试
    tests = [
        test_imports,
        test_config,
        test_data_generator,
        test_model,
        test_training,
        test_evaluation,
        test_visualization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    # 输出测试结果
    print("\n" + "="*50)
    print(f"测试结果: {passed}/{total} 通过")
    if passed == total:
        print("✓ 所有测试通过! 项目组件工作正常。")
    else:
        print("✗ 部分测试失败，请检查相关模块。")
    print("="*50)


if __name__ == "__main__":
    main()