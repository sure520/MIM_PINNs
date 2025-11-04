"""
测试修改后的direct_trainer.py与新的MIMHomPINNFusion模型的兼容性
"""

import torch
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_import():
    """测试模型导入"""
    try:
        from models.fusion_model import MIMHomPINNFusion
        print("✓ MIMHomPINNFusion模型导入成功")
        return True
    except ImportError as e:
        print(f"✗ MIMHomPINNFusion模型导入失败: {e}")
        return False

def test_trainer_import():
    """测试训练器导入"""
    try:
        from utils.direct_trainer import DirectTrainer
        print("✓ DirectTrainer训练器导入成功")
        return True
    except ImportError as e:
        print(f"✗ DirectTrainer训练器导入失败: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    try:
        from models.fusion_model import MIMHomPINNFusion
        
        # 创建模型实例
        model = MIMHomPINNFusion(
            input_dim=1,
            output_dim=5,  # y1, y2, y3, y4, omega2
            hidden_dim=50,
            num_layers=4
        )
        print("✓ MIMHomPINNFusion模型创建成功")
        
        # 测试前向传播
        x = torch.tensor([[0.5]], dtype=torch.float32)
        output = model(x)
        print(f"✓ 模型前向传播成功，输出形状: {[t.shape for t in output]}")
        
        return True
    except Exception as e:
        print(f"✗ 模型创建或前向传播失败: {e}")
        return False

def test_loss_functions():
    """测试损失函数"""
    try:
        from models.fusion_model import MIMHomPINNFusion
        
        model = MIMHomPINNFusion(
            input_dim=1,
            output_dim=5,
            hidden_dim=50,
            num_layers=4
        )
        
        # 测试数据
        x = torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.float32)
        x_b = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
        
        # 测试总损失函数
        total_loss, loss_dict = model.compute_total_loss(
            x=x, 
            x_b=x_b, 
            T=600, 
            v=50, 
            omega2=None
        )
        
        print("✓ 总损失函数计算成功")
        print(f"  总损失: {total_loss.item():.6f}")
        print(f"  残差损失: {loss_dict['residual_loss'].item():.6f}")
        print(f"  边界损失: {loss_dict['boundary_loss'].item():.6f}")
        print(f"  振幅损失: {loss_dict['amplitude_loss'].item():.6f}")
        print(f"  非零解损失: {loss_dict['nonzero_loss'].item():.6f}")
        print(f"  特征值: {loss_dict['omega2'].item():.6f}")
        
        return True
    except Exception as e:
        print(f"✗ 损失函数测试失败: {e}")
        return False

def test_trainer_compatibility():
    """测试训练器与模型的兼容性"""
    try:
        from models.fusion_model import MIMHomPINNFusion
        from utils.direct_trainer import DirectTrainer
        
        # 创建模型
        model = MIMHomPINNFusion(
            input_dim=1,
            output_dim=5,
            hidden_dim=50,
            num_layers=4
        )
        
        # 创建数据生成器（模拟）
        class MockDataGenerator:
            def generate_all_data(self, N_f, N_b, N_test, domain):
                import numpy as np
                x = np.random.uniform(domain[0], domain[1], (N_f, 1))
                x_b = np.array([[domain[0]], [domain[1]]])
                x_test = np.random.uniform(domain[0], domain[1], (N_test, 1))
                return x, x_b, x_test
        
        data_gen = MockDataGenerator()
        
        # 创建训练器
        trainer = DirectTrainer(
            model=model,
            data_gen=data_gen,
            config_type='balanced'
        )
        
        print("✓ 训练器与模型兼容性测试成功")
        
        # 测试损失计算
        total_loss, loss_dict = trainer.compute_loss(trainer.x, trainer.x_b)
        print(f"  训练器损失计算成功，总损失: {total_loss.item():.6f}")
        
        return True
    except Exception as e:
        print(f"✗ 训练器兼容性测试失败: {e}")
        return False

def test_training_step():
    """测试单步训练"""
    try:
        from models.fusion_model import MIMHomPINNFusion
        from utils.direct_trainer import DirectTrainer
        
        # 创建模型
        model = MIMHomPINNFusion(
            input_dim=1,
            output_dim=5,
            hidden_dim=50,
            num_layers=4
        )
        
        # 创建数据生成器（模拟）
        class MockDataGenerator:
            def generate_all_data(self, N_f, N_b, N_test, domain):
                import numpy as np
                x = np.random.uniform(domain[0], domain[1], (N_f, 1))
                x_b = np.array([[domain[0]], [domain[1]]])
                x_test = np.random.uniform(domain[0], domain[1], (N_test, 1))
                return x, x_b, x_test
        
        data_gen = MockDataGenerator()
        
        # 创建训练器
        trainer = DirectTrainer(
            model=model,
            data_gen=data_gen,
            config_type='balanced'
        )
        
        # 测试单步训练
        total_loss, loss_dict = trainer._train_step()
        print("✓ 单步训练测试成功")
        print(f"  训练后总损失: {total_loss.item():.6f}")
        
        return True
    except Exception as e:
        print(f"✗ 单步训练测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试direct_trainer.py与MIMHomPINNFusion模型的兼容性")
    print("=" * 60)
    
    tests = [
        ("模型导入测试", test_model_import),
        ("训练器导入测试", test_trainer_import),
        ("模型创建测试", test_model_creation),
        ("损失函数测试", test_loss_functions),
        ("训练器兼容性测试", test_trainer_compatibility),
        ("单步训练测试", test_training_step)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 60)
    print("测试结果汇总:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总测试数: {total}, 通过: {passed}, 失败: {total - passed}")
    
    if passed == total:
        print("\n🎉 所有测试通过！direct_trainer.py与MIMHomPINNFusion模型兼容性良好")
        return True
    else:
        print("\n⚠️ 部分测试失败，需要检查兼容性问题")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)