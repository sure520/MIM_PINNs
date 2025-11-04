import torch
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.fusion_model import MIMHomPINNFusion
from utils.data_generator import DataGenerator

def test_first_order_eigenvalue():
    """测试一阶特征值训练"""
    print("开始测试一阶特征值训练...")
    
    # 初始化模型
    model = MIMHomPINNFusion(width=30, depth=2, model_type='MIM1', device='cpu')
    
    # 初始化数据生成器
    data_gen = DataGenerator(N_f=2000, N_b=100)
    x_train, x_boundary = data_gen.generate_training_data()
    
    # 设置方程参数
    T = 600.0
    v = 50.0
    
    # 初始化特征值参数
    omega2 = torch.tensor(100.0, dtype=torch.float32, requires_grad=True)
    
    # 测试compute_total_loss函数
    print("测试compute_total_loss函数...")
    try:
        total_loss, loss_dict = model.compute_total_loss(
            x=x_train,
            x_b=x_boundary,
            T=T,
            v=v,
            omega2=omega2,
            k=1,  # 一阶特征值
            weights={
                'residual': 1.0,
                'boundary': 100.0,
                'amplitude': 100.0,
                'hierarchy': 0.0,  # 一阶特征值不使用层级约束
                'nonzero': 1e-4
            }
        )
        
        print(f"Total loss shape: {total_loss.shape}")
        print(f"Total loss value: {total_loss}")
        
        # 尝试调用.item()
        scalar_value = total_loss.item()
        print(f"Total loss as scalar: {scalar_value}")
        
        # 检查各个损失项
        for key, value in loss_dict.items():
            if hasattr(value, 'shape'):
                print(f"{key} shape: {value.shape}, value: {value}")
                if value.numel() > 1:
                    print(f"WARNING: {key} has more than one element!")
            else:
                print(f"{key}: {value}")
                
        print("测试成功完成！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_first_order_eigenvalue()