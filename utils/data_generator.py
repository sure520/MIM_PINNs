import torch
import numpy as np


class DataGenerator:
    """
    数据生成器，用于生成训练和测试数据
    """
    def __init__(self, config=None, device='cpu'):
        if config is not None:
            # 优先使用传入的device参数
            if device != 'cpu':
                self.device = device
            else:
                # 如果device参数是默认值，则从config中获取
                self.device = config.get('device', 'cpu')
                # 如果config是包含device键的字典，则提取设备字符串
                if isinstance(self.device, dict) and 'device' in self.device:
                    self.device = self.device['device']
        else:
            self.device = device
    
    def generate_collocation_points(self, N_g=100):
        """
        生成内部配置点
        Args:
            N_g: 内部点数量
        Returns:
            x: 内部点张量，形状为(N_g, 1)
        """
        # 在[0,1]区间内均匀采样
        x = torch.linspace(0, 1, N_g, device=self.device).reshape(-1, 1)
        x.requires_grad = True  # 需要梯度用于自动微分
        return x
    
    def generate_boundary_points(self, N_b=4):
        """
        生成边界点
        Args:
            N_b: 边界点数量
        Returns:
            x_b: 边界点张量，形状为(N_b, 1)
        """
        # 边界点包括x=0和x=1
        x_b = torch.tensor([[0.0], [1.0], [0.0], [1.0]], device=self.device)
        x_b.requires_grad = True  # 需要梯度用于自动微分
        return x_b
    
    def generate_test_points(self, N_test=1000):
        """
        生成测试点
        Args:
            N_test: 测试点数量
        Returns:
            x_test: 测试点张量，形状为(N_test, 1)
        """
        # 在[0,1]区间内均匀采样
        x_test = torch.linspace(0, 1, N_test, device=self.device).reshape(-1, 1)
        return x_test
    
    def generate_starting_function(self, x, k=1):
        """
        生成起始函数
        Args:
            x: 输入点
            k: 起始函数的k值
        Returns:
            y0: 起始函数值
            omega0_2: 起始特征值
        """
        # 起始函数: y0(x) = sin(kπx)
        # 使用torch.pi确保张量在正确设备上
        y0 = torch.sin(k * torch.pi * x[:, 0])
        
        # 起始特征值: ω₀² = (kπ)⁴
        omega0_2 = (k * torch.pi)**4
        
        return y0, omega0_2
    
    def generate_all_data(self, N_g=100, N_b=4, N_test=1000):
        """
        生成所有所需的数据点
        Args:
            N_g: 内部点数量
            N_b: 边界点数量
            N_test: 测试点数量
        Returns:
            x: 内部点
            x_b: 边界点
            x_test: 测试点
        """
        x = self.generate_collocation_points(N_g)
        x_b = self.generate_boundary_points(N_b)
        x_test = self.generate_test_points(N_test)
        
        return x, x_b, x_test