"""
数据生成器，用于生成训练和测试数据
"""

import torch
import numpy as np


class DataGenerator:
    """
    数据生成器类
    """
    
    def __init__(self, domain=[0.0, 1.0], n_domain=1000, n_boundary=200, n_test=500):
        """
        初始化数据生成器
        
        Args:
            domain: 计算域 [start, end]
            n_domain: 域内点数
            n_boundary: 边界点数
            n_test: 测试点数
        """
        self.domain = domain
        self.n_domain = n_domain
        self.n_boundary = n_boundary
        self.n_test = n_test
        
        # 生成数据
        self.x_domain = None
        self.x_boundary = None
        self.x_test = None
        
        self.generate_data()
    
    def generate_data(self):
        """生成所有数据点"""
        # 生成域内点（均匀分布）
        x_domain = np.linspace(self.domain[0], self.domain[1], self.n_domain)
        self.x_domain = torch.tensor(x_domain, dtype=torch.float32).unsqueeze(1)
        
        # 生成边界点
        # 重复边界点以达到所需数量
        n_0 = self.n_boundary // 2
        n_1 = self.n_boundary - n_0
        if self.n_boundary > 2:
            x_boundary = np.concatenate([
                np.full(n_0, self.domain[0]),
                np.full(n_1, self.domain[1])
            ])
            # 打乱顺序（可选，让 0 和 1 随机分布）
            np.random.shuffle(x_boundary)
        else:
            x_boundary = np.array([self.domain[0], self.domain[1]])
        self.x_boundary = torch.tensor(x_boundary, dtype=torch.float32).unsqueeze(1)
        
        # 生成测试点
        x_test = np.linspace(self.domain[0], self.domain[1], self.n_test)
        self.x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
    
    def get_training_data(self, device='cuda'):
        """
        获取训练数据
        
        Args:
            device: 设备
            
        Returns:
            x_domain: 域内点
            x_boundary: 边界点
        """
        return self.x_domain.to(device), self.x_boundary.to(device)
    
    def get_test_data(self, device='cuda'):
        """
        获取测试数据
        
        Args:
            device: 设备
            
        Returns:
            x_test: 测试点
        """
        return self.x_test.to(device)
    
    def get_all_data(self, device='cuda'):
        """
        获取所有数据
        
        Args:
            device: 设备
            
        Returns:
            x_domain: 域内点
            x_boundary: 边界点
            x_test: 测试点
        """
        return (
            self.x_domain.to(device),
            self.x_boundary.to(device),
            self.x_test.to(device)
        )
    
    def generate_all_data(self, N_f=None, N_b=None, N_test=None, domain=None, device='cuda'):
        """
        生成所有数据（兼容直接训练器接口）
        
        Args:
            N_f: 域内点数
            N_b: 边界点数
            N_test: 测试点数
            domain: 计算域
            device: 设备
            
        Returns:
            x_domain: 域内点
            x_boundary: 边界点
            x_test: 测试点
        """
        # 如果提供了新参数，则更新配置
        if domain is not None:
            self.domain = domain
        if N_f is not None:
            self.n_domain = N_f
        if N_b is not None:
            self.n_boundary = N_b
        if N_test is not None:
            self.n_test = N_test
            
        # 重新生成数据
        self.generate_data()
        
        # 返回PyTorch张量格式（直接训练器期望的格式）
        return (
            self.x_domain.to(device),
            self.x_boundary.to(device),
            self.x_test.to(device)
        )