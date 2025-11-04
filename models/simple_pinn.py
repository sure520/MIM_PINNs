"""
简单的PINN模型，用于测试直接训练器
"""

import torch
import torch.nn as nn


class SimplePINN(nn.Module):
    """
    简单的物理信息神经网络模型
    """
    
    def __init__(self, input_dim=1, hidden_dim=50, output_dim=1, num_layers=3):
        """
        初始化简单PINN模型
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_layers: 隐藏层数量
        """
        super(SimplePINN, self).__init__()
        
        # 输入层
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        
        # 隐藏层
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 特征值参数（可训练）
        self.omega2 = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, input_dim]
            
        Returns:
            y1: 函数值 [batch_size]
            y2: 一阶导数 [batch_size]
            y3: 二阶导数 [batch_size]
            y4: 三阶导数 [batch_size]
            omega2: 特征值 [batch_size]
        """
        # 确保输入是二维的
        if x.dim() == 1:
            x = x.unsqueeze(1)
        
        # 计算网络输出
        y1 = self.network(x).squeeze()
        
        # 计算导数
        y1_x = torch.autograd.grad(
            y1, x, 
            grad_outputs=torch.ones_like(y1),
            create_graph=True,
            retain_graph=True
        )[0].squeeze()
        
        y2_x = torch.autograd.grad(
            y1_x, x,
            grad_outputs=torch.ones_like(y1_x),
            create_graph=True,
            retain_graph=True
        )[0].squeeze()
        
        y3_x = torch.autograd.grad(
            y2_x, x,
            grad_outputs=torch.ones_like(y2_x),
            create_graph=True,
            retain_graph=True
        )[0].squeeze()
        
        # 广播特征值到与x相同的长度
        omega2_val = self.omega2.expand_as(y1)
        
        # 返回所有变量（为了与MIM模型兼容）
        return y1, y1_x, y2_x, y3_x, omega2_val
    
    def compute_pde_residual(self, x, T=600, v=50):
        """
        计算PDE残差
        
        Args:
            x: 输入点
            T: 方程参数T
            v: 方程参数v
            
        Returns:
            residual: PDE残差
        """
        y1, y1_x, y2_x, y3_x, omega2 = self.forward(x)
        
        # 计算四阶导数
        y4_x = torch.autograd.grad(
            y3_x, x,
            grad_outputs=torch.ones_like(y3_x),
            create_graph=True,
            retain_graph=True
        )[0].squeeze()
        
        # 计算PDE残差: y'''' - ((T+vx)y'' + vy' + ω²y) = 0
        residual = y4_x - ((T + v * x.squeeze()) * y2_x + v * y1_x + omega2 * y1)
        
        return residual
    
    def compute_boundary_residual(self, x_b):
        """
        计算边界条件残差
        
        Args:
            x_b: 边界点
            
        Returns:
            residual: 边界残差
        """
        y1, _, _, _, _ = self.forward(x_b)
        
        # 边界条件: y(0)=0, y(1)=0
        residual = y1**2
        
        return residual