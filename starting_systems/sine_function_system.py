"""
正弦函数起始系统

基于MIM论文中的起始系统设计，用于求解四阶微分方程。
起始系统：y^(4) - ω₀²y = 0，解析解为 y(x) = sin(kπx)
参考论文：MIM: A deep mixed residual method for solving high-order partial differential equations
"""

import torch
import numpy as np
from .base_system import BaseStartingSystem


class SineFunctionStartingSystem(BaseStartingSystem):
    """正弦函数起始系统"""
    
    def __init__(self, k: int = 1, domain: tuple = (0, 1), name: str = None):
        """
        初始化正弦函数起始系统
        
        Args:
            k: 波数，决定起始函数的模式
            domain: 定义域范围
            name: 起始系统名称，如果为None则自动生成
        """
        self.k = k
        self.domain = domain
        self.omega0_2 = (k * np.pi)**4  # 起始特征值 ω₀² = (kπ)⁴
        
        if name is None:
            name = f"SineFunctionSystem_k{k}"
            
        super().__init__(name, dimension=1)
    
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算正弦函数起始系统在给定点的值
        
        Args:
            x: 输入点坐标，形状为 (N, 1)
            
        Returns:
            y: 起始函数值，形状为 (N,)
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        
        # 归一化到定义域
        x_min, x_max = self.domain
        x_normalized = (x - x_min) / (x_max - x_min)
        
        # 计算正弦函数：y(x) = sin(kπx)
        y = torch.sin(self.k * np.pi * x_normalized.squeeze(-1))
        
        return y
    
    def get_residual(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        计算起始系统的残差
        
        起始系统：y^(4) - ω₀²y = 0
        残差：R = y^(4) - ω₀²y
        
        Args:
            x: 输入点坐标
            y: 函数值（可以是神经网络输出）
            
        Returns:
            residual: 起始系统残差
        """
        # 启用梯度计算
        x = x.clone().requires_grad_(True)
        
        # 计算一阶导数
        y_x = torch.autograd.grad(
            y, x, grad_outputs=torch.ones_like(y),
            create_graph=True, retain_graph=True
        )[0]
        
        # 计算二阶导数
        y_xx = torch.autograd.grad(
            y_x, x, grad_outputs=torch.ones_like(y_x),
            create_graph=True, retain_graph=True
        )[0]
        
        # 计算三阶导数
        y_xxx = torch.autograd.grad(
            y_xx, x, grad_outputs=torch.ones_like(y_xx),
            create_graph=True, retain_graph=True
        )[0]
        
        # 计算四阶导数
        y_xxxx = torch.autograd.grad(
            y_xxx, x, grad_outputs=torch.ones_like(y_xxx),
            create_graph=True, retain_graph=True
        )[0]
        
        # 计算残差：R = y^(4) - ω₀²y
        residual = y_xxxx - self.omega0_2 * y
        
        return residual
    
    def get_mim_residuals(self, x: torch.Tensor, y: torch.Tensor):
        """
        计算MIM格式的起始系统残差
        
        基于MIM方法，将四阶方程转化为一阶系统：
        y1 = y, y2 = y', y3 = y'', y4 = y'''
        
        Returns:
            residuals: 四个残差项 (R1, R2, R3, R4)
            variables: 四个变量 (y1, y2, y3, y4)
        """
        # 启用梯度计算
        x = x.clone().requires_grad_(True)
        
        # 计算各阶导数
        y1 = y
        y2 = torch.autograd.grad(y1, x, grad_outputs=torch.ones_like(y1), create_graph=True)[0]
        y3 = torch.autograd.grad(y2, x, grad_outputs=torch.ones_like(y2), create_graph=True)[0]
        y4 = torch.autograd.grad(y3, x, grad_outputs=torch.ones_like(y3), create_graph=True)[0]
        
        # 计算MIM残差
        R1 = y2 - torch.autograd.grad(y1, x, grad_outputs=torch.ones_like(y1), create_graph=True)[0]
        R2 = y3 - torch.autograd.grad(y2, x, grad_outputs=torch.ones_like(y2), create_graph=True)[0]
        R3 = y4 - torch.autograd.grad(y3, x, grad_outputs=torch.ones_like(y3), create_graph=True)[0]
        R4 = torch.autograd.grad(y4, x, grad_outputs=torch.ones_like(y4), create_graph=True)[0] - self.omega0_2 * y1
        
        residuals = (R1, R2, R3, R4)
        variables = (y1, y2, y3, y4)
        
        return residuals, variables
    
    def get_parameters(self) -> dict:
        """获取起始系统的参数"""
        return {
            'k': self.k,
            'omega0_2': self.omega0_2,
            'domain': self.domain,
            'dimension': self.dimension,
            'name': self.name
        }
    
    def get_description(self) -> str:
        """获取起始系统详细描述"""
        return f"{self.name} (k={self.k}, ω₀²={(self.k * np.pi)**4:.2f}, domain={self.domain})"
    
    @classmethod
    def get_multiple_systems(cls, k_list: list, domain: tuple = (0, 1)):
        """获取多个正弦函数起始系统"""
        return [cls(k, domain) for k in k_list]