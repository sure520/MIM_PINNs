"""
高斯函数起始系统

用于2D不规则区域的起始系统设计，基于高斯基函数。
参考论文：HomPINNs论文中的2D心形域示例
"""

import torch
import numpy as np
from .base_system import BaseStartingSystem


class GaussianFunctionStartingSystem(BaseStartingSystem):
    """高斯函数起始系统"""
    
    def __init__(self, centers: list, sigma: float = 0.1, domain: tuple = None, name: str = None):
        """
        初始化高斯函数起始系统
        
        Args:
            centers: 高斯中心点列表，每个中心为(x,y)坐标
            sigma: 高斯函数的标准差
            domain: 定义域范围，如果为None则自动计算
            name: 起始系统名称，如果为None则自动生成
        """
        self.centers = centers
        self.sigma = sigma
        self.domain = domain
        
        if name is None:
            name = f"GaussianFunctionSystem_{len(centers)}centers"
            
        super().__init__(name, dimension=2)
    
    def _gaussian_2d(self, x: torch.Tensor, center: tuple) -> torch.Tensor:
        """
        计算二维高斯函数值
        
        Args:
            x: 输入点坐标，形状为 (N, 2)
            center: 高斯中心点 (cx, cy)
            
        Returns:
            gaussian: 高斯函数值，形状为 (N,)
        """
        cx, cy = center
        
        # 计算到中心的距离平方
        dist_sq = (x[:, 0] - cx)**2 + (x[:, 1] - cy)**2
        
        # 计算高斯函数值
        gaussian = torch.exp(-dist_sq / (2 * self.sigma**2))
        
        return gaussian
    
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算高斯函数起始系统在给定点的值
        
        Args:
            x: 输入点坐标，形状为 (N, 2)
            
        Returns:
            u: 起始函数值，形状为 (N,)
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        
        if x.shape[1] != 2:
            raise ValueError("输入点坐标必须是二维的")
        
        # 计算所有高斯函数的加权和
        u = torch.zeros(x.shape[0], device=x.device)
        
        for i, center in enumerate(self.centers):
            # 每个高斯函数的权重可以调整，这里简单设为1
            weight = 1.0
            gaussian = self._gaussian_2d(x, center)
            u += weight * gaussian
        
        # 归一化
        u_max = torch.max(torch.abs(u))
        if u_max > 0:
            u = u / u_max
        
        return u
    
    def get_residual(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        计算起始系统的残差
        
        对于2D问题，起始系统残差可以定义为：
        G(u) = u - u_s
        
        Args:
            x: 输入点坐标
            u: 函数值（可以是神经网络输出）
            
        Returns:
            residual: 起始系统残差
        """
        # 计算起始函数值
        u_s = self.evaluate(x)
        
        # 计算残差：G(u) = u - u_s
        residual = u - u_s
        
        return residual
    
    def get_laplacian_residual(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        计算拉普拉斯算子的残差（用于泊松方程等）
        
        Args:
            x: 输入点坐标
            u: 函数值
            
        Returns:
            residual: 拉普拉斯残差
        """
        # 启用梯度计算
        x = x.clone().requires_grad_(True)
        
        # 计算一阶导数
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        # 计算二阶导数（拉普拉斯算子）
        u_xx = torch.autograd.grad(u_x[:, 0], x, grad_outputs=torch.ones_like(u_x[:, 0]), create_graph=True)[0][:, 0]
        u_yy = torch.autograd.grad(u_x[:, 1], x, grad_outputs=torch.ones_like(u_x[:, 1]), create_graph=True)[0][:, 1]
        
        laplacian_u = u_xx + u_yy
        
        # 计算起始函数的拉普拉斯
        u_s = self.evaluate(x)
        u_s_x = torch.autograd.grad(u_s, x, grad_outputs=torch.ones_like(u_s), create_graph=True)[0]
        u_s_xx = torch.autograd.grad(u_s_x[:, 0], x, grad_outputs=torch.ones_like(u_s_x[:, 0]), create_graph=True)[0][:, 0]
        u_s_yy = torch.autograd.grad(u_s_x[:, 1], x, grad_outputs=torch.ones_like(u_s_x[:, 1]), create_graph=True)[0][:, 1]
        
        laplacian_u_s = u_s_xx + u_s_yy
        
        # 残差：Δu - Δu_s
        residual = laplacian_u - laplacian_u_s
        
        return residual
    
    def get_parameters(self) -> dict:
        """获取起始系统的参数"""
        return {
            'centers': self.centers,
            'sigma': self.sigma,
            'domain': self.domain,
            'dimension': self.dimension,
            'name': self.name
        }
    
    def get_description(self) -> str:
        """获取起始系统详细描述"""
        return f"{self.name} ({len(self.centers)}个中心点, σ={self.sigma})"
    
    @classmethod
    def create_grid_system(cls, grid_shape: tuple, domain: tuple = (0, 1), sigma: float = 0.1):
        """
        创建网格形式的高斯函数起始系统
        
        Args:
            grid_shape: 网格形状 (nx, ny)
            domain: 定义域范围 ((x_min, x_max), (y_min, y_max))
            sigma: 高斯函数的标准差
            
        Returns:
            GaussianFunctionStartingSystem实例
        """
        if domain is None:
            domain = ((0, 1), (0, 1))
        
        nx, ny = grid_shape
        x_domain, y_domain = domain
        
        # 生成网格点
        x_centers = np.linspace(x_domain[0], x_domain[1], nx)
        y_centers = np.linspace(y_domain[0], y_domain[1], ny)
        
        centers = []
        for x in x_centers:
            for y in y_centers:
                centers.append((x, y))
        
        return cls(centers, sigma, domain)
    
    @classmethod
    def create_circle_system(cls, center: tuple, radius: float, n_points: int = 8, sigma: float = 0.1):
        """
        创建圆形分布的高斯函数起始系统
        
        Args:
            center: 圆心坐标 (cx, cy)
            radius: 圆半径
            n_points: 圆周上的点数
            sigma: 高斯函数的标准差
            
        Returns:
            GaussianFunctionStartingSystem实例
        """
        centers = []
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            centers.append((x, y))
        
        # 添加圆心
        centers.append(center)
        
        return cls(centers, sigma)