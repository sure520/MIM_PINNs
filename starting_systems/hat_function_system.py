"""
1D帽函数起始系统

基于HomPINNs论文中的10个起始函数设计，使用帽函数作为基函数。
参考论文：Homotopy physics-informed neural networks for learning multiple solutions of nonlinear elliptic differential equations
"""

import torch
import numpy as np
from .base_system import BaseStartingSystem


class HatFunctionStartingSystem(BaseStartingSystem):
    """1D帽函数起始系统"""
    
    # 10个起始函数的系数组合（基于论文中的设计）
    STARTING_FUNCTIONS_COEFFS = [
        (1, np.sqrt(2), -1),           # 起始函数1
        (-1, -np.sqrt(2), 1),          # 起始函数2
        (np.sqrt(2), 1, -np.sqrt(2)),  # 起始函数3
        (-np.sqrt(2), -1, np.sqrt(2)), # 起始函数4
        (1, -1, np.sqrt(2)),           # 起始函数5
        (-1, np.sqrt(2), -np.sqrt(2)), # 起始函数6
        (np.sqrt(2), -np.sqrt(2), 1),  # 起始函数7
        (-np.sqrt(2), 1, -1),          # 起始函数8
        (1, -np.sqrt(2), -np.sqrt(2)), # 起始函数9
        (-1, 1, np.sqrt(2))            # 起始函数10
    ]
    
    def __init__(self, function_index: int = 0, name: str = None):
        """
        初始化帽函数起始系统
        
        Args:
            function_index: 起始函数索引 (0-9)
            name: 起始系统名称，如果为None则自动生成
        """
        if function_index < 0 or function_index >= len(self.STARTING_FUNCTIONS_COEFFS):
            raise ValueError(f"function_index必须在0-{len(self.STARTING_FUNCTIONS_COEFFS)-1}范围内")
        
        self.function_index = function_index
        self.coefficients = self.STARTING_FUNCTIONS_COEFFS[function_index]
        
        if name is None:
            name = f"HatFunctionSystem_{function_index + 1}"
            
        super().__init__(name, dimension=1)
    
    def _hat_function_0(self, x: torch.Tensor) -> torch.Tensor:
        """帽函数0: 仅在[0,0.5]非零"""
        return torch.where((x >= 0) & (x <= 0.5), 1 - 2 * x, torch.zeros_like(x))
    
    def _hat_function_1(self, x: torch.Tensor) -> torch.Tensor:
        """帽函数1: 在[0,1]非零"""
        result = torch.zeros_like(x)
        mask1 = (x >= 0) & (x <= 0.5)
        mask2 = (x > 0.5) & (x <= 1)
        
        result[mask1] = 2 * x[mask1]
        result[mask2] = 2 - 2 * x[mask2]
        
        return result
    
    def _hat_function_2(self, x: torch.Tensor) -> torch.Tensor:
        """帽函数2: 仅在[0.5,1]非零"""
        return torch.where((x >= 0.5) & (x <= 1), 2 * x - 1, torch.zeros_like(x))
    
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算帽函数起始系统在给定点的值
        
        Args:
            x: 输入点坐标，形状为 (N, 1)
            
        Returns:
            u: 起始函数值，形状为 (N,)
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        
        # 确保x在[0,1]范围内
        x_clamped = torch.clamp(x, 0, 1)
        
        # 计算三个帽函数
        phi0 = self._hat_function_0(x_clamped).squeeze(-1)
        phi1 = self._hat_function_1(x_clamped).squeeze(-1)
        phi2 = self._hat_function_2(x_clamped).squeeze(-1)
        
        # 线性组合
        v0, v1, v2 = self.coefficients
        u = v0 * phi0 + v1 * phi1 + v2 * phi2
        
        return u
    
    def get_residual(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        计算起始系统的残差
        
        基于HomPINNs论文，对于非线性椭圆方程，起始系统残差为：
        G(u) = u^4 - u_s^4
        
        Args:
            x: 输入点坐标
            u: 函数值（可以是神经网络输出）
            
        Returns:
            residual: 起始系统残差
        """
        # 计算起始函数值
        u_s = self.evaluate(x)
        
        # 计算残差：G(u) = u^4 - u_s^4
        residual = u**4 - u_s**4
        
        return residual
    
    def get_parameters(self) -> dict:
        """获取起始系统的参数"""
        return {
            'function_index': self.function_index,
            'coefficients': self.coefficients,
            'dimension': self.dimension,
            'name': self.name
        }
    
    def get_description(self) -> str:
        """获取起始系统详细描述"""
        v0, v1, v2 = self.coefficients
        return f"{self.name} (系数: v0={v0:.3f}, v1={v1:.3f}, v2={v2:.3f})"
    
    @classmethod
    def get_all_systems(cls):
        """获取所有10个帽函数起始系统"""
        return [cls(i) for i in range(len(cls.STARTING_FUNCTIONS_COEFFS))]
    
    @classmethod
    def get_random_system(cls):
        """获取随机帽函数起始系统"""
        idx = np.random.randint(0, len(cls.STARTING_FUNCTIONS_COEFFS))
        return cls(idx)