"""
起始系统基类

定义起始系统的统一接口，所有具体起始系统都应继承此类。
基于HomPINNs论文中的起始系统设计原则。
"""

import torch
import numpy as np
from abc import ABC, abstractmethod


class BaseStartingSystem(ABC):
    """起始系统基类"""
    
    def __init__(self, name: str, dimension: int = 1):
        """
        初始化起始系统
        
        Args:
            name: 起始系统名称
            dimension: 问题维度 (1D, 2D等)
        """
        self.name = name
        self.dimension = dimension
        
    @abstractmethod
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算起始系统在给定点的值
        
        Args:
            x: 输入点坐标，形状为 (N, d)，其中d为维度
            
        Returns:
            u: 起始函数值，形状为 (N,)
        """
        pass
    
    @abstractmethod
    def get_residual(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        计算起始系统的残差
        
        Args:
            x: 输入点坐标
            u: 函数值（可以是神经网络输出）
            
        Returns:
            residual: 起始系统残差
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> dict:
        """
        获取起始系统的参数
        
        Returns:
            parameters: 参数字典
        """
        pass
    
    def get_description(self) -> str:
        """获取起始系统描述"""
        return f"{self.name} (dimension: {self.dimension})"
    
    def __str__(self):
        return self.get_description()
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', dimension={self.dimension})"