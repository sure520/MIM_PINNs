"""
起始系统管理器

用于统一管理和调度不同的起始系统，支持动态注册、查询和选择起始系统。
"""

import torch
from typing import Dict, List, Optional, Type, Union
from .base_system import BaseStartingSystem
from .hat_function_system import HatFunctionStartingSystem
from .sine_function_system import SineFunctionStartingSystem
from .gaussian_function_system import GaussianFunctionStartingSystem


class StartingSystemManager:
    """起始系统管理器"""
    
    def __init__(self):
        """初始化起始系统管理器"""
        self._systems: Dict[str, BaseStartingSystem] = {}
        self._system_types: Dict[str, Type[BaseStartingSystem]] = {}
        
        # 注册内置起始系统类型
        self.register_system_type('hat_function', HatFunctionStartingSystem)
        self.register_system_type('sine_function', SineFunctionStartingSystem)
        self.register_system_type('gaussian_function', GaussianFunctionStartingSystem)
    
    def register_system(self, system: BaseStartingSystem) -> None:
        """
        注册起始系统实例
        
        Args:
            system: 起始系统实例
        """
        self._systems[system.name] = system
    
    def register_system_type(self, type_name: str, system_class: Type[BaseStartingSystem]) -> None:
        """
        注册起始系统类型
        
        Args:
            type_name: 系统类型名称
            system_class: 起始系统类
        """
        self._system_types[type_name] = system_class
    
    def create_system(self, 
                     system_type: str, 
                     name: str, 
                     **kwargs) -> BaseStartingSystem:
        """
        创建起始系统实例
        
        Args:
            system_type: 系统类型名称
            name: 系统名称
            **kwargs: 系统参数
            
        Returns:
            system: 起始系统实例
        """
        if system_type not in self._system_types:
            raise ValueError(f"未知的起始系统类型: {system_type}")
        
        system_class = self._system_types[system_type]
        system = system_class(name=name, **kwargs)
        
        # 自动注册
        self.register_system(system)
        
        return system
    
    def get_system(self, name: str) -> Optional[BaseStartingSystem]:
        """
        获取起始系统实例
        
        Args:
            name: 系统名称
            
        Returns:
            system: 起始系统实例，如果不存在则返回None
        """
        return self._systems.get(name)
    
    def list_systems(self) -> List[str]:
        """
        列出所有已注册的起始系统名称
        
        Returns:
            system_names: 系统名称列表
        """
        return list(self._systems.keys())
    
    def list_system_types(self) -> List[str]:
        """
        列出所有已注册的起始系统类型
        
        Returns:
            type_names: 系统类型名称列表
        """
        return list(self._system_types.keys())
    
    def remove_system(self, name: str) -> bool:
        """
        移除起始系统实例
        
        Args:
            name: 系统名称
            
        Returns:
            success: 是否成功移除
        """
        if name in self._systems:
            del self._systems[name]
            return True
        return False
    
    def create_hat_function_systems(self) -> List[HatFunctionStartingSystem]:
        """
        创建所有10个帽函数起始系统
        
        Returns:
            systems: 帽函数起始系统列表
        """
        systems = []
        for i in range(len(HatFunctionStartingSystem.STARTING_FUNCTIONS_COEFFS)):
            system = HatFunctionStartingSystem(function_index=i)
            self.register_system(system)
            systems.append(system)
        
        return systems
    
    def create_sine_function_systems(self, k_list: List[int] = None, 
                                   domain: tuple = (0, 1)) -> List[SineFunctionStartingSystem]:
        """
        创建多个正弦函数起始系统
        
        Args:
            k_list: 波数列表，如果为None则使用[1, 2, 3]
            domain: 定义域范围
            
        Returns:
            systems: 正弦函数起始系统列表
        """
        if k_list is None:
            k_list = [1, 2, 3]
        
        systems = []
        for k in k_list:
            system = SineFunctionStartingSystem(k=k, domain=domain)
            self.register_system(system)
            systems.append(system)
        
        return systems
    
    def create_gaussian_grid_system(self, 
                                   grid_shape: tuple = (3, 3), 
                                   domain: tuple = ((0, 1), (0, 1)), 
                                   sigma: float = 0.1) -> GaussianFunctionStartingSystem:
        """
        创建网格形式的高斯函数起始系统
        
        Args:
            grid_shape: 网格形状
            domain: 定义域范围
            sigma: 高斯函数标准差
            
        Returns:
            system: 高斯函数起始系统
        """
        system = GaussianFunctionStartingSystem.create_grid_system(
            grid_shape=grid_shape, domain=domain, sigma=sigma
        )
        self.register_system(system)
        return system
    
    def evaluate_system(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """
        计算指定起始系统在给定点的值
        
        Args:
            name: 系统名称
            x: 输入点坐标
            
        Returns:
            u: 起始函数值
        """
        system = self.get_system(name)
        if system is None:
            raise ValueError(f"起始系统 '{name}' 不存在")
        
        return system.evaluate(x)
    
    def get_system_residual(self, name: str, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        计算指定起始系统的残差
        
        Args:
            name: 系统名称
            x: 输入点坐标
            u: 函数值
            
        Returns:
            residual: 起始系统残差
        """
        system = self.get_system(name)
        if system is None:
            raise ValueError(f"起始系统 '{name}' 不存在")
        
        return system.get_residual(x, u)
    
    def get_system_info(self, name: str) -> Dict:
        """
        获取起始系统的详细信息
        
        Args:
            name: 系统名称
            
        Returns:
            info: 系统信息字典
        """
        system = self.get_system(name)
        if system is None:
            raise ValueError(f"起始系统 '{name}' 不存在")
        
        info = system.get_parameters()
        info['description'] = system.get_description()
        info['class_name'] = system.__class__.__name__
        
        return info
    
    def get_all_systems_info(self) -> Dict[str, Dict]:
        """
        获取所有起始系统的信息
        
        Returns:
            systems_info: 系统信息字典
        """
        systems_info = {}
        for name in self.list_systems():
            systems_info[name] = self.get_system_info(name)
        
        return systems_info
    
    def select_best_system(self, 
                         x: torch.Tensor, 
                         target_function: callable,
                         metric: str = 'mse') -> str:
        """
        选择与目标函数最匹配的起始系统
        
        Args:
            x: 输入点坐标
            target_function: 目标函数
            metric: 匹配度量 ('mse', 'mae', 'correlation')
            
        Returns:
            best_system_name: 最佳起始系统名称
        """
        target_values = target_function(x)
        
        best_system = None
        best_score = float('inf') if metric in ['mse', 'mae'] else -float('inf')
        
        for name in self.list_systems():
            system = self.get_system(name)
            system_values = system.evaluate(x)
            
            if metric == 'mse':
                score = torch.mean((system_values - target_values)**2).item()
                if score < best_score:
                    best_score = score
                    best_system = name
            elif metric == 'mae':
                score = torch.mean(torch.abs(system_values - target_values)).item()
                if score < best_score:
                    best_score = score
                    best_system = name
            elif metric == 'correlation':
                # 计算相关系数
                cov = torch.mean((system_values - torch.mean(system_values)) * 
                               (target_values - torch.mean(target_values)))
                std_system = torch.std(system_values)
                std_target = torch.std(target_values)
                
                if std_system > 0 and std_target > 0:
                    correlation = cov / (std_system * std_target)
                    score = correlation.item()
                    if score > best_score:
                        best_score = score
                        best_system = name
        
        return best_system if best_system is not None else ""


# 创建全局管理器实例
_manager = StartingSystemManager()


def get_manager() -> StartingSystemManager:
    """获取全局起始系统管理器"""
    return _manager


def register_system(system: BaseStartingSystem) -> None:
    """注册起始系统到全局管理器"""
    _manager.register_system(system)


def get_system(name: str) -> Optional[BaseStartingSystem]:
    """从全局管理器获取起始系统"""
    return _manager.get_system(name)


def create_system(system_type: str, name: str, **kwargs) -> BaseStartingSystem:
    """通过全局管理器创建起始系统"""
    return _manager.create_system(system_type, name, **kwargs)