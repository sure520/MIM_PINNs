"""
起始系统模块

该模块包含用于HomPINNs方法的起始系统实现，包括：
1. 1D帽函数起始系统
2. 正弦函数起始系统  
3. 高斯函数起始系统
4. 起始系统管理器

基于MIM和HomPINNs论文的方法实现。
"""

from .base_system import BaseStartingSystem
from .hat_function_system import HatFunctionStartingSystem
from .sine_function_system import SineFunctionStartingSystem
from .gaussian_function_system import GaussianFunctionStartingSystem
from .system_manager import StartingSystemManager, get_manager, register_system, get_system, create_system

__all__ = [
    'BaseStartingSystem',
    'HatFunctionStartingSystem', 
    'SineFunctionStartingSystem',
    'GaussianFunctionStartingSystem',
    'StartingSystemManager',
    'get_manager',
    'register_system', 
    'get_system',
    'create_system'
]