"""
MIM-HomPINNs融合方法：变系数四阶常微分方程多解求解

包含MIM和HomPINNs方法的融合实现，支持多种起始系统设计。
"""

# 导入起始系统模块
from .starting_systems import *

__all__ = [
    # 起始系统相关
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