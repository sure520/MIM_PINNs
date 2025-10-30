"""
起始系统模块使用示例

演示如何使用starting_systems模块中的各种起始系统。
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from starting_systems import (
    HatFunctionStartingSystem, 
    SineFunctionStartingSystem, 
    GaussianFunctionStartingSystem,
    get_manager,
    create_system
)


def demo_hat_function_systems():
    """演示帽函数起始系统的使用"""
    print("=== 1D帽函数起始系统演示 ===")
    
    # 创建管理器
    manager = get_manager()
    
    # 创建所有10个帽函数起始系统
    hat_systems = manager.create_hat_function_systems()
    
    print(f"已创建 {len(hat_systems)} 个帽函数起始系统")
    
    # 测试第一个起始系统
    system = hat_systems[0]
    print(f"系统名称: {system.name}")
    print(f"系统描述: {system.get_description()}")
    
    # 生成测试点
    x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
    
    # 计算函数值
    u_values = system.evaluate(x_test)
    
    # 绘制函数图像
    plt.figure(figsize=(10, 6))
    plt.plot(x_test.numpy(), u_values.detach().numpy(), label=system.name)
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('1D帽函数起始系统示例')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return hat_systems


def demo_sine_function_systems():
    """演示正弦函数起始系统的使用"""
    print("\n=== 正弦函数起始系统演示 ===")
    
    # 创建管理器
    manager = get_manager()
    
    # 创建多个正弦函数起始系统
    k_list = [1, 2, 3]
    sine_systems = manager.create_sine_function_systems(k_list=k_list)
    
    print(f"已创建 {len(sine_systems)} 个正弦函数起始系统")
    
    # 生成测试点
    x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
    
    # 绘制所有正弦函数
    plt.figure(figsize=(10, 6))
    for system in sine_systems:
        u_values = system.evaluate(x_test)
        plt.plot(x_test.numpy(), u_values.detach().numpy(), label=system.name)
    
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('正弦函数起始系统示例')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return sine_systems


def demo_gaussian_function_systems():
    """演示高斯函数起始系统的使用"""
    print("\n=== 高斯函数起始系统演示 ===")
    
    # 创建管理器
    manager = get_manager()
    
    # 创建网格形式的高斯函数起始系统
    gaussian_system = manager.create_gaussian_grid_system(
        grid_shape=(3, 3), 
        domain=((0, 1), (0, 1)), 
        sigma=0.1
    )
    
    print(f"系统名称: {gaussian_system.name}")
    print(f"系统描述: {gaussian_system.get_description()}")
    
    # 生成2D测试网格
    x = torch.linspace(0, 1, 50)
    y = torch.linspace(0, 1, 50)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # 展平网格点
    points = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    # 计算函数值
    u_values = gaussian_system.evaluate(points)
    U = u_values.reshape(X.shape)
    
    # 绘制2D等高线图
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X.numpy(), Y.numpy(), U.detach().numpy(), levels=50)
    plt.colorbar(contour)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('2D高斯函数起始系统示例')
    plt.show()
    
    return gaussian_system


def demo_system_manager():
    """演示起始系统管理器的使用"""
    print("\n=== 起始系统管理器演示 ===")
    
    # 获取管理器
    manager = get_manager()
    
    # 列出所有系统类型
    system_types = manager.list_system_types()
    print(f"可用的系统类型: {system_types}")
    
    # 创建不同类型的起始系统
    hat_system = manager.create_system('hat_function', 'test_hat', function_index=0)
    sine_system = manager.create_system('sine_function', 'test_sine', k=1)
    
    # 列出所有已注册的系统
    systems = manager.list_systems()
    print(f"已注册的系统: {systems}")
    
    # 获取系统信息
    for name in systems:
        info = manager.get_system_info(name)
        print(f"系统 {name}: {info}")
    
    # 测试系统选择功能
    def target_function(x):
        """目标函数：sin(2πx)"""
        return torch.sin(2 * np.pi * x[:, 0])
    
    x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
    best_system = manager.select_best_system(x_test, target_function, metric='mse')
    
    print(f"与目标函数最匹配的系统: {best_system}")
    
    return manager


def demo_mim_residuals():
    """演示MIM格式残差计算"""
    print("\n=== MIM格式残差计算演示 ===")
    
    # 创建正弦函数起始系统
    system = SineFunctionStartingSystem(k=1)
    
    # 生成测试点
    x_test = torch.linspace(0, 1, 10).reshape(-1, 1)
    x_test.requires_grad = True
    
    # 计算函数值
    u_values = system.evaluate(x_test)
    
    # 计算MIM格式残差
    residuals = system.get_mim_residuals(x_test, u_values)
    
    print(f"MIM残差数量: {len(residuals)}")
    for i, residual in enumerate(residuals):
        print(f"残差 {i+1} 的均值: {residual.mean().item():.6f}")
    
    return residuals


def main():
    """主演示函数"""
    print("起始系统模块使用演示")
    print("=" * 50)
    
    try:
        # 演示各种起始系统
        hat_systems = demo_hat_function_systems()
        sine_systems = demo_sine_function_systems()
        gaussian_system = demo_gaussian_function_systems()
        
        # 演示管理器功能
        manager = demo_system_manager()
        
        # 演示MIM残差计算
        residuals = demo_mim_residuals()
        
        print("\n演示完成！")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()