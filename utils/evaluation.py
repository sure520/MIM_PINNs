import torch
import numpy as np


class Evaluator:
    """
    评估器类，负责评估模型的性能
    """
    def __init__(self, model, data_gen, device='cuda'):
        """
        初始化评估器
        Args:
            model: 模型
            data_gen: 数据生成器
            device: 计算设备
        """
        self.model = model
        self.data_gen = data_gen
        self.device = device
    
    def evaluate(self, model, omega2, T=600, v=50):
        """
        评估模型的性能
        Args:
            model: 模型
            omega2: 特征值
            T: 方程参数T
            v: 方程参数v
        Returns:
            residual_error: 残差误差
            boundary_error: 边界误差
        """
        # 生成测试数据
        x_test = self.data_gen.generate_test_points(N_test=1000)
        x_b_test = self.data_gen.generate_boundary_points()
        
        # 确保测试点需要梯度
        x_test.requires_grad_(True)
        x_b_test.requires_grad_(True)
        
        # 计算残差误差
        residual_error = self._compute_residual_error(model, x_test, omega2, T, v)
        
        # 计算边界误差
        boundary_error = self._compute_boundary_error(model, x_b_test)
        
        return residual_error, boundary_error
    
    def _compute_residual_error(self, model, x, omega2, T, v):
        """
        计算残差误差
        Args:
            model: 模型
            x: 输入点
            omega2: 特征值
            T: 方程参数T
            v: 方程参数v
        Returns:
            residual_error: 残差误差
        """
        # 计算残差
        R1, R2, R3, R4, y1, y2, y3, y4, _ = model.compute_residuals(x, T, v, omega2)
        
        # 计算L2误差
        residual_error = torch.sqrt((R1**2 + R2**2 + R3**2 + R4**2).mean()).item()
        
        return residual_error
    
    def _compute_boundary_error(self, model, x_b):
        """
        计算边界误差
        Args:
            model: 模型
            x_b: 边界点
        Returns:
            boundary_error: 边界误差
        """
        # 计算边界残差
        R_b, y1_b, y3_b = model.compute_boundary_residuals(x_b)
        
        # 计算L2误差
        boundary_error = torch.sqrt(R_b.mean()).item()
        
        return boundary_error
    
    def compute_solution_norm(self, model):
        """
        计算解的范数
        Args:
            model: 模型
        Returns:
            norm: 解的L2范数
        """
        x_test = self.data_gen.generate_test_points()
        y1, _, _, _, _ = model(x_test)
        norm = torch.sqrt(torch.mean(y1**2)).item()
        return norm
    
    def compute_derivative_norms(self, model):
        """
        计算各阶导数的范数
        Args:
            model: 模型
        Returns:
            norms: 各阶导数的L2范数列表
        """
        x_test = self.data_gen.generate_test_points()
        y1, y2, y3, y4, _ = model(x_test)
        
        # 计算各阶导数的范数
        norms = [
            torch.sqrt(torch.mean(y1**2)).item(),  # y的范数
            torch.sqrt(torch.mean(y2**2)).item(),  # y'的范数
            torch.sqrt(torch.mean(y3**2)).item(),  # y''的范数
            torch.sqrt(torch.mean(y4**2)).item()   # y'''的范数
        ]
        
        return norms
    
    def compare_solutions(self, model1, model2):
        """
        比较两个解的差异
        Args:
            model1: 第一个模型
            model2: 第二个模型
        Returns:
            solution_diff: 解的差异
        """
        # 生成测试点
        x_test = self.data_gen.generate_test_points(N_test=1000)
        
        # 计算解的差异
        y1_1, _, _, _, _ = model1(x_test)
        y1_2, _, _, _, _ = model2(x_test)
        
        solution_diff = torch.sqrt(torch.mean((y1_1 - y1_2)**2)).item()
        
        return solution_diff