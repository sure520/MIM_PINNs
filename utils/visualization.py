import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import seaborn as sns
from matplotlib import rcParams

# 设置英文字体
plt.rcParams['font.sans-serif'] = ['Arial']  # 使用英文字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class Visualizer:
    """
    可视化类，负责结果的可视化
    """
    def __init__(self, save_dir='results'):
        """
        初始化可视化器
        Args:
            save_dir: 保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置绘图风格
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_solutions(self, models, omega2_list, ks, T=600, v=50, save_name='solutions.png', figsize=(15, 10)):
        """
        绘制多解图像
        Args:
            models: 模型列表
            omega2_list: 特征值列表
            ks: k值列表
            T: 方程参数T
            v: 方程参数v
            save_name: 保存文件名
            figsize: 图像大小
        """
        # 生成测试点（与模型在同一设备上）
        device = next(models[0].parameters()).device
        x_test = torch.linspace(0, 1, 1000, device=device).unsqueeze(1)
        
        # 收集所有数据以确定合适的纵轴范围
        all_y1 = []
        all_y2 = []
        all_y3 = []
        all_y4 = []
        
        for model in models:
            y1, y2, y3, y4, _ = model(x_test)
            all_y1.append(y1.detach().cpu().numpy().flatten())
            all_y2.append(y2.detach().cpu().numpy().flatten())
            all_y3.append(y3.detach().cpu().numpy().flatten())
            all_y4.append(y4.detach().cpu().numpy().flatten())
        
        # 计算每个变量的全局最小最大值
        y1_min = min(np.min(arr) for arr in all_y1)
        y1_max = max(np.max(arr) for arr in all_y1)
        y2_min = min(np.min(arr) for arr in all_y2)
        y2_max = max(np.max(arr) for arr in all_y2)
        y3_min = min(np.min(arr) for arr in all_y3)
        y3_max = max(np.max(arr) for arr in all_y3)
        y4_min = min(np.min(arr) for arr in all_y4)
        y4_max = max(np.max(arr) for arr in all_y4)
        
        # 添加10%的边距
        y1_range = y1_max - y1_min
        y1_margin = 0.1 * y1_range if y1_range > 0 else 0.1
        
        y2_range = y2_max - y2_min
        y2_margin = 0.1 * y2_range if y2_range > 0 else 0.1
        
        y3_range = y3_max - y3_min
        y3_margin = 0.1 * y3_range if y3_range > 0 else 0.1
        
        y4_range = y4_max - y4_min
        y4_margin = 0.1 * y4_range if y4_range > 0 else 0.1
        
        # 创建图像
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Multiple Solutions of Variable Coefficient 4th-Order ODE (T={T}, v={v})', fontsize=16)
        
        # 绘制解y(x)
        ax = axes[0, 0]
        for i, (model, omega2, k) in enumerate(zip(models, omega2_list, ks)):
            y1, y2, y3, y4, _ = model(x_test)
            y1_np = y1.detach().cpu().numpy().flatten()
            x_np = x_test.detach().cpu().numpy().flatten()
            ax.plot(x_np, y1_np, label=f'Solution {i+1}: k={k}, ω²={omega2:.4f}')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y(x)')
        ax.set_title('Solution y(x)')
        ax.set_ylim(y1_min - y1_margin, y1_max + y1_margin)
        ax.legend()
        ax.grid(True)
        
        # 绘制一阶导数y'(x)
        ax = axes[0, 1]
        for i, (model, omega2, k) in enumerate(zip(models, omega2_list, ks)):
            y1, y2, y3, y4, _ = model(x_test)
            y2_np = y2.detach().cpu().numpy().flatten()
            x_np = x_test.detach().cpu().numpy().flatten()
            ax.plot(x_np, y2_np, label=f'Solution {i+1}: k={k}')
        
        ax.set_xlabel('x')
        ax.set_ylabel("y'(x)")
        ax.set_title("First Derivative y'(x)")
        ax.set_ylim(y2_min - y2_margin, y2_max + y2_margin)
        ax.legend()
        ax.grid(True)
        
        # 绘制二阶导数y''(x)
        ax = axes[1, 0]
        for i, (model, omega2, k) in enumerate(zip(models, omega2_list, ks)):
            y1, y2, y3, y4, _ = model(x_test)
            y3_np = y3.detach().cpu().numpy().flatten()
            x_np = x_test.detach().cpu().numpy().flatten()
            ax.plot(x_np, y3_np, label=f'Solution {i+1}: k={k}')
        
        ax.set_xlabel('x')
        ax.set_ylabel("y''(x)")
        ax.set_title("Second Derivative y''(x)")
        ax.set_ylim(y3_min - y3_margin, y3_max + y3_margin)
        ax.legend()
        ax.grid(True)
        
        # 绘制三阶导数y'''(x)
        ax = axes[1, 1]
        for i, (model, omega2, k) in enumerate(zip(models, omega2_list, ks)):
            y1, y2, y3, y4, _ = model(x_test)
            y4_np = y4.detach().cpu().numpy().flatten()
            x_np = x_test.detach().cpu().numpy().flatten()
            ax.plot(x_np, y4_np, label=f'Solution {i+1}: k={k}')
        
        ax.set_xlabel('x')
        ax.set_ylabel("y'''(x)")
        ax.set_title("Third Derivative y'''(x)")
        ax.set_ylim(y4_min - y4_margin, y4_max + y4_margin)
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300)
        plt.close()
    
    def plot_solution(self, model, x_test, title="Solution", filename="solution.png", figsize=(10, 6)):
        """
        绘制单个解图像
        Args:
            model: 模型
            x_test: 测试点
            title: 图像标题
            filename: 保存文件名
            figsize: 图像大小
        """
        # 计算解
        y1, y2, y3, y4, _ = model(x_test)
        
        # 转换为numpy
        x_np = x_test.detach().cpu().numpy().flatten()
        y1_np = y1.detach().cpu().numpy().flatten()
        y2_np = y2.detach().cpu().numpy().flatten()
        y3_np = y3.detach().cpu().numpy().flatten()
        y4_np = y4.detach().cpu().numpy().flatten()
        
        # 创建图像
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16)
        
        # 绘制解y(x)
        ax = axes[0, 0]
        ax.plot(x_np, y1_np, 'b-')
        ax.set_xlabel('x')
        ax.set_ylabel('y(x)')
        ax.set_title('Solution y(x)')
        ax.grid(True)
        
        # 绘制一阶导数y'(x)
        ax = axes[0, 1]
        ax.plot(x_np, y2_np, 'r-')
        ax.set_xlabel('x')
        ax.set_ylabel("y'(x)")
        ax.set_title("First Derivative y'(x)")
        ax.grid(True)
        
        # 绘制二阶导数y''(x)
        ax = axes[1, 0]
        ax.plot(x_np, y3_np, 'g-')
        ax.set_xlabel('x')
        ax.set_ylabel("y''(x)")
        ax.set_title("Second Derivative y''(x)")
        ax.grid(True)
        
        # 绘制三阶导数y'''(x)
        ax = axes[1, 1]
        ax.plot(x_np, y4_np, 'm-')
        ax.set_xlabel('x')
        ax.set_ylabel("y'''(x)")
        ax.set_title("Third Derivative y'''(x)")
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
        plt.close()
    
    def plot_loss_history(self, loss_history, title="Loss History", filename="loss_history.png", figsize=(12, 8)):
        """
        绘制单个解的损失历史
        Args:
            loss_history: 损失历史字典
            title: 图像标题
            filename: 保存文件名
            figsize: 图像大小
        """
        # 创建图像
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16)
        
        # 绘制总损失
        ax = axes[0, 0]
        epochs = np.arange(len(loss_history['total_loss'])) * 100
        ax.plot(epochs, loss_history['total_loss'])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title('Total Loss')
        ax.grid(True)
        
        # 绘制F损失
        ax = axes[0, 1]
        ax.plot(epochs, loss_history['F_loss'])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F Loss')
        ax.set_title('F Loss')
        ax.grid(True)
        
        # 绘制G损失
        ax = axes[1, 0]
        ax.plot(epochs, loss_history['G_loss'])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('G Loss')
        ax.set_title('G Loss')
        ax.grid(True)
        
        # 绘制特征值变化
        ax = axes[1, 1]
        ax.plot(epochs, loss_history['omega2'])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('ω²')
        ax.set_title('Eigenvalue Variation')
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
        plt.close()
    
    def plot_eigenvalue_distribution(self, omega2_values, title="Eigenvalue Distribution", filename="eigenvalue_distribution.png", figsize=(8, 6)):
        """
        绘制特征值分布图
        Args:
            omega2_values: 特征值列表
            title: 图像标题
            filename: 保存文件名
            figsize: 图像大小
        """
        # 创建图像
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制特征值分布
        indices = range(len(omega2_values))
        bars = ax.bar(indices, omega2_values)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{omega2_values[i]:.4f}',
                    ha='center', va='bottom')
        
        # 设置标题和标签
        ax.set_xlabel('Solution Index')
        ax.set_ylabel('Eigenvalue ω²')
        ax.set_title(title)
        ax.set_xticks(indices)
        ax.set_xticklabels([f'Solution {i+1}' for i in indices])
        ax.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=300)
        plt.close()
    
    def plot_loss_curves(self, loss_histories, ks, save_name='loss_curves.png', figsize=(15, 10)):
        """
        绘制损失曲线
        Args:
            loss_histories: 损失历史列表
            ks: k值列表
            save_name: 保存文件名
            figsize: 图像大小
        """
        # 创建图像
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Training Loss Curves', fontsize=16)
        
        # 计算各损失的全局范围
        total_losses = []
        F_losses = []
        G_losses = []
        R_b_losses = []
        omega2s = []
        
        for loss_history in loss_histories:
            total_losses.extend(loss_history['total_loss'])
            F_losses.extend(loss_history['F_loss'])
            G_losses.extend(loss_history['G_loss'])
            R_b_losses.extend(loss_history['R_b_loss'])
            omega2s.extend(loss_history['omega2'])
        
        # 定义计算合适范围的函数
        def get_loss_limits(loss_values):
            min_val = np.min(loss_values)
            max_val = np.max(loss_values)
            # 对于可能包含0的损失值，使用对数思维确定范围
            if max_val < 1e-10:
                return (-1e-10, 1e-10)
            # 对于损失值，使用对数空间的10%边距
            log_min = np.log10(max(min_val, 1e-20))
            log_max = np.log10(max_val)
            log_margin = 0.1 * (log_max - log_min)
            return (10**(log_min - log_margin), 10**(log_max + log_margin))
        
        # 对于特征值，使用线性空间
        def get_linear_limits(values):
            min_val = np.min(values)
            max_val = np.max(values)
            range_val = max_val - min_val
            if range_val < 1e-10:
                return (min_val - 1e-10, max_val + 1e-10)
            margin = 0.1 * range_val
            return (min_val - margin, max_val + margin)
        
        # 计算各损失的范围
        total_loss_limits = get_loss_limits(total_losses)
        F_loss_limits = get_loss_limits(F_losses)
        G_loss_limits = get_loss_limits(G_losses)
        R_b_loss_limits = get_loss_limits(R_b_losses)
        omega2_limits = get_linear_limits(omega2s)
        
        # 绘制总损失
        ax = axes[0, 0]
        for i, (loss_history, k) in enumerate(zip(loss_histories, ks)):
            epochs = np.arange(len(loss_history['total_loss'])) * 100
            ax.plot(epochs, loss_history['total_loss'], label=f'k={k}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title('Total Loss')
        # 如果损失范围很大，使用对数坐标
        if max(total_loss_limits) / min(total_loss_limits) > 100:
            ax.set_yscale('log')
        ax.legend()
        ax.grid(True)
        
        # 绘制F损失
        ax = axes[0, 1]
        for i, (loss_history, k) in enumerate(zip(loss_histories, ks)):
            epochs = np.arange(len(loss_history['F_loss'])) * 100
            ax.plot(epochs, loss_history['F_loss'], label=f'k={k}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F Loss')
        ax.set_title('F Loss')
        # 如果损失范围很大，使用对数坐标
        if max(F_loss_limits) / min(F_loss_limits) > 100:
            ax.set_yscale('log')
        ax.legend()
        ax.grid(True)
        
        # 绘制G损失
        ax = axes[0, 2]
        for i, (loss_history, k) in enumerate(zip(loss_histories, ks)):
            epochs = np.arange(len(loss_history['G_loss'])) * 100
            ax.plot(epochs, loss_history['G_loss'], label=f'k={k}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('G Loss')
        ax.set_title('G Loss')
        # 如果损失范围很大，使用对数坐标
        if max(G_loss_limits) / min(G_loss_limits) > 100:
            ax.set_yscale('log')
        ax.legend()
        ax.grid(True)
        
        # 绘制边界损失
        ax = axes[1, 0]
        for i, (loss_history, k) in enumerate(zip(loss_histories, ks)):
            epochs = np.arange(len(loss_history['R_b_loss'])) * 100
            ax.plot(epochs, loss_history['R_b_loss'], label=f'k={k}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Boundary Loss')
        ax.set_title('Boundary Loss')
        # 边界损失通常很小，使用对数坐标
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True)
        
        # 绘制特征值变化
        ax = axes[1, 1]
        for i, (loss_history, k) in enumerate(zip(loss_histories, ks)):
            epochs = np.arange(len(loss_history['omega2'])) * 100
            ax.plot(epochs, loss_history['omega2'], label=f'k={k}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('ω²')
        ax.set_title('Eigenvalue Variation')
        ax.set_ylim(omega2_limits)
        ax.legend()
        ax.grid(True)
        
        # 绘制损失对比（对数坐标）
        ax = axes[1, 2]
        for i, (loss_history, k) in enumerate(zip(loss_histories, ks)):
            epochs = np.arange(len(loss_history['total_loss'])) * 100
            ax.semilogy(epochs, loss_history['total_loss'], label=f'k={k}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss (log scale)')
        ax.set_title('Total Loss (log scale)')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300)
        plt.close()
    
    def plot_residuals(self, model, omega2, T=600, v=50, save_name='residuals.png', figsize=(15, 5)):
        """
        绘制残差分布
        Args:
            model: 模型
            omega2: 特征值
            T: 方程参数T
            v: 方程参数v
            save_name: 保存文件名
            figsize: 图像大小
        """
        # 生成测试点（与模型在同一设备上）
        device = next(model.parameters()).device
        x_test = torch.linspace(0, 1, 1000, device=device).unsqueeze(1)
        
        # 计算残差
        R1, R2, R3, R4, y1, y2, y3, y4, _ = model.compute_residuals(x_test, T, v, omega2)
        
        # 转换为numpy
        x_np = x_test.detach().cpu().numpy().flatten()
        R1_np = R1.detach().cpu().numpy().flatten()
        R2_np = R2.detach().cpu().numpy().flatten()
        R3_np = R3.detach().cpu().numpy().flatten()
        R4_np = R4.detach().cpu().numpy().flatten()
        
        # 计算每个残差的范围，并添加边距
        def calculate_limits(residual):
            min_val = np.min(residual)
            max_val = np.max(residual)
            range_val = max_val - min_val
            # 如果范围太小，使用一个合理的默认范围
            if range_val < 1e-10:
                abs_max = max(abs(min_val), abs(max_val))
                if abs_max < 1e-10:
                    return (-1e-10, 1e-10)  # 防止所有值都是0的情况
                return (-abs_max * 1.1, abs_max * 1.1)
            margin = 0.1 * range_val
            return (min_val - margin, max_val + margin)
        
        R1_limits = calculate_limits(R1_np)
        R2_limits = calculate_limits(R2_np)
        R3_limits = calculate_limits(R3_np)
        R4_limits = calculate_limits(R4_np)
        
        # 创建图像
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        fig.suptitle(f'Residual Distribution (ω²={omega2:.4f})', fontsize=16)
        
        # 绘制R1残差
        ax = axes[0]
        ax.plot(x_np, R1_np, 'r-', label='R1 Residual')
        ax.set_xlabel('x')
        ax.set_ylabel('Residual')
        ax.set_title('R1 Residual')
        ax.set_ylim(R1_limits)
        ax.legend()
        ax.grid(True)
        
        # 绘制R2残差
        ax = axes[1]
        ax.plot(x_np, R2_np, 'b-', label='R2 Residual')
        ax.set_xlabel('x')
        ax.set_ylabel('Residual')
        ax.set_title('R2 Residual')
        ax.set_ylim(R2_limits)
        ax.legend()
        ax.grid(True)
        
        # 绘制R3残差
        ax = axes[2]
        ax.plot(x_np, R3_np, 'g-', label='R3 Residual')
        ax.set_xlabel('x')
        ax.set_ylabel('Residual')
        ax.set_title('R3 Residual')
        ax.set_ylim(R3_limits)
        ax.legend()
        ax.grid(True)
        
        # 绘制R4残差
        ax = axes[3]
        ax.plot(x_np, R4_np, 'm-', label='R4 Residual')
        ax.set_xlabel('x')
        ax.set_ylabel('Residual')
        ax.set_title('R4 Residual')
        ax.set_ylim(R4_limits)
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300)
        plt.close()
    
    def plot_comparison_with_analytical(self, model, omega2, k, T=600, v=50, 
                                       save_name='comparison.png', figsize=(12, 8)):
        """
        与解析解比较
        Args:
            model: 模型
            omega2: 特征值
            k: k值
            T: 方程参数T
            v: 方程参数v
            save_name: 保存文件名
            figsize: 图像大小
        """
        # 生成测试点（与模型在同一设备上）
        device = next(model.parameters()).device
        x_test = torch.linspace(0, 1, 1000, device=device).unsqueeze(1)
        
        # 计算数值解
        y1, y2, y3, y4, _ = model(x_test)
        
        # 计算解析解（假设为sin(kπx)）
        x_np = x_test.detach().cpu().numpy().flatten()
        y1_analytical = np.sin(k * np.pi * x_np)
        y2_analytical = k * np.pi * np.cos(k * np.pi * x_np)
        y3_analytical = -(k * np.pi)**2 * np.sin(k * np.pi * x_np)
        y4_analytical = -(k * np.pi)**3 * np.cos(k * np.pi * x_np)
        
        # 转换为numpy
        y1_np = y1.detach().cpu().numpy().flatten()
        y2_np = y2.detach().cpu().numpy().flatten()
        y3_np = y3.detach().cpu().numpy().flatten()
        y4_np = y4.detach().cpu().numpy().flatten()
        
        # 创建图像
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Numerical vs Analytical Solution Comparison (k={k}, ω²={omega2:.4f})', fontsize=16)
        
        # 绘制解y(x)
        ax = axes[0, 0]
        ax.plot(x_np, y1_np, 'b-', label='Numerical Solution')
        ax.plot(x_np, y1_analytical, 'r--', label='Analytical Solution')
        ax.set_xlabel('x')
        ax.set_ylabel('y(x)')
        ax.set_title('Solution y(x)')
        ax.legend()
        ax.grid(True)
        
        # 绘制一阶导数y'(x)
        ax = axes[0, 1]
        ax.plot(x_np, y2_np, 'b-', label='Numerical Solution')
        ax.plot(x_np, y2_analytical, 'r--', label='Analytical Solution')
        ax.set_xlabel('x')
        ax.set_ylabel("y'(x)")
        ax.set_title("First Derivative y'(x)")
        ax.legend()
        ax.grid(True)
        
        # 绘制二阶导数y''(x)
        ax = axes[1, 0]
        ax.plot(x_np, y3_np, 'b-', label='Numerical Solution')
        ax.plot(x_np, y3_analytical, 'r--', label='Analytical Solution')
        ax.set_xlabel('x')
        ax.set_ylabel("y''(x)")
        ax.set_title("Second Derivative y''(x)")
        ax.legend()
        ax.grid(True)
        
        # 绘制三阶导数y'''(x)
        ax = axes[1, 1]
        ax.plot(x_np, y4_np, 'b-', label='Numerical Solution')
        ax.plot(x_np, y4_analytical, 'r--', label='Analytical Solution')
        ax.set_xlabel('x')
        ax.set_ylabel("y'''(x)")
        ax.set_title("Third Derivative y'''(x)")
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300)
        plt.close()
    
    def plot_omega2_convergence(self, omega2_list, ks, save_name='omega2_convergence.png', figsize=(10, 6)):
        """
        绘制特征值收敛情况
        Args:
            omega2_list: 特征值列表
            ks: k值列表
            save_name: 保存文件名
            figsize: 图像大小
        """
        # 创建图像
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制特征值
        bars = ax.bar(range(len(omega2_list)), omega2_list)
        
        # 设置x轴标签
        ax.set_xticks(range(len(omega2_list)))
        ax.set_xticklabels([f'k={k}' for k in ks])
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{omega2_list[i]:.4f}',
                    ha='center', va='bottom')
        
        # 设置标题和标签
        ax.set_xlabel('k值')
        ax.set_ylabel('特征值ω²')
        ax.set_title('特征值收敛情况')
        ax.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300)
        plt.close()