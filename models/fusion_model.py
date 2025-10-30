import torch
import torch.nn as nn
import numpy as np


class ResidualBlock(nn.Module):
    """
    ResNet残差块，用于构建MIM网络
    """
    def __init__(self, width):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(width, width)
        self.linear2 = nn.Linear(width, width)
        self.activation = nn.Tanh()

    def forward(self, x):
        residual = x
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        out += residual  # 残差连接
        out = self.activation(out)
        return out


class MIM1(nn.Module):
    """
    MIM¹网络架构：单个网络同时输出所有变量
    """
    def __init__(self, width=30, depth=2):
        super(MIM1, self).__init__()
        self.width = width
        self.depth = depth
        
        # 输入层，将1维输入映射到width维
        self.input_layer = nn.Linear(1, width)
        
        # 残差块
        self.residual_blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(depth)])
        
        # 输出层，将width维映射到5维 (y1, y2, y3, y4, omega2)
        self.output_layer = nn.Linear(width, 5)
        
        self.activation = nn.Tanh()

    def forward(self, x):
        # 输入处理
        out = self.activation(self.input_layer(x))
        
        # 通过残差块
        out = self.residual_blocks(out)
        
        # 输出处理
        out = self.output_layer(out)
        
        # 分离各个变量
        y1 = out[:, 0]  # 原函数y
        y2 = out[:, 1]  # 一阶导数y'
        y3 = out[:, 2]  # 二阶导数y''
        y4 = out[:, 3]  # 三阶导数y''''
        omega2 = out[:, 4]  # 特征值ω²
        
        # 强制omega2为标量（取第一个元素，确保全局常数）
        omega2 = omega2[0]
        omega2 = omega2.repeat(x.shape[0])  # 广播到与x同长度
        
        # 边界条件处理：确保y1(0)=y1(1)=0和y3(0)=y3(1)=0
        # 通过乘以x(1-x)因子自动满足边界条件
        boundary_factor = x[:, 0] * (1 - x[:, 0])
        y1 = y1 * boundary_factor
        y3 = y3 * boundary_factor
            
        return y1, y2, y3, y4, omega2


class MIM2(nn.Module):
    """
    MIM²网络架构：多个网络分别输出不同变量
    """
    def __init__(self, width=30, depth=2):
        super(MIM2, self).__init__()
        self.width = width
        self.depth = depth
        
        # DNN1 用于逼近y1和y3
        self.dnn1_input = nn.Linear(1, width)
        self.dnn1_blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(depth)])
        self.dnn1_output = nn.Linear(width, 2)  # 输出y1和y3
        
        # DNN2 用于逼近y2和y4
        self.dnn2_input = nn.Linear(1, width)
        self.dnn2_blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(depth)])
        self.dnn2_output = nn.Linear(width, 2)  # 输出y2和y4
        
        # DNN3 用于输出全局常数omega2（不依赖x输入）
        self.dnn3 = nn.Sequential(
            nn.Linear(1, width),  # 输入为常数1（占位）
            nn.Tanh(),
            *[ResidualBlock(width) for _ in range(depth)],
            nn.Linear(width, 1)
        )
        
        self.activation = nn.Tanh()

    def forward(self, x):
        # DNN1 forward - 输出y1和y3
        out1 = self.activation(self.dnn1_input(x))
        out1 = self.dnn1_blocks(out1)
        y1_y3 = self.dnn1_output(out1)
        y1 = y1_y3[:, 0]
        y3 = y1_y3[:, 1]
        
        # DNN2 forward - 输出y2和y4
        out2 = self.activation(self.dnn2_input(x))
        out2 = self.dnn2_blocks(out2)
        y2_y4 = self.dnn2_output(out2)
        y2 = y2_y4[:, 0]
        y4 = y2_y4[:, 1]
        
        # DNN3 forward - 输出全局常数omega2
        # 输入常数1，确保omega2与x无关
        omega2 = self.dnn3(torch.ones_like(x)).squeeze()[0]
        omega2 = omega2.repeat(x.shape[0])  # 广播到批量长度
        
        # 边界条件处理：确保y1(0)=y1(1)=0和y3(0)=y3(1)=0
        # 通过乘以x(1-x)因子自动满足边界条件
        boundary_factor = x[:, 0] * (1 - x[:, 0])
        y1 = y1 * boundary_factor
        y3 = y3 * boundary_factor
            
        return y1, y2, y3, y4, omega2


class MIMHomPINNFusion(nn.Module):
    """
    MIM与HomPINNs融合模型
    结合MIM的一阶系统转化方法和HomPINNs的同伦多解探索方法
    """
    def __init__(self, width=30, depth=2, model_type='MIM1', device='cpu'):
        super(MIMHomPINNFusion, self).__init__()
        self.model_type = model_type
        self.device = device
        
        # 确保device是字符串而不是字典
        if isinstance(device, dict) and 'device' in device:
            device = device['device']
        
        # 根据模型类型创建网络
        if model_type == 'MIM1':
            self.model = MIM1(width, depth)
        elif model_type == 'MIM2':
            self.model = MIM2(width, depth)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
            
        # 将模型移动到指定设备
        self.model = self.model.to(device)
    
    def check_device_consistency(self, *tensors, device=None):
        """
        检查所有张量是否在同一设备上
        Args:
            *tensors: 要检查的张量
            device: 目标设备（如果为None，则使用模型设备）
        Returns:
            bool: 所有张量是否在同一设备上
        """
        if device is None:
            device = self.device
        
        for i, tensor in enumerate(tensors):
            if tensor.device != device:
                print(f"警告: 张量 {i} 设备不一致: {tensor.device} vs {device}")
                return False
        
        return True
    
    def forward(self, x):
        """前向传播"""
        return self.model(x)
    
    def compute_residuals(self, x, T=600, v=50, omega2=None):
        """
        计算一阶系统的残差
        Args:
            x: 输入点
            T: 方程参数T
            v: 方程参数v
            omega2: 特征值（如果为None，则从模型输出获取）
        Returns:
            R1, R2, R3, R4: 各个残差项
            y1, y2, y3, y4, omega2_val: 各个变量值
        """
        # 前向传播
        y1, y2, y3, y4, omega2_val = self.forward(x)
        
        # 如果提供了omega2，则使用提供的值
        if omega2 is not None:
            omega2_val = omega2_val * 0 + omega2  # 保持形状但使用提供的值
        
        # 计算各阶导数
        y1_x = torch.autograd.grad(
            outputs=y1, 
            inputs=x, 
            grad_outputs=torch.ones_like(y1), 
            create_graph=True, 
            retain_graph=True
        )[0]
        
        y2_x = torch.autograd.grad(
            outputs=y2, 
            inputs=x, 
            grad_outputs=torch.ones_like(y2), 
            create_graph=True, 
            retain_graph=True
        )[0]
        
        y3_x = torch.autograd.grad(
            outputs=y3, 
            inputs=x, 
            grad_outputs=torch.ones_like(y3), 
            create_graph=True, 
            retain_graph=True
        )[0]
        
        y4_x = torch.autograd.grad(
            outputs=y4, 
            inputs=x, 
            grad_outputs=torch.ones_like(y4), 
            create_graph=True, 
            retain_graph=True
        )[0]
        
        # 确保所有张量形状一致（一维）
        y1_x = y1_x.squeeze()
        y2_x = y2_x.squeeze()
        y3_x = y3_x.squeeze()
        y4_x = y4_x.squeeze()
        
        # 计算残差项
        R1 = y2 - y1_x  # y2 - y1' = 0
        R2 = y3 - y2_x  # y3 - y2' = 0
        R3 = y4 - y3_x  # y4 - y3' = 0
        
        # R4残差量级归一化：降低变系数项权重，避免网络"避重就轻"
        beta = 1e-4  # 归一化系数
        R4 = beta * (y4_x - ((T + v*x) * y2_x + v * y1_x + omega2_val * y1))  # y4' - ((T+vx)y'' + vy' + ω²y) = 0
        
        return R1, R2, R3, R4, y1, y2, y3, y4, omega2_val, y1_x, y2_x, y3_x, y4_x
    
    def compute_boundary_residuals(self, x_b):
        """
        计算边界条件残差
        Args:
            x_b: 边界点
        Returns:
            R_b: 边界残差
            y1_b, y3_b: 边界处的y1和y3值
        """
        # 前向传播
        y1, y2, y3, y4, omega2 = self.forward(x_b)
        
        # 边界条件残差: y1(0)=0, y1(1)=0, y3(0)=0, y3(1)=0
        R_b = y1**2 + y3**2
        
        # 计算边界处的一阶导数（检查是否合理，非必须但可增强约束）
        y1_x_b = torch.autograd.grad(y1, x_b, grad_outputs=torch.ones_like(y1), create_graph=True)[0].squeeze()
        # 轻微惩罚异常导数
        R_b = R_b + (y1_x_b**2).mean() * 0.1
        
        return R_b, y1, y3
    
    def compute_homotopy_loss(self, x, x_b, t, T=600, v=50, omega2=None, alpha=10, k=1, beta=1e-4, epsilon=1e-6):
        """
        计算同伦损失函数（包含非零解惩罚项）
        
        完整损失函数：L_total(t) = t·F + (1-t)·G + α(t)·R_b + β·L_nonzero
        
        Args:
            x: 内部点
            x_b: 边界点
            t: 同伦参数
            T: 方程参数T
            v: 方程参数v
            omega2: 特征值
            alpha: 边界惩罚系数
            k: 起始函数的k值
            beta: 非零解惩罚权重（默认1e-4）
            epsilon: 数值稳定项（默认1e-6）
        Returns:
            loss: 总损失
            F: 目标系统损失
            G: 起始系统损失
            R_b: 边界损失
            L_nonzero: 非零解惩罚项
        """
        # 确保所有输入张量在同一设备上
        x = x.to(self.device)
        x_b = x_b.to(self.device)
        
        # 计算目标系统损失F，同时获取各阶导数
        R1, R2, R3, R4, y1, y2, y3, y4, omega2_val, y1_x, y2_x, y3_x, y4_x = self.compute_residuals(x, T, v, omega2)
        F = (R1**2 + R2**2 + R3**2 + R4**2).mean()
        
        # 计算起始系统损失G（基于HomPINNs论文正确方法）
        # 起始系统: y^(4) - ω₀²y = 0, 解析解为 y_s(x) = sin(kπx)
        omega0_2 = (k * torch.pi)**4
        
        # 计算起始函数真实值及其导数（确保在同一设备上）
        y_s = torch.sin(k * torch.pi * x)  # 起始函数: y_s(x) = sin(kπx)
        y_s_x = k * torch.pi * torch.cos(k * torch.pi * x)  # 一阶导数: y_s'(x) = kπ cos(kπx)
        y_s_xx = -(k * torch.pi)**2 * torch.sin(k * torch.pi * x)  # 二阶导数: y_s''(x) = -(kπ)^2 sin(kπx)
        y_s_xxx = -(k * torch.pi)**3 * torch.cos(k * torch.pi * x)  # 三阶导数: y_s'''(x) = -(kπ)^3 cos(kπx)
        y_s_xxxx = (k * torch.pi)**4 * torch.sin(k * torch.pi * x)  # 四阶导数: y_s''''(x) = (kπ)^4 sin(kπx)
        
        # 起始系统损失G：神经网络输出与起始函数真实值及其导数的差异
        # 根据HomPINNs论文，起始系统损失应该是神经网络输出与起始函数的差异
        # 这里我们使用函数值及其导数的MSE损失
        G_y = ((y1 - y_s)**2).mean()  # 函数值差异
        G_yx = ((y1_x - y_s_x)**2).mean()  # 一阶导数差异
        G_yxx = ((y2_x - y_s_xx)**2).mean()  # 二阶导数差异（注意：y2应该是y1'）
        
        # 起始系统损失G = 函数值差异 + 导数差异
        # 权重可以根据需要调整，这里给函数值差异更高权重
        G = G_y + 0.1 * G_yx + 0.01 * G_yxx
        
        # 计算边界损失
        R_b, y1_b, y3_b = self.compute_boundary_residuals(x_b)
        R_b = R_b.mean()
        
        # t=0时降低边界权重，优先收敛起始系统
        alpha_dynamic = alpha if t > 0 else alpha * 0.1
        
        # 计算非零解惩罚项 L_nonzero = 1 / (||y||₂² + ε)
        # 解的L2范数平方：||y||₂² = (1/N_g) Σ y(x_i)²
        y_norm_squared = (y1**2).mean()  # 使用y1作为主要解变量
        L_nonzero = 1.0 / (y_norm_squared + epsilon)
        
        # 计算总损失：L_total(t) = t·F + (1-t)·G + α(t)·R_b + β·L_nonzero
        loss = t * F + (1 - t) * G + alpha_dynamic * R_b + beta * L_nonzero
        
        return loss, F, G, R_b, L_nonzero