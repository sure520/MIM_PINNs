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
    def __init__(self, width=30, depth=2,omega_deta=100, omega_low_2=None):
        super(MIM1, self).__init__()
        self.width = width
        self.depth = depth
        # 创建omega2参数时先不指定设备，在模型初始化后通过to(device)统一移动
        self.omega2 = nn.Parameter(torch.tensor(100.0, dtype=torch.float32, requires_grad=True))
        
        # 注册omega2参数，确保它能正确跟随模型设备移动
        self.register_parameter('omega2', self.omega2)
        
        # 输入层，将1维输入映射到width维
        self.input_layer = nn.Linear(1, width)
        
        # 残差块
        self.residual_blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(depth)])
        
        # 输出层，将width维映射到4维 (y1, y2, y3, y4)
        self.output_layer = nn.Linear(width, 4)
        
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
        
        # 输出特征值
        omega2 = self.omega2
            
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
    def __init__(self, width=30, depth=2, model_type='MIM1', device='cuda'):
        super(MIMHomPINNFusion, self).__init__()
        self.model_type = model_type
        
        # 确保device是字符串而不是字典
        if isinstance(device, dict) and 'device' in device:
            device = device['device']
        
        # 设置设备 - 统一处理设备参数
        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            # 默认使用cuda
            self.device = torch.device('cuda')
        
        # 根据模型类型创建网络
        if model_type == 'MIM1':
            self.model = MIM1(width, depth)
        elif model_type == 'MIM2':
            self.model = MIM2(width, depth)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
            
        # 将模型移动到指定设备
        self.model = self.model.to(self.device)
        # 确保设备一致性
    
    def _ensure_device_consistency(self):
        """
        确保所有模型参数都在正确的设备上
        """
        # 首先将整个模型移动到正确设备
        self.model = self.model.to(self.device)
        
        # 检查模型所有参数是否在正确设备上
        for name, param in self.model.named_parameters():
            if param.device != self.device:
                print(f"警告: 参数 {name} 设备不一致: {param.device} vs {self.device}")
                # 将参数移动到正确设备
                param.data = param.data.to(self.device)
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(self.device)
        
        # 检查模型所有缓冲区是否在正确设备上
        for name, buffer in self.model.named_buffers():
            if buffer.device != self.device:
                print(f"警告: 缓冲区 {name} 设备不一致: {buffer.device} vs {self.device}")
                # 将缓冲区移动到正确设备
                buffer.data = buffer.data.to(self.device)
    
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
    
    def compute_residuals(self, x, T=600, v=50):
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
        # 确保x有requires_grad=True，创建新的张量避免原地修改问题
        x = x.clone().detach().requires_grad_(True)
        
        # 前向传播
        y1, y2, y3, y4, omega2_val = self.forward(x)
        
        # 计算各阶导数
        try:
            # 确保使用正确的设备（字符串形式）
            device_str = str(self.device)
            if device_str.startswith('cuda:'):
                device_str = 'cuda'
            
            y1_x = torch.autograd.grad(
                outputs=y1, 
                inputs=x, 
                grad_outputs=torch.ones_like(y1, device=device_str), 
                create_graph=True, 
                retain_graph=True
            )[0]        
            y2_x = torch.autograd.grad(
                outputs=y1_x, 
                inputs=x, 
                grad_outputs=torch.ones_like(y1_x, device=device_str), 
                create_graph=True, 
                retain_graph=True
            )[0]
            y3_x = torch.autograd.grad(
                outputs=y2_x, 
                inputs=x, 
                grad_outputs=torch.ones_like(y2_x, device=device_str), 
                create_graph=True, 
                retain_graph=True
            )[0]
            y4_x = torch.autograd.grad(
                outputs=y3_x, 
                inputs=x, 
                grad_outputs=torch.ones_like(y3_x, device=device_str), 
                create_graph=True, 
                retain_graph=True
            )[0]
        except Exception as e:
            raise RuntimeError(f"计算导数时出错: {e}")
        
        # 确保所有张量形状一致（一维）
        y1_x = y1_x.squeeze()
        y2_x = y2_x.squeeze()
        y3_x = y3_x.squeeze()
        y4_x = y4_x.squeeze()
        
        # 计算残差项
        R1 = y2 - y1_x  # y2 - y1' = 0
        R2 = y3 - y2_x  # y3 - y2' = 0
        R3 = y4 - y3_x  # y4 - y3' = 0
        
        # 确保x是标量或与其他张量形状匹配
        if len(x.shape) > 1:
            x = x.squeeze()
        R4 = (y4_x - ((T + v*x) * y3 - v * y2 - omega2_val * y1))  # y4' - ((T+vx)y'' - vy' - ω²y) = 0
        
        # 在返回前确保所有残差项都是标量
        # 为了调试，我们直接返回每个残差项的最小二乘损失
        R1 = (R1**2).mean()
        R2 = (R2**2).mean()
        R3 = (R3**2).mean()
        R4 = (R4**2).mean()
        
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

        y1 = y1.squeeze()
        y3 = y3.squeeze()
        
        # 边界条件残差: y1(0)=0, y1(1)=0, y3(0)=0, y3(1)=0
        R_b = (y1**2).mean() + (y3**2).mean()
        
        # # 计算边界处的一阶导数（检查是否合理，非必须但可增强约束）
        # y1_x_b = torch.autograd.grad(y1, x_b, grad_outputs=torch.ones_like(y1), create_graph=True)[0].squeeze()
        # # 轻微惩罚异常导数
        # R_b = R_b + (y1_x_b**2).mean() * 0.1
        
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
        
        # 确保使用正确的设备（字符串形式）
        device_str = str(self.device)
        if device_str.startswith('cuda:'):
            device_str = 'cuda'
        
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

    def compute_residual_loss(self, x, T=600, v=50):
        """
        计算控制方程残差损失 L_r（核心物理约束）
        
        基于MIM降阶方法，将四阶方程转化为一阶系统：
        R1 = y2 - y1' = 0
        R2 = y3 - y2' = 0  
        R3 = y4 - y3' = 0
        R4 = y4' - ((T+vx)y'' + vy' + ω²y) = 0
        
        Args:
            x: 内部点
            T: 方程参数T
            v: 方程参数v
        Returns:
            L_r: 残差损失（标量）
            y1, y2, y3, y4, omega2_val: 各变量值
        """
        if len(x.shape) == 1:
            x = x.reshape(-1, 1).requires_grad_(True)
        
        # 确保x有requires_grad=True
        x.requires_grad_(True)
        
        # 前向传播
        y1, y2, y3, y4, omega2_val = self.forward(x)
        
        # 计算各阶导数
        try:
            y1_x = torch.autograd.grad(
                outputs=y1, 
                inputs=x, 
                grad_outputs=torch.ones_like(y1, device=self.device), 
                create_graph=True, 
                retain_graph=True
            )[0]        
            y2_x = torch.autograd.grad(
                outputs=y1_x, 
                inputs=x, 
                grad_outputs=torch.ones_like(y1_x, device=self.device), 
                create_graph=True, 
                retain_graph=True
            )[0]
            y3_x = torch.autograd.grad(
                outputs=y2_x, 
                inputs=x, 
                grad_outputs=torch.ones_like(y2_x, device=self.device), 
                create_graph=True, 
                retain_graph=True
            )[0]
            y4_x = torch.autograd.grad(
                outputs=y3_x, 
                inputs=x, 
                grad_outputs=torch.ones_like(y3_x, device=self.device), 
                create_graph=True, 
                retain_graph=True
            )[0]
        except Exception as e:
            raise RuntimeError(f"计算导数时出错: {e}")
        
        # 计算残差项（保持张量形状，避免squeeze操作）
        R1 = y2 - y1_x  # y2 - y1' = 0
        R2 = y3 - y2_x  # y3 - y2' = 0
        R3 = y4 - y3_x  # y4 - y3' = 0
        
        # 确保x与其他张量形状匹配，避免squeeze操作破坏梯度链
        # 使用广播机制确保形状匹配，而不是squeeze
        if len(x.shape) > 1:
            x_reshaped = x.view(-1)  # 使用view而不是squeeze
        else:
            x_reshaped = x
        
        # 计算R4残差项，确保所有操作都在计算图中
        # 使用广播机制确保形状匹配
        R4 = (y4_x - ((T + v*x_reshaped.unsqueeze(1)) * y3 - v * y2 - omega2_val * y1))  # y4' - ((T+vx)y'' - vy' - ω²y) = 0
        
        # 计算每个残差项的均方损失，保持梯度计算链
        R1_loss = (R1**2).mean()
        R2_loss = (R2**2).mean()
        R3_loss = (R3**2).mean()
        R4_loss = (R4**2).mean()
        
        # 计算总残差损失
        L_r = R1_loss + R2_loss + R3_loss + R4_loss
        
        return L_r, y1, y2, y3, y4, omega2_val
    
    def compute_boundary_loss(self, x_b, T=600, v=50):
        """
        计算边界条件损失 L_b（硬约束保障）
        
        针对简支-简支边界条件：y(0)=y(1)=0, y''(0)=y''(1)=0
        
        Args:
            x_b: 边界点
            T: 方程参数T
            v: 方程参数v
            omega2: 特征值
        Returns:
            L_b: 边界损失（标量）
        """
        # 前向传播
        y1, y2, y3, y4, omega2 = self.forward(x_b)

        # 边界条件残差: y1(0)=0, y1(1)=0, y3(0)=0, y3(1)=0
        # 避免squeeze操作，直接计算均方损失
        L_b = (y1**2).mean() + (y3**2).mean()
        
        return L_b
    
    def compute_amplitude_constraint_loss(self, x_a=0.5, y_a=1.0, T=600, v=50):
        """
        计算振幅约束损失 L_a（规避模态歧义问题）
        
        特征值问题中，同一特征值对应无穷多倍乘关系的eigenfunction，
        需通过振幅约束固定模态形态，避免训练震荡。
        
        Args:
            x_a: 约束点位置（避开潜在节点，默认x=0.5）
            y_a: 约束点目标值（任意非零值，默认1.0）
            T: 方程参数T
            v: 方程参数v
            omega2: 特征值
        Returns:
            L_a: 振幅约束损失
        """
        # 确保x_a是tensor格式，并且是二维的 [n, 1]
        if not isinstance(x_a, torch.Tensor):
            x_a = torch.tensor([[x_a]], dtype=torch.float32, device=self.device).requires_grad_(True)
        elif len(x_a.shape) == 1:
            x_a = x_a.reshape(-1, 1).requires_grad_(True)
        
        # 在约束点处计算解的值
        y1_a, _, _, _, _ = self.forward(x_a)
        
        # 确保y_a是正确形状的tensor
        if not isinstance(y_a, torch.Tensor):
            y_a = torch.tensor(y_a, dtype=torch.float32, device=self.device)
        
        # 振幅约束损失：在约束点处解的值与目标值的平方差
        L_a = ((y1_a - y_a)**2).mean()
        
        return L_a
    
    def compute_eigenvalue_hierarchy_loss(self, omega2_val, omega_low_2, k=1, epsilon=5.0, a=20.0):
        """
        计算特征值层级约束损失 L_c（实现多阶特征值引导功能）
        
        利用已求解的低阶特征值ω_low²，引导高阶特征值收敛，避免特征值趋同问题。
        
        Args:
            omega2_val: 当前特征值
            omega_low_2: 低阶特征值
            k: 当前训练的特征值阶数（k=1表示一阶特征值，不使用层级约束）
            epsilon: 安全边际（默认5.0）
        Returns:
            L_c: 特征值层级约束损失（标量）
        """
        # 对于一阶特征值(k=1)，不使用层级约束，直接返回0
        if k <= 1 or omega_low_2 is None:
            return torch.tensor(0.0, device=omega2_val.device)
        
        # 对于高阶特征值(k>1)，使用Sigmoid函数实现特征值范围约束
        # L_c = -σ(a·(ω² - (ω_low² + ε))) + 1
        z = a * (omega2_val - (omega_low_2 + epsilon))
        sigma_z = 1.0 / (1.0 + torch.exp(-z))  # Sigmoid函数
        L_c = -sigma_z + 1.0
        
        # 确保返回的是标量张量
        if len(L_c.shape) > 0:
            L_c = L_c.mean()
        
        return L_c
    
    def compute_nonzero_solution_loss(self, y1, epsilon=1e-6):
        """
        计算非零解惩罚损失 L_nz（排除零解）
        
        零解（y(x)≡0）对任意ω²均满足方程，需通过惩罚强制排除。
        
        Args:
            y1: 主要解变量（位移）
            epsilon: 数值稳定项（默认1e-6）
        Returns:
            L_nz: 非零解惩罚损失（标量）
        """
        # 确保y1是正确形状的tensor
        if not isinstance(y1, torch.Tensor):
            y1 = torch.tensor(y1, dtype=torch.float32, device=self.device)
        
        # 解的L2范数平方：||y||₂² = (1/N) Σ y(x_i)²，确保是标量
        y_norm_squared = (y1**2).mean()
        if len(y_norm_squared.shape) > 0:
            y_norm_squared = y_norm_squared.mean()
        
        # 非零解惩罚：L_nz = 1 / (||y||₂² + ε)，确保是标量
        L_nz = 1.0 / (y_norm_squared + epsilon)
        if len(L_nz.shape) > 0:
            L_nz = L_nz.mean()
        
        return L_nz
    
    def compute_total_loss(self, x, x_b, T=600, v=50, omega_low_2=None, k=1,
                          weights=None, x_a=0.5, y_a=1.0, epsilon_hierarchy=5.0, a=20.0, epsilon_nonzero=1e-6):
        """
        计算完整的总损失函数
        
        总损失：L_total = ω_r·L_r + ω_b·L_b + ω_a·L_a + ω_c·L_c + ω_nz·L_nz
        
        Args:
            x: 内部点
            x_b: 边界点
            T: 方程参数T
            v: 方程参数v
            omega2: 特征值
            omega_low_2: 低阶特征值（用于层级约束）
            k: 当前训练的特征值阶数（k=1表示一阶特征值，不使用层级约束）
            weights: 各损失项权重字典
            x_a: 振幅约束点位置
            y_a: 振幅约束目标值
            a: 层级约束缩放参数
            epsilon_hierarchy: 层级约束安全边际
            epsilon_nonzero: 非零解惩罚数值稳定项
        Returns:
            total_loss: 总损失（标量）
            loss_dict: 各损失项详细值
        """
        # 默认权重设置（可根据训练阶段动态调整）
        default_weights = {
            'residual': 1.0,      # ω_r: 残差损失权重
            'boundary': 100.0,    # ω_b: 边界损失权重
            'amplitude': 100.0,   # ω_a: 振幅约束权重
            'hierarchy': 100.0,   # ω_c: 层级约束权重
            'nonzero': 1e-4       # ω_nz: 非零解惩罚权重
        }
        
        if weights is not None:
            default_weights.update(weights)
        
        weights = default_weights
        
        
        # 确保所有输入都是在正确的设备上并且需要梯度
        x = x.to(self.device).requires_grad_(True)
        x_b = x_b.to(self.device).requires_grad_(True)
        if omega_low_2 is not None:
            omega_low_2 = omega_low_2.to(self.device)

        
        # 计算各损失项
        # 1. 控制方程残差损失
        L_r, y1, y2, y3, y4, omega2_val = self.compute_residual_loss(x, T, v)
        # 确保L_r是标量
        L_r = L_r.mean() if len(L_r.shape) > 0 else L_r
        
        # 2. 边界条件损失
        L_b = self.compute_boundary_loss(x_b)
        # 确保L_b是标量
        L_b = L_b.mean() if len(L_b.shape) > 0 else L_b
        
        # 3. 振幅约束损失
        L_a = self.compute_amplitude_constraint_loss(x_a, y_a, T, v)
        # 确保L_a是标量
        L_a = L_a.mean() if len(L_a.shape) > 0 else L_a
        
        # 4. 特征值层级约束损失
        L_c = self.compute_eigenvalue_hierarchy_loss(omega2_val, omega_low_2, k, epsilon_hierarchy, a)
        # 确保L_c是标量
        L_c = L_c.mean() if len(L_c.shape) > 0 else L_c
        
        # 5. 非零解惩罚损失
        L_nz = self.compute_nonzero_solution_loss(y1, epsilon_nonzero)
        # 确保L_nz是标量
        L_nz = L_nz.mean() if len(L_nz.shape) > 0 else L_nz
        
        # 计算加权损失项，确保每一步都是标量操作
        weighted_L_r = weights['residual'] * L_r
        
        weighted_L_b = weights['boundary'] * L_b
        
        weighted_L_a = weights['amplitude'] * L_a
        
        weighted_L_c = weights['hierarchy'] * L_c
        
        weighted_L_nz = weights['nonzero'] * L_nz
        
        # 计算总损失
        total_loss = weighted_L_r + weighted_L_b + weighted_L_a + weighted_L_c + weighted_L_nz
        
        # 返回详细损失信息
        loss_dict = {
            'total_loss': total_loss,
            'residual_loss': L_r,
            'boundary_loss': L_b,
            'amplitude_loss': L_a,
            'hierarchy_loss': L_c,
            'nonzero_loss': L_nz,
            'omega2': omega2_val
        }
        
        return total_loss, loss_dict
