import torch
import torch.optim as optim
import numpy as np
import time
import os
from tqdm import tqdm


class Trainer:
    """
    训练器类，负责模型的训练过程
    """
    def __init__(self, model, data_gen, lr=0.002, epochs=20000, n_homotopy_steps=11, 
                 decay_rate=0.9, alpha=10, T=600, v=50, omega2_init=1.0, 
                 homotopy_init_k=1, device='cpu', save_dir='results', verbose=1):
        """
        初始化训练器
        Args:
            model: 模型
            data_gen: 数据生成器
            lr: 学习率
            epochs: 每个同伦步骤的训练轮数
            n_homotopy_steps: 同伦步骤数
            decay_rate: 学习率衰减率
            alpha: 边界惩罚系数
            T: 方程参数T
            v: 方程参数v
            omega2_init: 特征值初始猜测
            homotopy_init_k: 起始函数的k值
            device: 计算设备
            save_dir: 保存目录
            verbose: 日志打印频率
        """
        self.model = model
        self.data_gen = data_gen
        self.lr = lr
        self.epochs = epochs
        self.n_homotopy_steps = n_homotopy_steps
        self.decay_rate = decay_rate
        self.alpha = alpha
        self.T = T
        self.v = v
        self.omega2_init = omega2_init
        self.homotopy_init_k = homotopy_init_k
        self.device = device
        self.save_dir = save_dir
        self.verbose = verbose
        
        # 生成数据（数据生成器已经处理了设备）
        self.x, self.x_b, self.x_test = data_gen.generate_all_data()
        
        # 初始化特征值
        self.omega2 = torch.tensor(omega2_init, device=device, requires_grad=True)
        
        # 初始化优化器
        self.optimizer = optim.Adam([
            {'params': model.parameters(), 'lr': lr},
            {'params': [self.omega2], 'lr': lr * 0.1}  # 特征值使用较小的学习率
        ])
    
    def train(self, k=1):
        """
        执行训练过程
        Args:
            k: 起始函数的k值
        Returns:
            model: 训练好的模型
            omega2: 学习到的特征值
            loss_history: 损失历史
        """
        # 重置模型参数和特征值
        self.model.apply(self._weights_init)
        self.omega2 = torch.tensor(self.omega2_init, device=self.device, requires_grad=True)
        
        # 重新初始化优化器
        self.optimizer = optim.Adam([
            {'params': self.model.parameters(), 'lr': self.lr},
            {'params': [self.omega2], 'lr': self.lr * 0.1}
        ])
        
        # 存储损失历史
        loss_history = {
            'total_loss': [],
            'F_loss': [],
            'G_loss': [],
            'R_b_loss': [],
            'omega2': [],
            'L_nonzero': []
        }
        
        # 同伦训练循环
        print(f"开始同伦训练，起始函数k={k}")
        
        for step in range(self.n_homotopy_steps):
            # 计算当前步骤的同伦参数t
            t = step / (self.n_homotopy_steps - 1)
            
            # 更新学习率
            current_lr = self.lr * (self.decay_rate ** step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            
            print(f"同伦步骤 {step+1}/{self.n_homotopy_steps}, t={t:.2f}, lr={current_lr:.6f}")
            
            # 训练当前步骤
            step_losses = self._train_step(t, k)
            
            # 记录损失
            loss_history['total_loss'].extend(step_losses['total_loss'])
            loss_history['F_loss'].extend(step_losses['F_loss'])
            loss_history['G_loss'].extend(step_losses['G_loss'])
            loss_history['R_b_loss'].extend(step_losses['R_b_loss'])
            loss_history['omega2'].extend(step_losses['omega2'])
            loss_history['L_nonzero'].extend(step_losses['L_nonzero'])
            
            # 打印当前步骤的结果
            print(f"  最终损失: {step_losses['total_loss'][-1]:.6f}, "
                  f"F损失: {step_losses['F_loss'][-1]:.6f}, "
                  f"G损失: {step_losses['G_loss'][-1]:.6f}, "
                  f"边界损失: {step_losses['R_b_loss'][-1]:.6f}, "
                  f"ω²: {step_losses['omega2'][-1]:.6f}, "
                  f"非零解惩罚: {step_losses['L_nonzero'][-1]:.6f}")
        
        # 最后一步使用L-BFGS精炼
        print("使用L-BFGS进行精炼...")
        self._refine_with_lbfgs(t=1.0, k=k)
        
        # 获取最终的特征值
        final_omega2 = self.omega2.item()
        
        return self.model, final_omega2, loss_history
    
    def _train_step(self, t, k):
        """
        训练单个同伦步骤
        Args:
            t: 同伦参数
            k: 起始函数的k值
        Returns:
            step_losses: 当前步骤的损失历史
        """
        step_losses = {
            'total_loss': [],
            'F_loss': [],
            'G_loss': [],
            'R_b_loss': [],
            'omega2': [],
            'L_nonzero': [],
        }
        
        # 训练循环
        for epoch in tqdm(range(self.epochs), disable=self.verbose == 0):
            self.optimizer.zero_grad()
            
            # 计算损失
            loss, F, G, R_b, L_nonzero = self.model.compute_homotopy_loss(
                self.x, self.x_b, t, self.T, self.v, self.omega2, self.alpha, k
            )
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 记录损失
            if epoch % 100 == 0:
                step_losses['total_loss'].append(loss.item())
                step_losses['F_loss'].append(F.item())
                step_losses['G_loss'].append(G.item())
                step_losses['R_b_loss'].append(R_b.item())
                step_losses['omega2'].append(self.omega2.item())
                step_losses['L_nonzero'].append(L_nonzero.item())
                
                if self.verbose > 1 and epoch % 1000 == 0:
                    print(f"    Epoch {epoch}: Loss={loss.item():.6f}, "
                          f"F={F.item():.6f}, G={G.item():.6f}, "
                          f"R_b={R_b.item():.6f}, ω²={self.omega2.item():.6f}")
        
        return step_losses
    
    def _refine_with_lbfgs(self, t, k):
        """
        使用L-BFGS进行精炼
        Args:
            t: 同伦参数
            k: 起始函数的k值
        """
        # 获取所有参数
        all_params = list(self.model.parameters()) + [self.omega2]
        
        # 创建L-BFGS优化器（不使用参数组）
        optimizer = optim.LBFGS(all_params, lr=1.0, max_iter=500)
        
        # 定义闭包函数
        def closure():
            optimizer.zero_grad()
            loss, _, _, _, _ = self.model.compute_homotopy_loss(
                self.x, self.x_b, t, self.T, self.v, self.omega2, self.alpha, k
            )
            loss.backward()
            return loss
        
        # 执行优化
        optimizer.step(closure)
    
    def _weights_init(self, m):
        """
        权重初始化函数
        Args:
            m: 网络层
        """
        if isinstance(m, torch.nn.Linear):
            # 使用更简单的权重初始化方法，避免PyTorch 2.8.0的兼容性问题
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)