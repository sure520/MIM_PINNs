"""
无需同伦步骤的直接训练器
基于MIM方法，直接求解PDE，不依赖同伦连续性方法
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import os
import sys
from tqdm import tqdm
import logging
from datetime import datetime

# 添加项目根目录到路径，以便导入配置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


try:
    from configs.direct_config import get_direct_config, validate_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("警告: 无法导入配置文件，使用默认配置")

# 导入新的融合模型
try:
    from models.fusion_model import MIMHomPINNFusion
    FUSION_MODEL_AVAILABLE = True
except ImportError:
    FUSION_MODEL_AVAILABLE = False
    print("警告: 无法导入融合模型，使用原有模型架构")


class DirectTrainer:
    """
    无需同伦步骤的直接训练器
    直接优化PDE残差和边界条件，不依赖同伦连续性方法
    """
    
    def __init__(self, model, data_gen, config=None, config_type='balanced', device='cuda' if torch.cuda.is_available() else 'cpu', save_dir='results'):
        """
        初始化训练器
        
        Args:
            model: PINNs模型
            data_gen: 数据生成器实例
            config: 训练配置字典或配置类型字符串
            config_type: 配置类型（当config为None时使用）
            device: 计算设备
            save_dir: 结果保存目录
        """
        self.model = model
        self.data_gen = data_gen
        self.device = device
        self.save_dir = save_dir
        
        # 配置处理
        if config is None:
            # 使用预设配置类型
            if CONFIG_AVAILABLE:
                self.config = get_direct_config(config_type)
            else:
                self.config = self._get_default_config()
        elif isinstance(config, str):
            # config是配置类型字符串
            if CONFIG_AVAILABLE:
                self.config = get_direct_config(config)
            else:
                self.config = self._get_default_config()
        else:
            # config是配置字典
            if CONFIG_AVAILABLE:
                # 验证配置
                is_valid, errors = validate_config(config)
                if not is_valid:
                    print("配置验证失败，使用默认配置:")
                    for error in errors:
                        print(f"  - {error}")
                    self.config = self._get_default_config()
                else:
                    self.config = config
            else:
                self.config = config
        
        # 确保配置完整性
        self._ensure_config_completeness()
        
        # 初始化训练状态
        self.epoch = 0
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.omega2_values = []
        
        # 设置日志记录
        self._setup_logging()
        
        # 移动到设备
        self.model.to(self.device)
        
        # 生成训练数据
        self.x, self.x_b, self.x_test = data_gen.generate_all_data(
            N_f=self.config['data']['N_f'],
            N_b=self.config['data']['N_b'],
            N_test=self.config['data']['N_test'],
            domain=self.config['data']['domain']
        )
        
        # 转换为tensor并移动到设备
        self.x = torch.tensor(self.x, dtype=torch.float32, requires_grad=True).to(self.device)
        self.x_b = torch.tensor(self.x_b, dtype=torch.float32).to(self.device)
        self.x_test = torch.tensor(self.x_test, dtype=torch.float32).to(self.device)
        
        # 初始化特征值
        self.omega2 = torch.tensor(
            self.config['training']['omega2_init'], 
            device=device, 
            requires_grad=True
        )
        
        # 设置优化器
        self.optimizer = self._setup_optimizer()
        
        # 设置学习率调度器
        self.scheduler = self._setup_scheduler()
        
        # 创建保存目录
        self._create_save_dir()
        
        # 记录初始化信息
        self._log_init_info()
        
        # 训练历史记录 - 适配新的损失函数结构
        self.history = {
            'total_loss': [],
            'pde_loss': [],           # 残差损失
            'bc_loss': [],            # 边界损失
            'amplitude_loss': [],     # 振幅约束损失
            'nonzero_loss': [],       # 非零解惩罚损失
            'hierarchy_loss': [],    # 层级约束损失（可选）
            'omega2': [],
            'learning_rate': []
        }
        
        # 训练开始时间
        self.start_time = None
    
    def _get_default_config(self):
        """获取默认配置"""
        return {
            'training': {
                'lr': 0.001,
                'epochs': 50000,
                'optimizer': 'adam',
                'lr_scheduler': 'step',
                'lr_decay_rate': 0.95,
                'lr_decay_steps': 5000,
                'omega2_init': 1.0,
                'alpha': 10.0,
                'beta': 1e-4,
                'verbose': 1,
                'save_interval': 1000,
                'early_stopping': True,
                'patience': 1000,
                'min_delta': 1e-6
            },
            'data': {
                'N_f': 5000,
                'N_b': 200,
                'N_test': 1000,
                'domain': [0, 1]
            },
            'equation': {
                'T': 600,
                'v': 50
            }
        }
    
    def _ensure_config_completeness(self):
        """确保配置完整性"""
        default_config = self._get_default_config()
        
        # 检查必需字段
        for section in ['training', 'data', 'equation']:
            if section not in self.config:
                self.config[section] = default_config[section].copy()
            else:
                # 确保每个section中的必需字段存在
                for key, default_value in default_config[section].items():
                    if key not in self.config[section]:
                        self.config[section][key] = default_value
    
    def _create_save_dir(self):
        """创建保存目录"""
        os.makedirs(self.save_dir, exist_ok=True)
    
    def _log_init_info(self):
        """记录初始化信息"""
        self.logger.info("直接训练器初始化完成")
        self.logger.info(f"设备: {self.device}")
        self.logger.info(f"保存目录: {self.save_dir}")
        self.logger.info(f"配置: {json.dumps(self.config, indent=2, ensure_ascii=False)}")
        
    def _setup_config(self, config):
        """设置训练配置（兼容旧版本）"""
        # 如果config是字典，将其转换为新的配置格式
        if isinstance(config, dict):
            # 检查是否是旧格式（平铺配置）
            if 'training' not in config:
                # 转换为新格式
                new_config = self._get_default_config()
                for key, value in config.items():
                    if key in new_config['training']:
                        new_config['training'][key] = value
                    elif key in new_config['data']:
                        new_config['data'][key] = value
                    elif key in new_config['equation']:
                        new_config['equation'][key] = value
                return new_config
        
        return config
    
    def _deep_update(self, target, source):
        """深度更新字典"""
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def _setup_optimizer(self):
        """设置优化器"""
        # 获取模型参数和特征值参数
        model_params = list(self.model.parameters())
        omega2_param = torch.nn.Parameter(torch.tensor([self.config['training']['omega2_init']], dtype=torch.float32))
        
        # 分别优化模型参数和特征值
        self.optimizer = self._create_optimizer(model_params + [omega2_param])
        self.omega2 = omega2_param
        
        self.logger.info(f"优化器设置完成: {self.config['training']['optimizer']}")
        
        return self.optimizer
    
    def _create_optimizer(self, params):
        """创建优化器"""
        optimizer_type = self.config['training']['optimizer'].lower()
        lr = self.config['training']['lr']
        
        if optimizer_type == 'adam':
            return optim.Adam(params, lr=lr)
        elif optimizer_type == 'sgd':
            return optim.SGD(params, lr=lr, momentum=0.9)
        elif optimizer_type == 'rmsprop':
            return optim.RMSprop(params, lr=lr)
        else:
            self.logger.warning(f"未知优化器类型: {optimizer_type}, 使用Adam")
            return optim.Adam(params, lr=lr)
    
    def _setup_scheduler(self):
        """设置学习率调度器"""
        scheduler_type = self.config['training']['lr_scheduler'].lower()
        
        if scheduler_type == 'step':
            return StepLR(self.optimizer, step_size=self.config['training']['lr_decay_steps'], gamma=self.config['training']['lr_decay_rate'])
        elif scheduler_type == 'exponential':
            return ExponentialLR(self.optimizer, gamma=self.config['training']['lr_decay_rate'])
        elif scheduler_type == 'plateau':
            return ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=500, verbose=True)
        else:
            return None
    
    def _setup_logging(self):
        """设置日志记录"""
        # 创建日志目录
        log_dir = os.path.join(self.save_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置日志文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'direct_trainer_{timestamp}.log')
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def compute_loss(self, x, x_b):
        """
        计算总损失 - 使用新的MIMHomPINNFusion模型损失函数
        
        Args:
            x: 内部点
            x_b: 边界点
            
        Returns:
            total_loss: 总损失
            loss_dict: 各损失项详细值
        """
        # 获取方程参数
        T = self.config['equation']['T']
        v = self.config['equation']['v']
        
        # 振幅约束点位置（默认在域中点）
        x_a = torch.tensor([0.5], dtype=torch.float32, device=self.device)
        y_a = 1.0  # 振幅约束目标值
        
        # 使用模型的总损失函数
        total_loss, loss_dict = self.model.compute_total_loss(
            x=x, 
            x_b=x_b, 
            T=T, 
            v=v, 
            omega2=self.omega2,
            weights={
                'residual': 1.0,
                'boundary': self.config['training']['alpha'],
                'amplitude': 100.0,
                'hierarchy': 100.0,
                'nonzero': self.config['training']['beta']
            },
            x_a=x_a,
            y_a=y_a
        )
        
        return total_loss, loss_dict
    
    def _compute_pde_residual(self, y1, y2, y3, y4, omega2_val):
        """计算PDE残差"""
        T = self.config['equation']['T']
        v = self.config['equation']['v']
        
        # 根据MIM方法的一阶系统计算残差
        # 这里需要根据具体的PDE方程实现
        # 示例：四阶方程 y'''' + T*y'' + v*y = omega2 * y
        residual = y4 + T * y2 + v * y1 - omega2_val * y1
        
        return residual
    
    def _compute_boundary_loss(self, x_b):
        """计算边界条件损失"""
        # 获取边界点上的模型输出
        y1_b, _, y3_b, _, _ = self.model(x_b)
        
        # 边界条件：y(0)=0, y(1)=0, y''(0)=0, y''(1)=0
        bc_loss = torch.mean(y1_b**2) + torch.mean(y3_b**2)
        
        return bc_loss
    
    def _compute_nonzero_loss(self, y1):
        """计算非零解惩罚损失"""
        # 确保解不为零
        epsilon = 1e-6
        nonzero_loss = 1.0 / (torch.mean(y1**2) + epsilon)
        
        return nonzero_loss
    
    def train(self):
        """执行训练循环"""
        self.start_time = time.time()
        
        # 训练进度条
        pbar = tqdm(range(self.config['training']['epochs']), desc="训练进度")
        
        for epoch in pbar:
            self.epoch = epoch
            
            # 单步训练
            train_loss, loss_dict = self._train_step()
            
            # 记录损失
            self._record_losses(train_loss, loss_dict)
            
            # 更新进度条
            self._update_progress_bar(pbar, train_loss, loss_dict)
            
            # 检查早停
            if self._check_early_stopping():
                self.logger.info(f"早停触发于第 {epoch} 轮")
                break
            
            # 保存检查点
            if epoch % self.config['training']['save_interval'] == 0:
                self._save_checkpoint(epoch)
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
        
        # 训练完成
        self._finalize_training()
    
    def _record_history(self, total_loss, loss_dict, current_lr):
        """记录训练历史 - 适配新的损失函数结构"""
        self.history['total_loss'].append(total_loss.item())
        self.history['pde_loss'].append(loss_dict['residual_loss'].item())
        self.history['bc_loss'].append(loss_dict['boundary_loss'].item())
        self.history['amplitude_loss'].append(loss_dict['amplitude_loss'].item())
        self.history['nonzero_loss'].append(loss_dict['nonzero_loss'].item())
        self.history['omega2'].append(loss_dict['omega2'].item())
        self.history['learning_rate'].append(current_lr)
        
        # 如果有层级约束损失，也记录下来
        if loss_dict['hierarchy_loss'] is not None:
            self.history['hierarchy_loss'].append(loss_dict['hierarchy_loss'].item())
    
    def _log_training_info(self, epoch, total_loss, loss_dict):
        """记录训练信息 - 适配新的损失函数结构"""
        current_lr = self.optimizer.param_groups[0]['lr']
        
        log_message = (
            f"Epoch {epoch}: "
            f"Total Loss: {total_loss.item():.6f}, "
            f"Residual Loss: {loss_dict['residual_loss'].item():.6f}, "
            f"Boundary Loss: {loss_dict['boundary_loss'].item():.6f}, "
            f"Amplitude Loss: {loss_dict['amplitude_loss'].item():.6f}, "
            f"Nonzero Loss: {loss_dict['nonzero_loss'].item():.6f}, "
            f"Omega²: {loss_dict['omega2'].item():.6f}, "
            f"LR: {current_lr:.6f}"
        )
        
        # 如果有层级约束损失，也显示出来
        if loss_dict['hierarchy_loss'] is not None:
            log_message += f", Hierarchy Loss: {loss_dict['hierarchy_loss'].item():.6f}"
        
        self.logger.info(log_message)
    
    def _train_step(self):
        """单步训练"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # 计算损失
        total_loss, loss_dict = self.compute_loss(self.x, self.x_b)
        
        # 反向传播
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss, loss_dict
    
    def _record_losses(self, total_loss, loss_dict):
        """记录损失"""
        current_lr = self.optimizer.param_groups[0]['lr']
        self._record_history(total_loss, loss_dict, current_lr)
        
        # 更新最佳损失
        if total_loss.item() < self.best_loss:
            self.best_loss = total_loss.item()
            self.best_epoch = self.epoch
            self._save_best_model()
    
    def _update_progress_bar(self, pbar, total_loss, loss_dict):
        """更新进度条 - 适配新的损失函数结构"""
        pbar.set_postfix({
            'Total': f"{total_loss.item():.4f}",
            'Residual': f"{loss_dict['residual_loss'].item():.4f}",
            'Boundary': f"{loss_dict['boundary_loss'].item():.4f}",
            'Amplitude': f"{loss_dict['amplitude_loss'].item():.4f}",
            'Omega²': f"{loss_dict['omega2'].item():.4f}"
        })
    
    def _check_early_stopping(self):
        """检查早停条件"""
        if not self.config['training']['early_stopping']:
            return False
        
        if len(self.history['total_loss']) < self.config['training']['patience']:
            return False
        
        # 检查最近patience轮内损失是否没有显著改善
        recent_losses = self.history['total_loss'][-self.config['training']['patience']:]
        min_recent_loss = min(recent_losses)
        current_loss = self.history['total_loss'][-1]
        
        if current_loss - min_recent_loss > self.config['training']['min_delta']:
            return True
        
        return False
    
    def _finalize_training(self):
        """训练完成后的处理"""
        # 加载最佳模型
        self._load_best_model()
        
        # 记录训练总结
        training_time = time.time() - self.start_time
        self.logger.info(f"训练完成，总耗时: {training_time:.2f}秒")
        self.logger.info(f"最佳损失: {self.best_loss:.6f} (第{self.best_epoch}轮)")
        
        # 保存训练历史
        self._save_training_history()
    
    def _save_training_history(self):
        """保存训练历史"""
        history_dir = os.path.join(self.save_dir, 'history')
        os.makedirs(history_dir, exist_ok=True)
        
        history_file = os.path.join(history_dir, 'training_history.json')
        
        # 转换为可序列化的格式
        serializable_history = {}
        for key, values in self.history.items():
            serializable_history[key] = [float(v) if torch.is_tensor(v) else v for v in values]
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"训练历史已保存到: {history_file}")
    
    def _save_checkpoint(self, epoch):
        """保存检查点"""
        checkpoint_dir = os.path.join(self.save_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'omega2': self.omega2.item(),
            'history': self.history,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
    
    def _save_best_model(self):
        """保存最佳模型"""
        model_dir = os.path.join(self.save_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'best_model.pth')
        
        best_model_state = {
            'model_state_dict': self.model.state_dict(),
            'omega2': self.omega2.item(),
            'config': self.config
        }
        
        torch.save(best_model_state, model_path)
    
    def _load_best_model(self):
        """加载最佳模型"""
        model_path = os.path.join(self.save_dir, 'models', 'best_model.pth')
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.omega2 = torch.tensor(
                checkpoint['omega2'], 
                device=self.device, 
                requires_grad=True
            )
            self.logger.info("已加载最佳模型")
    
    def _weights_init(self, m):
        """权重初始化函数"""
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    def evaluate(self):
        """
        评估训练好的模型
        
        Returns:
            eval_results: 评估结果字典
        """
        self.model.eval()
        
        with torch.no_grad():
            # 在测试点上计算预测
            y1_test, y2_test, y3_test, y4_test, omega2_test = self.model(self.x_test)
            
            # 计算PDE残差
            pde_residual = self._compute_pde_residual(y1_test, y2_test, y3_test, y4_test, omega2_test)
            pde_error = torch.mean(pde_residual**2).item()
            
            # 计算边界条件误差
            bc_error = self._compute_boundary_loss(self.x_b).item()
            
            # 计算解的L2范数
            solution_norm = torch.mean(y1_test**2).item()
            
            eval_results = {
                'pde_error': pde_error,
                'bc_error': bc_error,
                'solution_norm': solution_norm,
                'omega2': self.omega2.item(),
                'final_loss': self.history['total_loss'][-1] if self.history['total_loss'] else float('inf')
            }
        
        self.logger.info(f"评估结果: PDE误差={pde_error:.6f}, "
                        f"边界误差={bc_error:.6f}, "
                        f"特征值={eval_results['omega2']:.6f}")
        
        return eval_results


def create_direct_trainer(model, data_gen, config=None, device='cpu', save_dir='results'):
    """
    创建直接训练器的便捷函数
    
    Args:
        model: 模型实例
        data_gen: 数据生成器实例
        config: 配置字典
        device: 计算设备
        save_dir: 保存目录
        
    Returns:
        trainer: DirectTrainer实例
    """
    return DirectTrainer(model, data_gen, config, 'balanced', device, save_dir)