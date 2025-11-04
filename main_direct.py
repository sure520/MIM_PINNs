"""
MIM-HomPINNs 直接训练主程序
使用DirectTrainer类进行训练，无需同伦步骤
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
from utils.data_generator import DataGenerator
from utils.direct_trainer import DirectTrainer, create_direct_trainer
from models.fusion_model import MIMHomPINNFusion
from configs.direct_config import get_direct_config, get_custom_config, print_config_summary

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_save_directory(base_dir="results"):
    """创建保存目录"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_dir, f"direct_training_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建子目录
    os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)
    
    return save_dir

def run_direct_training(k_values, config, device, save_dir):
    """
    使用DirectTrainer运行直接训练
    
    Args:
        k_values: 要尝试的k值列表
        config: 训练配置
        device: 计算设备
        save_dir: 保存目录
        
    Returns:
        results: 训练结果字典
    """
    results = {}
    
    print(f"开始直接训练，k值范围: {k_values}")
    print(f"使用设备: {device}")
    print(f"结果保存目录: {save_dir}")
    
    for k in k_values:
        print(f"\n正在训练 k={k} 的模型...")
        
        try:
            # 创建数据生成器
            data_gen = DataGenerator(domain=[0, 1])
            
            # 创建模型
            model = MIMHomPINNFusion()
            
            # 创建直接训练器
            trainer = create_direct_trainer(
                model=model,
                data_gen=data_gen,
                config=config,
                device=device,
                save_dir=save_dir
            )
            
            # 开始训练
            start_time = time.time()
            trainer.train()
            training_time = time.time() - start_time
            
            # 评估模型
            eval_results = trainer.evaluate()
            
            # 保存结果
            results[k] = {
                'training_time': training_time,
                'eval_results': eval_results,
                'history': trainer.history,
                'omega2': eval_results['omega2']
            }
            
            print(f"k={k} 训练完成，用时: {training_time:.2f}秒")
            print(f"最终损失: {eval_results['final_loss']:.6f}")
            print(f"特征值: {eval_results['omega2']:.6f}")
            
            # 保存单个模型的结果
            model_save_path = os.path.join(save_dir, "models", f"model_k{k}.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'omega2': eval_results['omega2'],
                'k': k,
                'config': config
            }, model_save_path)
            
            # 保存训练日志
            log_save_path = os.path.join(save_dir, "logs", f"training_log_k{k}.txt")
            with open(log_save_path, 'w') as f:
                f.write(f"k={k} 训练日志\n")
                f.write(f"训练时间: {training_time:.2f}秒\n")
                f.write(f"最终损失: {eval_results['final_loss']:.6f}\n")
                f.write(f"特征值: {eval_results['omega2']:.6f}\n")
                f.write(f"PDE误差: {eval_results['pde_error']:.6f}\n")
                f.write(f"边界误差: {eval_results['bc_error']:.6f}\n")
            
        except Exception as e:
            print(f"k={k} 训练失败: {str(e)}")
            results[k] = {'error': str(e)}
    
    # 保存所有结果
    results_save_path = os.path.join(save_dir, "training_results.json")
    with open(results_save_path, 'w') as f:
        # 转换numpy类型为Python原生类型，以便JSON序列化
        serializable_results = {}
        for k, result in results.items():
            if 'error' not in result:
                serializable_results[k] = {
                    'training_time': result['training_time'],
                    'eval_results': {key: float(value) if isinstance(value, (np.float32, np.float64)) else value 
                                    for key, value in result['eval_results'].items()},
                    'omega2': float(result['omega2']) if isinstance(result['omega2'], (np.float32, np.float64)) else result['omega2']
                }
            else:
                serializable_results[k] = result
        
        json.dump(serializable_results, f, indent=4)
    
    return results

def visualize_results(results, save_dir):
    """
    可视化训练结果
    
    Args:
        results: 训练结果字典
        save_dir: 保存目录
    """
    # 创建损失历史图
    plt.figure(figsize=(12, 8))
    
    # 子图1: 总损失
    plt.subplot(2, 2, 1)
    for k, result in results.items():
        if 'error' not in result and 'history' in result:
            plt.plot(result['history']['total_loss'], label=f'k={k}')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('总损失历史')
    plt.legend()
    plt.grid(True)
    
    # 子图2: PDE残差损失
    plt.subplot(2, 2, 2)
    for k, result in results.items():
        if 'error' not in result and 'history' in result:
            plt.plot(result['history']['pde_loss'], label=f'k={k}')
    plt.xlabel('Epoch')
    plt.ylabel('PDE Loss')
    plt.title('PDE残差损失历史')
    plt.legend()
    plt.grid(True)
    
    # 子图3: 边界条件损失
    plt.subplot(2, 2, 3)
    for k, result in results.items():
        if 'error' not in result and 'history' in result:
            plt.plot(result['history']['bc_loss'], label=f'k={k}')
    plt.xlabel('Epoch')
    plt.ylabel('BC Loss')
    plt.title('边界条件损失历史')
    plt.legend()
    plt.grid(True)
    
    # 子图4: 特征值变化
    plt.subplot(2, 2, 4)
    for k, result in results.items():
        if 'error' not in result and 'history' in result:
            plt.plot(result['history']['omega2'], label=f'k={k}')
    plt.xlabel('Epoch')
    plt.ylabel('Omega²')
    plt.title('特征值变化历史')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "figures", "loss_history.png"), dpi=300)
    plt.close()
    
    # 创建k值与最终损失的关系图
    plt.figure(figsize=(10, 6))
    
    k_vals = []
    final_losses = []
    omega2_vals = []
    
    for k, result in results.items():
        if 'error' not in result and 'eval_results' in result:
            k_vals.append(k)
            final_losses.append(result['eval_results']['final_loss'])
            omega2_vals.append(result['eval_results']['omega2'])
    
    # 子图1: 最终损失
    plt.subplot(1, 2, 1)
    plt.plot(k_vals, final_losses, 'o-')
    plt.xlabel('k值')
    plt.ylabel('最终损失')
    plt.title('k值与最终损失的关系')
    plt.grid(True)
    
    # 子图2: 特征值
    plt.subplot(1, 2, 2)
    plt.plot(k_vals, omega2_vals, 'o-')
    plt.xlabel('k值')
    plt.ylabel('特征值')
    plt.title('k值与特征值的关系')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "figures", "k_analysis.png"), dpi=300)
    plt.close()

def print_results_summary(results):
    """
    打印结果总结
    
    Args:
        results: 训练结果字典
    """
    print("\n===== 训练结果总结 =====")
    
    successful_k = []
    failed_k = []
    
    for k, result in results.items():
        if 'error' in result:
            failed_k.append((k, result['error']))
        else:
            successful_k.append(k)
    
    print(f"成功训练的k值: {successful_k}")
    print(f"训练失败的k值: {[k for k, _ in failed_k]}")
    
    if failed_k:
        print("\n失败原因:")
        for k, error in failed_k:
            print(f"  k={k}: {error}")
    
    if successful_k:
        print("\n最佳结果:")
        best_k = None
        best_loss = float('inf')
        
        for k in successful_k:
            final_loss = results[k]['eval_results']['final_loss']
            if final_loss < best_loss:
                best_loss = final_loss
                best_k = k
        
        print(f"  最佳k值: {best_k}")
        print(f"  最佳损失: {best_loss:.6f}")
        print(f"  对应特征值: {results[best_k]['eval_results']['omega2']:.6f}")
        print(f"  训练时间: {results[best_k]['training_time']:.2f}秒")

def main(config_type='balanced', k_values=None, **kwargs):
    """
    主函数
    
    Args:
        config_type: 配置类型 ('balanced', 'quick', 'high_precision')
        k_values: 要尝试的k值列表，默认为[2, 3, 4, 5]
        **kwargs: 额外的配置参数，会覆盖默认配置
        
    Returns:
        results: 训练结果字典
        save_dir: 保存目录路径
    """
    # 1. 加载配置
    try:
        config = get_direct_config(config_type)
        print(f"已加载 {config_type} 配置")
        
        # 如果有额外的配置参数，应用它们
        if kwargs:
            config = get_custom_config(config_type, **kwargs)
            print("已应用自定义配置参数")
        
        # 打印配置摘要
        print_config_summary(config, f"{config_type} 配置")
    except Exception as e:
        print(f"配置加载失败: {e}")
        # 使用默认配置
        config = {
            'training': {
                'lr': 0.001,
                'epochs': 5000,
                'optimizer': 'adam',
                'lr_scheduler': 'step',
                'lr_decay_rate': 0.95,
                'lr_decay_steps': 1000,
                'omega2_init': 1.0,
                'alpha': 10.0,
                'beta': 1e-4,
                'verbose': 1,
                'save_interval': 500,
                'early_stopping': True,
                'patience': 500,
                'min_delta': 1e-6
            },
            'data': {
                'N_f': 2000,
                'N_b': 100,
                'N_test': 200,
                'domain': [0, 1]
            },
            'equation': {
                'T': 600,
                'v': 50
            }
        }
        print("使用默认配置")
    
    # 2. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 3. 创建保存目录
    save_dir = create_save_directory()
    print(f"结果将保存到: {save_dir}")
    
    # 4. 保存配置
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # 5. 设置k值范围
    if k_values is None:
        k_values = [2, 3, 4, 5]
    
    # 6. 运行直接训练
    print("\n开始直接训练过程...")
    results = run_direct_training(k_values, config, device, save_dir)
    
    # 7. 可视化结果
    print("\n生成可视化结果...")
    visualize_results(results, save_dir)
    
    # 8. 打印总结
    print("\n训练完成!")
    print_results_summary(results)
    
    return results, save_dir

if __name__ == "__main__":
    import argparse
    
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='MIM-HomPINNs 直接训练程序')
    parser.add_argument('--config', type=str, default='balanced', 
                       choices=['balanced', 'quick', 'high_precision'],
                       help='配置类型 (默认: balanced)')
    parser.add_argument('--k-values', type=int, nargs='+', default=[2, 3, 4, 5],
                       help='要尝试的k值列表 (默认: [2, 3, 4, 5])')
    parser.add_argument('--lr', type=float, default=None,
                       help='学习率 (覆盖配置文件中的值)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数 (覆盖配置文件中的值)')
    parser.add_argument('--optimizer', type=str, default=None,
                       choices=['adam', 'sgd', 'rmsprop'],
                       help='优化器类型 (覆盖配置文件中的值)')
    parser.add_argument('--scheduler', type=str, default=None,
                       choices=['step', 'exponential', 'plateau', 'none'],
                       help='学习率调度器类型 (覆盖配置文件中的值)')
    parser.add_argument('--alpha', type=float, default=None,
                       help='边界惩罚系数 (覆盖配置文件中的值)')
    parser.add_argument('--beta', type=float, default=None,
                       help='非零解惩罚系数 (覆盖配置文件中的值)')
    parser.add_argument('--n-f', type=int, default=None,
                       help='配置点数量 (覆盖配置文件中的值)')
    parser.add_argument('--n-b', type=int, default=None,
                       help='边界点数量 (覆盖配置文件中的值)')
    
    args = parser.parse_args()
    
    # 准备自定义配置参数
    custom_params = {}
    
    if args.lr is not None:
        custom_params['training_lr'] = args.lr
    if args.epochs is not None:
        custom_params['training_epochs'] = args.epochs
    if args.optimizer is not None:
        custom_params['training_optimizer'] = args.optimizer
    if args.scheduler is not None:
        custom_params['training_lr_scheduler'] = args.scheduler
    if args.alpha is not None:
        custom_params['training_alpha'] = args.alpha
    if args.beta is not None:
        custom_params['training_beta'] = args.beta
    if args.n_f is not None:
        custom_params['data_N_f'] = args.n_f
    if args.n_b is not None:
        custom_params['data_N_b'] = args.n_b
    
    # 运行主函数
    results, save_dir = main(
        config_type=args.config,
        k_values=args.k_values,
        **custom_params
    )