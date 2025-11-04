"""
MIM-HomPINNs 分阶直接训练主程序
使用HierarchicalDirectTrainer类进行"无同伦、分阶聚焦"训练，依次求解各阶特征值
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
from data.data_generator import DataGenerator
from utils.direct_trainer import HierarchicalDirectTrainer, create_hierarchical_trainer
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

def run_hierarchical_training(k_max, config, device, save_dir):
    """
    使用HierarchicalDirectTrainer运行分阶训练
    
    Args:
        k_max: 要训练的最大特征值阶数
        config: 训练配置
        device: 计算设备
        save_dir: 保存目录
        
    Returns:
        results: 训练结果字典
    """
    results = {}
    omega_low_2 = None  # 初始无低阶特征值
    
    print(f"开始分阶训练，最大阶数: {k_max}")
    print(f"使用设备: {device}")
    print(f"结果保存目录: {save_dir}")
    
    # 依次训练各阶特征值
    for k in range(1, k_max + 1):
        print(f"\n正在训练第 {k} 阶特征值...")
        
        try:
            # 创建数据生成器
            data_gen = DataGenerator(
                domain=config['data']['domain'],
                n_domain=config['data']['N_f'],
                n_boundary=config['data']['N_b'],
                n_test=config['data']['N_test']
            )
            
            # 创建模型
            if k == 1:
                # 第一阶训练，创建新模型
                model = MIMHomPINNFusion(device=device)
            else:
                # 高阶训练，加载前一阶的模型参数
                model = MIMHomPINNFusion(device=device)
                model_path = os.path.join(save_dir, "models", f"model_k{k-1}.pth")
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"已加载第 {k-1} 阶模型参数")
                else:
                    print(f"警告: 未找到第 {k-1} 阶模型参数，使用随机初始化")
            
            # 创建分阶训练器
            trainer = create_hierarchical_trainer(
                model=model,
                data_gen=data_gen,
                k=k,
                omega_low_2=omega_low_2,
                config=config,
                device=device,
                save_dir=os.path.join(save_dir, f'{k}_results')
            )
            
            # 开始训练
            start_time = time.time()
            trainer.train()
            training_time = time.time() - start_time
            
            # 评估模型
            eval_results = trainer.evaluate()
            
            # 保存当前阶特征值，作为下一阶的低阶特征值
            omega_low_2 = eval_results['omega2']
            
            # 保存结果
            results[k] = {
                'training_time': training_time,
                'eval_results': eval_results,
                'history': trainer.history,
                'omega2': eval_results['omega2']
            }
            
            print(f"第 {k} 阶特征值训练完成，用时: {training_time:.2f}秒")
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
                f.write(f"第 {k} 阶特征值训练日志\n")
                f.write(f"训练时间: {training_time:.2f}秒\n")
                f.write(f"最终损失: {eval_results['final_loss']:.6f}\n")
                f.write(f"特征值: {eval_results['omega2']:.6f}\n")
                f.write(f"PDE误差: {eval_results['pde_error']:.6f}\n")
                f.write(f"边界误差: {eval_results['bc_error']:.6f}\n")
            
            # 生成特征函数图像
            plot_eigenfunction(model, eval_results['omega2'], k, save_dir)
            
        except Exception as e:
            print(f"第 {k} 阶特征值训练失败: {str(e)}")
            results[k] = {'error': str(e)}
            # 如果当前阶训练失败，停止后续训练
            break
    
    # 保存所有结果
    results_save_path = os.path.join(save_dir, "hierarchical_training_results.json")
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

def plot_eigenfunction(model, omega2, k, save_dir):
    """
    绘制特征函数图像
    
    Args:
        model: 训练好的模型
        omega2: 特征值
        k: 特征值阶数
        save_dir: 保存目录
    """
    model.eval()
    
    # 生成测试点
    x_test = np.linspace(0, 1, 1000)
    x_tensor = torch.tensor(x_test, dtype=torch.float32, device=next(model.parameters()).device).unsqueeze(1)
    
    with torch.no_grad():
        # 获取模型输出
        y_pred = model(x_tensor)[:, 0].cpu().numpy()
    
    # 绘制特征函数
    plt.figure(figsize=(10, 6))
    plt.plot(x_test, y_pred, label=f'第 {k} 阶特征函数')
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title(f'第 {k} 阶特征函数 (ω² = {omega2:.4f})')
    plt.grid(True)
    plt.legend()
    
    # 保存图像
    plt.savefig(os.path.join(save_dir, "figures", f"eigenfunction_k{k}.png"), dpi=300)
    plt.close()

def visualize_hierarchical_results(results, save_dir):
    """
    可视化分阶训练结果
    
    Args:
        results: 训练结果字典
        save_dir: 保存目录
    """
    # 创建图像保存目录
    figures_dir = os.path.join(save_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # 提取成功的训练结果
    successful_ks = []
    omega2_values = []
    training_times = []
    final_losses = []
    
    for k, result in results.items():
        if 'error' not in result:
            successful_ks.append(k)
            omega2_values.append(result['omega2'])
            training_times.append(result['training_time'])
            final_losses.append(result['eval_results']['final_loss'])
    
    if not successful_ks:
        print("没有成功的训练结果，无法生成可视化图像")
        return
    
    # 1. 特征值随阶数变化的图像
    plt.figure(figsize=(10, 6))
    plt.plot(successful_ks, omega2_values, 'o-', linewidth=2, markersize=8)
    plt.xlabel('特征值阶数 k')
    plt.ylabel('特征值 ω²')
    plt.title('特征值随阶数变化')
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, "eigenvalues_vs_k.png"), dpi=300)
    plt.close()
    
    # 2. 训练时间随阶数变化的图像
    plt.figure(figsize=(10, 6))
    plt.plot(successful_ks, training_times, 's-', color='orange', linewidth=2, markersize=8)
    plt.xlabel('特征值阶数 k')
    plt.ylabel('训练时间 (秒)')
    plt.title('训练时间随阶数变化')
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, "training_time_vs_k.png"), dpi=300)
    plt.close()
    
    # 3. 最终损失随阶数变化的图像
    plt.figure(figsize=(10, 6))
    plt.plot(successful_ks, final_losses, '^-', color='green', linewidth=2, markersize=8)
    plt.xlabel('特征值阶数 k')
    plt.ylabel('最终损失')
    plt.title('最终损失随阶数变化')
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, "final_loss_vs_k.png"), dpi=300)
    plt.close()
    
    # 4. 特征值增长率的图像
    if len(omega2_values) > 1:
        growth_rates = []
        for i in range(1, len(omega2_values)):
            growth_rate = (omega2_values[i] - omega2_values[i-1]) / omega2_values[i-1]
            growth_rates.append(growth_rate)
        
        plt.figure(figsize=(10, 6))
        plt.plot(successful_ks[1:], growth_rates, 'd-', color='purple', linewidth=2, markersize=8)
        plt.xlabel('特征值阶数 k')
        plt.ylabel('特征值增长率')
        plt.title('特征值增长率随阶数变化')
        plt.grid(True)
        plt.savefig(os.path.join(figures_dir, "eigenvalue_growth_rate.png"), dpi=300)
        plt.close()
    
    print(f"可视化结果已保存到 {figures_dir} 目录")

def print_hierarchical_results_summary(results):
    """
    打印分阶训练结果摘要
    
    Args:
        results: 训练结果字典
    """
    print("\n" + "="*60)
    print("分阶训练结果摘要")
    print("="*60)
    
    successful_ks = []
    failed_ks = []
    
    for k, result in results.items():
        if 'error' in result:
            failed_ks.append(k)
        else:
            successful_ks.append(k)
    
    print(f"成功训练的特征值阶数: {successful_ks}")
    print(f"失败的特征值阶数: {failed_ks}")
    
    if successful_ks:
        print("\n各阶特征值详情:")
        print("-"*60)
        print(f"{'阶数':<6} {'特征值 ω²':<15} {'最终损失':<12} {'训练时间(s)':<12}")
        print("-"*60)
        
        for k in successful_ks:
            result = results[k]
            omega2 = result['omega2']
            final_loss = result['eval_results']['final_loss']
            training_time = result['training_time']
            print(f"{k:<6} {omega2:<15.6f} {final_loss:<12.6f} {training_time:<12.2f}")
        
        # 计算特征值增长率
        if len(successful_ks) > 1:
            print("\n特征值增长率:")
            print("-"*60)
            print(f"{'阶数':<6} {'特征值':<15} {'增长率':<12}")
            print("-"*60)
            
            prev_omega2 = None
            for k in successful_ks:
                result = results[k]
                omega2 = result['omega2']
                if prev_omega2 is not None:
                    growth_rate = (omega2 - prev_omega2) / prev_omega2 * 100
                    print(f"{k:<6} {omega2:<15.6f} {growth_rate:<12.2f}%")
                else:
                    print(f"{k:<6} {omega2:<15.6f} {'基准':<12}")
                prev_omega2 = omega2
    
    if failed_ks:
        print("\n失败原因:")
        print("-"*60)
        for k in failed_ks:
            print(f"阶数 {k}: {results[k]['error']}")
    
    print("="*60)

def main(config_type='balanced', k_max=3, **kwargs):
    """
    主函数
    
    Args:
        config_type: 配置类型 ('balanced', 'quick', 'high_precision')
        k_max: 要训练的最大特征值阶数
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
        raise RuntimeError(f"配置加载失败: {e}")
    
    # 2. 设置设备
    device = torch.device('cuda')
    print(f"使用设备: {device}")
    
    # 3. 创建保存目录
    save_dir = create_save_directory()
    print(f"结果将保存到: {save_dir}")
    
    # 4. 保存配置
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # 5. 运行分阶训练
    print("\n开始分阶训练过程...")
    results = run_hierarchical_training(k_max, config, device, save_dir)
    
    # 6. 可视化结果
    print("\n生成可视化结果...")
    visualize_hierarchical_results(results, save_dir)
    
    # 7. 打印总结
    print("\n训练完成!")
    print_hierarchical_results_summary(results)
    
    return results, save_dir

if __name__ == "__main__":
    import argparse
    
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='MIM-HomPINNs 分阶直接训练程序')
    parser.add_argument('--config', type=str, default='balanced', 
                       choices=['balanced', 'quick', 'high_precision'],
                       help='配置类型 (默认: balanced)')
    parser.add_argument('--k-max', type=int, default=3,
                       help='要训练的最大特征值阶数 (默认: 3)')
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
        k_max=args.k_max,
        **custom_params
    )