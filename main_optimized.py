"""
MIM-HomPINNs融合项目 - 优化配置主程序
使用预定义的优化超参数配置，无需命令行输入
"""

import os
import torch
import numpy as np
from datetime import datetime

# 导入项目模块
from configs.optimized_config import get_optimized_config
from utils.data_generator import DataGenerator
from models.fusion_model import MIMHomPINNFusion
from utils.training import Trainer
from utils.evaluation import Evaluator
from utils.visualization import Visualizer


def setup_environment():
    """设置运行环境"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 获取配置
    config = get_optimized_config()
    
    # 设置设备
    device = config['device']['device']
    if device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    print(f"使用设备: {device}")
    print(f"随机种子: 42")
    
    return config, device


def create_save_directory():
    """创建保存目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"optimized_results/run_{timestamp}"
    
    # 确保路径使用正斜杠，避免Windows路径问题
    save_dir = save_dir.replace("\\", "/")
    
    try:
        # 创建主目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建子目录
        os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "logs"), exist_ok=True)
        
        print(f"✅ 保存目录创建成功: {save_dir}")
        
        # 验证目录权限
        test_file = os.path.join(save_dir, "test_permission.txt")
        with open(test_file, 'w') as f:
            f.write("Permission test")
        os.remove(test_file)
        print("✅ 目录权限验证通过")
        
        return save_dir
    except Exception as e:
        print(f"❌ 创建保存目录失败: {e}")
        # 尝试使用当前目录作为备用
        save_dir = f"./optimized_results/run_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        print(f"⚠️ 使用备用目录: {save_dir}")
        return save_dir


def run_training(config, device, save_dir):
    """运行训练过程"""
    print("\n" + "="*60)
    print("开始MIM-HomPINNs融合模型训练")
    print("="*60)
    
    try:
        # 创建必要的目录结构
        import os
        models_dir = os.path.join(save_dir, "models")
        logs_dir = os.path.join(save_dir, "logs")
        
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        
        # 创建数据生成器
        data_gen = DataGenerator(config['data'])
        
        # 创建模型
        model = MIMHomPINNFusion(
            width=config['model']['resnet_neurons'],
            depth=config['model']['resnet_layers'],
            model_type=config['model']['model_type'],
            device=device
        )
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            data_gen=data_gen,
            lr=config['training']['lr'],
            epochs=config['training']['epochs'],
            n_homotopy_steps=config['training']['n_homotopy_steps'],
            decay_rate=config['training']['decay_rate'],
            alpha=config['training']['alpha'],
            T=config['equation']['T'],
            v=config['equation']['v'],
            omega2_init=config['training']['omega2_init'],
            homotopy_init_k=config['training']['homotopy_init_ks'][0],  # 使用第一个k值
            device=device,
            save_dir=save_dir,
            verbose=config['training']['verbose']
        )
        
        # 存储所有解
        all_solutions = []
        all_omega2_values = []
        
        # 对每个起始函数k值进行训练
        for k in config['training']['homotopy_init_ks']:
            print(f"\n开始训练起始函数 k={k}")
            
            try:
                # 训练模型
                trained_model, omega2, loss_history = trainer.train(k=k)
                
                # 保存模型
                model_path = f"{models_dir}/model_k{k}_omega2_{omega2:.4f}.pth"
                model_path = model_path.replace("\\", "/")
                
                torch.save({
                    'model_state_dict': trained_model.state_dict(),
                    'omega2': omega2,
                    'k': k,
                    'loss_history': loss_history
                }, model_path)
                
                # 保存训练日志
                log_path = f"{logs_dir}/training_log_k{k}.txt"
                log_path = log_path.replace("\\", "/")
                with open(log_path, 'w', encoding='utf-8') as f:
                    f.write(f"训练完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"特征值 ω²: {omega2:.6f}\n")
                    f.write(f"参数 k: {k}\n")
                    f.write(f"最终损失: {loss_history['total_loss'][-1]:.6f}\n")
                    f.write(f"模型保存路径: {model_path}\n")
                    for t in range(len(loss_history['total_loss'])):
                        f.write(f"第{t}同伦步骤: \n总损失={loss_history['total_loss'][t]:.6f}, \nF损失={loss_history['F_loss'][t]:.6f}, \nG损失={loss_history['G_loss'][t]:.6f}, \nR_b损失={loss_history['R_b_loss'][t]:.6f}\n")
                
                print(f"✅ 模型已保存到: {model_path}")
                print(f"✅ 训练日志已保存到: {log_path}")
                
                # 检查解是否重复
                is_new_solution = True
                for existing_omega2 in all_omega2_values:
                    if abs(omega2 - existing_omega2) < config['training']['solution_threshold']:
                        is_new_solution = False
                        print(f"  解 ω²={omega2:.4f} 与已有解相似，跳过保存")
                        break
                
                if is_new_solution:
                    all_solutions.append({
                        'model': trained_model,
                        'omega2': omega2,
                        'k': k,
                        'loss_history': loss_history
                    })
                    all_omega2_values.append(omega2)
                    print(f"  发现新解 ω²={omega2:.4f}")
                
                # 检查是否达到最大解数量
                if len(all_solutions) >= config['training']['max_solutions']:
                    print(f"已达到最大解数量 {config['training']['max_solutions']}")
                    break
                    
            except Exception as e:
                print(f"❌ 训练起始函数 k={k} 时发生错误: {e}")
                # 保存错误日志
                error_log_path = f"{save_dir}/logs/training_error_k{k}.txt"
                error_log_path = error_log_path.replace("\\", "/")
                with open(error_log_path, 'w', encoding='utf-8') as f:
                    f.write(f"错误时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"错误信息: {str(e)}\n")
                    f.write(f"错误类型: {type(e).__name__}\n")
                print(f"❌ 错误日志已保存到: {error_log_path}")
                continue
        
        return all_solutions, all_omega2_values
    
    except Exception as e:
        print(f"❌ 训练过程中发生错误: {e}")
        # 保存错误日志
        error_log_path = f"{save_dir}/logs/training_error.txt"
        error_log_path = error_log_path.replace("\\", "/")
        with open(error_log_path, 'w', encoding='utf-8') as f:
            f.write(f"错误时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"错误信息: {str(e)}\n")
            f.write(f"错误类型: {type(e).__name__}\n")
        
        print(f"❌ 错误日志已保存到: {error_log_path}")
        raise


def evaluate_solutions(config, all_solutions, save_dir):
    """评估所有解"""
    print("\n" + "="*60)
    print("开始解评估")
    print("="*60)
    
    # 创建数据生成器
    data_gen = DataGenerator(config['data'])
    
    # 创建评估器
    evaluator = Evaluator(None, data_gen)  # 模型将在循环中传入
    
    evaluation_results = []
    
    for i, solution in enumerate(all_solutions):
        print(f"\n评估解 {i+1}: ω²={solution['omega2']:.4f}")
        
        # 计算残差误差和边界误差
        residual_error, boundary_error = evaluator.evaluate(
            solution['model'], 
            solution['omega2'], 
            T=config['equation']['T'], 
            v=config['equation']['v']
        )
        
        # 计算解范数
        solution_norm = evaluator.compute_solution_norm(solution['model'])
        
        # 计算导数范数
        derivative_norms = evaluator.compute_derivative_norms(solution['model'])
        
        result = {
            'index': i,
            'omega2': solution['omega2'],
            'k': solution['k'],
            'residual_error': residual_error,
            'boundary_error': boundary_error,
            'solution_norm': solution_norm,
            'derivative_norms': derivative_norms
        }
        
        evaluation_results.append(result)
        
        print(f"  残差误差: {residual_error:.6e}")
        print(f"  边界误差: {boundary_error:.6e}")
        print(f"  解范数: {solution_norm:.6f}")
        print(f"  导数范数: {[f'{n:.6f}' for n in derivative_norms]}")
    
    return evaluation_results


def visualize_results(config, all_solutions, evaluation_results, save_dir):
    """可视化结果"""
    print("\n" + "="*60)
    print("开始结果可视化")
    print("="*60)
    
    try:
        # 创建可视化器
        visualizer = Visualizer(save_dir)
        
        # 提取模型和损失历史
        models = [sol['model'] for sol in all_solutions]
        loss_histories = [sol['loss_history'] for sol in all_solutions]
        ks = [sol['k'] for sol in all_solutions]
        omega2_values = [sol['omega2'] for sol in all_solutions]
        
        # 生成测试点
        data_gen = DataGenerator(config['data'])
        x_test = data_gen.generate_test_points(config['data']['N_test'])
        
        # 绘制多解图像
        visualizer.plot_solutions(models, omega2_values, ks, x_test)
        print("✅ 多解图像已生成")
        
        # 绘制损失曲线
        visualizer.plot_loss_curves(loss_histories, ks)
        print("✅ 损失曲线已生成")
        
        # 可视化所有解
        for i, solution in enumerate(all_solutions):
            print(f"生成解 {i+1} 的可视化")
            
            # 绘制解函数
            visualizer.plot_solution(
                solution['model'], 
                x_test, 
                title=f"解 {i+1} (ω²={solution['omega2']:.4f})",
                filename=f"solution_{i+1}.png"
            )
            
            # 绘制损失历史
            visualizer.plot_loss_history(
                solution['loss_history'],
                title=f"解 {i+1} 训练损失历史",
                filename=f"loss_history_{i+1}.png"
            )
        
        # 绘制特征值分布
        visualizer.plot_eigenvalue_distribution(
            omega2_values,
            title="发现的特征值分布",
            filename="eigenvalue_distribution.png"
        )
        print("✅ 特征值分布图已生成")
        
        # 生成总结报告
        summary = {
            'num_solutions': len(all_solutions),
            'omega2_values': omega2_values,
            'ks': ks,
            'parameters': {
                'T': config['equation']['T'],
                'v': config['equation']['v'],
                'width': config['model']['resnet_neurons'],
                'depth': config['model']['resnet_layers'],
                'model_type': config['model']['model_type'],
                'lr': config['training']['lr'],
                'epochs': config['training']['epochs'],
                'n_homotopy_steps': config['training']['n_homotopy_steps']
            }
        }
        
        # 保存总结
        import json
        summary_path = f"{save_dir}/summary.json"
        summary_path = summary_path.replace("\\", "/")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        
        print(f"✅ 结果摘要已保存到: {summary_path}")
        
        # 保存可视化日志
        log_path = f"{save_dir}/logs/visualization_log.txt"
        log_path = log_path.replace("\\", "/")
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"可视化完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总解数量: {len(all_solutions)}\n")
            f.write(f"特征值列表: {omega2_values}\n")
            f.write(f"参数k列表: {ks}\n")
            f.write(f"摘要文件路径: {summary_path}\n")
        
        print(f"✅ 可视化日志已保存到: {log_path}")
        print(f"✅ 结果已保存到: {save_dir}")
    
    except Exception as e:
        print(f"❌ 可视化过程中发生错误: {e}")
        # 保存错误日志
        error_log_path = f"{save_dir}/logs/visualization_error.txt"
        error_log_path = error_log_path.replace("\\", "/")
        with open(error_log_path, 'w', encoding='utf-8') as f:
            f.write(f"错误时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"错误信息: {str(e)}\n")
            f.write(f"错误类型: {type(e).__name__}\n")
        
        print(f"❌ 错误日志已保存到: {error_log_path}")
        raise


def main():
    """主函数"""
    try:
        # 1. 设置环境
        config, device = setup_environment()
        
        # 2. 创建保存目录
        save_dir = create_save_directory()
        
        # 3. 运行训练
        all_solutions, all_omega2_values = run_training(config, device, save_dir)
        
        # 4. 评估解
        evaluation_results = evaluate_solutions(config, all_solutions, save_dir)
        
        # 5. 可视化结果
        visualize_results(config, all_solutions, evaluation_results, save_dir)
        
        # 6. 输出最终结果
        print("\n" + "="*60)
        print("训练完成！")
        print("="*60)
        print(f"发现 {len(all_solutions)} 个不同的解:")
        for i, omega2 in enumerate(all_omega2_values):
            print(f"  解 {i+1}: ω² = {omega2:.4f}")
        print(f"\n结果已保存到: {save_dir}")
        
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()