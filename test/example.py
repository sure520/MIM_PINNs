#!/usr/bin/env python3
"""
示例脚本：使用MIM-HomPINNs融合方法求解变系数四阶常微分方程多解问题

这个脚本展示了如何使用项目中的各个模块来求解变系数四阶常微分方程的多解问题。
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import get_config, update_config
from utils.data_generator import DataGenerator
from models.fusion_model import MIMHomPINNFusion
from utils.training import Trainer
from utils.evaluation import Evaluator
from utils.visualization import Visualizer


def main():
    """主函数"""
    print("="*50)
    print("MIM-HomPINNs融合方法示例")
    print("="*50)
    
    # 1. 获取配置
    print("\n1. 加载配置...")
    config = get_config()
    
    # 修改配置用于示例
    config['model']['type'] = 'MIM1'  # 使用单网络架构
    config['data']['n_collocation'] = 2000  # 减少配置点数量以加快速度
    config['training']['epochs'] = 1000  # 减少训练轮数以加快速度
    config['training']['n_homotopy_steps'] = 3  # 减少同伦步骤数
    config['equation']['T'] = 600  # 方程参数T
    config['equation']['v'] = 50  # 方程参数v
    config['training']['max_solutions'] = 2  # 寻找2个解
    
    # 2. 创建数据生成器
    print("\n2. 创建数据生成器...")
    data_gen = DataGenerator(config['data'])
    
    # 3. 创建模型
    print("\n3. 创建模型...")
    model = MIMHomPINNFusion(config['model'])
    
    # 4. 创建训练器
    print("\n4. 创建训练器...")
    trainer = Trainer(model, data_gen, config['training'])
    
    # 5. 创建评估器
    print("\n5. 创建评估器...")
    evaluator = Evaluator(model, data_gen)
    
    # 6. 创建可视化器
    print("\n6. 创建可视化器...")
    save_dir = os.path.join('results', 'example')
    os.makedirs(save_dir, exist_ok=True)
    visualizer = Visualizer(save_dir)
    
    # 7. 多解探索
    print("\n7. 开始多解探索...")
    solutions = []
    omega2_list = []
    ks = []
    loss_histories = []
    
    for solution_idx in range(config['training']['max_solutions']):
        print(f"\n正在寻找第 {solution_idx+1} 个解...")
        
        # 7.1 初始化模型
        trainer.initialize_weights()
        
        # 7.2 同伦训练
        print("  开始同伦训练...")
        model, omega2, loss_history = trainer.homotopy_training(
            T=config['equation']['T'],
            v=config['equation']['v'],
            k_start=solution_idx + 1  # 从k=1,2,...开始
        )
        
        # 7.3 评估解的有效性
        print("  评估解的有效性...")
        residual_error, boundary_error = evaluator.evaluate(
            model, omega2, config['equation']['T'], config['equation']['v']
        )
        
        print(f"  残差误差: {residual_error:.6f}")
        print(f"  边界误差: {boundary_error:.6f}")
        
        # 7.4 检查解是否有效
        if residual_error > 1e-3 or boundary_error > 1e-3:
            print("  解无效，跳过...")
            continue
        
        # 7.5 计算解的范数
        solution_norm = evaluator.compute_solution_norm(model)
        print(f"  解的范数: {solution_norm:.6f}")
        
        # 7.6 检查是否为新解
        is_new_solution = True
        for prev_model, prev_omega2 in solutions:
            distance = evaluator.compare_solutions(model, prev_model)
            if distance < 1e-3:
                is_new_solution = False
                print("  解与之前找到的解过于相似，跳过...")
                break
        
        if not is_new_solution:
            continue
        
        # 7.7 保存解
        solutions.append((model, omega2))
        omega2_list.append(omega2)
        ks.append(solution_idx + 1)
        loss_histories.append(loss_history)
        
        print(f"  找到有效解! ω² = {omega2:.6f}")
    
    # 8. 结果可视化
    print("\n8. 结果可视化...")
    
    if solutions:
        # 8.1 绘制所有解
        visualizer.plot_solutions(
            solutions, omega2_list, ks, 
            T=config['equation']['T'], v=config['equation']['v'],
            save_name='all_solutions.png'
        )
        
        # 8.2 绘制损失曲线
        visualizer.plot_loss_curves(
            loss_histories, omega2_list, ks,
            save_name='loss_curves.png'
        )
        
        # 8.3 绘制残差分布
        for i, (model, omega2) in enumerate(solutions):
            visualizer.plot_residuals(
                model, omega2, 
                T=config['equation']['T'], v=config['equation']['v'],
                save_name=f'residuals_{i+1}.png'
            )
        
        # 8.4 与解析解比较
        for i, (model, omega2) in enumerate(solutions):
            visualizer.plot_comparison_with_analytical(
                model, omega2, ks[i],
                T=config['equation']['T'], v=config['equation']['v'],
                save_name=f'comparison_{i+1}.png'
            )
        
        # 8.5 绘制特征值收敛情况
        visualizer.plot_omega2_convergence(
            omega2_list, ks,
            save_name='omega2_convergence.png'
        )
        
        print(f"可视化结果已保存到 {save_dir} 目录")
    
    # 9. 结果总结
    print("\n9. 结果总结...")
    print(f"找到 {len(solutions)} 个有效解:")
    for i, (model, omega2) in enumerate(solutions):
        residual_error, boundary_error = evaluator.evaluate(
            model, omega2, config['equation']['T'], config['equation']['v']
        )
        print(f"  解 {i+1}: ω² = {omega2:.6f}, 残差误差 = {residual_error:.6f}, 边界误差 = {boundary_error:.6f}")
    
    print("\n示例完成!")


if __name__ == "__main__":
    main()