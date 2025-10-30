import torch
import numpy as np
import argparse
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime

# 导入自定义模块
from models.fusion_model import MIMHomPINNFusion
from utils.data_generator import DataGenerator
from utils.training import Trainer
from utils.evaluation import Evaluator
from utils.visualization import Visualizer
from configs.config import get_config


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='MIM与HomPINNs融合方法求解变系数四阶常微分方程多解')
    
    # 模型参数
    parser.add_argument('--width', type=int, default=30, help='网络宽度')
    parser.add_argument('--depth', type=int, default=2, help='ResNet块数量')
    parser.add_argument('--model_type', type=str, default='MIM1', choices=['MIM1', 'MIM2'], 
                        help='MIM网络架构类型')
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=0.002, help='初始学习率')
    parser.add_argument('--epochs', type=int, default=20000, help='每个同伦步骤的训练轮数')
    parser.add_argument('--n_homotopy_steps', type=int, default=11, help='同伦步骤数')
    parser.add_argument('--decay_rate', type=float, default=0.9, help='学习率衰减率')
    
    # 数据参数
    parser.add_argument('--N_g', type=int, default=100, help='内部采样点数')
    parser.add_argument('--N_b', type=int, default=4, help='边界采样点数')
    parser.add_argument('--N_test', type=int, default=1000, help='测试点数')
    
    # 物理参数
    parser.add_argument('--T', type=float, default=600, help='方程参数T')
    parser.add_argument('--v', type=float, default=50, help='方程参数v')
    parser.add_argument('--omega2_init', type=float, default=1.0, help='特征值初始猜测')
    
    # 同伦参数
    parser.add_argument('--alpha', type=float, default=10, help='边界惩罚系数')
    parser.add_argument('--homotopy_init_k', type=int, default=1, help='起始函数的k值')
    parser.add_argument('--max_solutions', type=int, default=10, help='最大求解数量')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='计算设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save_dir', type=str, default='results', help='结果保存目录')
    parser.add_argument('--verbose', type=int, default=1, help='日志打印频率')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)
    
    # 创建数据生成器
    config = get_config()
    data_gen = DataGenerator(config['data'])
    
    # 创建模型
    model = MIMHomPINNFusion(
        width=args.width,
        depth=args.depth,
        model_type=args.model_type,
        device=device
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        data_gen=data_gen,
        lr=args.lr,
        epochs=args.epochs,
        n_homotopy_steps=args.n_homotopy_steps,
        decay_rate=args.decay_rate,
        alpha=args.alpha,
        T=args.T,
        v=args.v,
        omega2_init=args.omega2_init,
        homotopy_init_k=args.homotopy_init_k,
        device=device,
        save_dir=save_dir,
        verbose=args.verbose
    )
    
    # 创建评估器
    evaluator = Evaluator(model, data_gen, device)
    
    # 创建可视化器
    visualizer = Visualizer(save_dir=os.path.join(save_dir, "figures"))
    
    # 存储所有找到的解
    all_solutions = []
    all_omega2 = []
    
    # 多解探索循环
    for k in range(args.homotopy_init_k, args.homotopy_init_k + args.max_solutions):
        print(f"\n=== 寻找第 {k - args.homotopy_init_k + 1} 个解 (起始函数k={k}) ===")
        
        # 训练模型
        model, omega2, loss_history = trainer.train(k=k)
        
        # 评估模型
        residual_error, boundary_error = evaluator.evaluate(model, omega2, args.T, args.v)
        
        # 检查解的有效性
        if residual_error < 5e-3 and boundary_error < 1e-3:
            # 检查是否为新解
            is_new_solution = True
            for prev_omega2 in all_omega2:
                if abs(omega2 - prev_omega2) / prev_omega2 < 0.05:  # 差异小于5%视为相同解
                    is_new_solution = False
                    break
            
            if is_new_solution:
                print(f"找到新解! 特征值: {omega2:.6f}, 残差误差: {residual_error:.6f}, 边界误差: {boundary_error:.6f}")
                
                # 保存解
                all_solutions.append(model.state_dict())
                all_omega2.append(omega2)
                
                # 保存模型
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'omega2': omega2,
                    'k': k,
                    'loss_history': loss_history,
                    'residual_error': residual_error,
                    'boundary_error': boundary_error
                }, os.path.join(save_dir, "models", f"solution_{k}.pth"))
                
                # 可视化结果
                visualizer.plot_solution(model, omega2, k, residual_error, boundary_error)
                visualizer.plot_loss_history(loss_history, k)
                
                # 如果找到足够多的解，提前结束
                if len(all_solutions) >= 5:
                    print(f"已找到 {len(all_solutions)} 个解，结束搜索。")
                    break
            else:
                print(f"解与已有解过于相似，跳过。")
        else:
            print(f"解不满足精度要求，跳过。残差误差: {residual_error:.6f}, 边界误差: {boundary_error:.6f}")
    
    # 汇总结果
    print("\n=== 求解结果汇总 ===")
    print(f"共找到 {len(all_solutions)} 个解:")
    for i, omega2 in enumerate(all_omega2):
        print(f"解 {i+1}: 特征值 ω² = {omega2:.6f}")
    
    # 可视化所有解
    if len(all_solutions) > 0:
        visualizer.plot_all_solutions(all_solutions, all_omega2, args.T, args.v)
    
    # 保存结果摘要
    summary = {
        'num_solutions': len(all_solutions),
        'omega2_values': all_omega2,
        'parameters': {
            'T': args.T,
            'v': args.v,
            'width': args.width,
            'depth': args.depth,
            'model_type': args.model_type,
            'lr': args.lr,
            'epochs': args.epochs,
            'n_homotopy_steps': args.n_homotopy_steps
        }
    }
    
    import json
    with open(os.path.join(save_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n结果已保存到: {save_dir}")


if __name__ == "__main__":
    main()