#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试训练结果保存功能
"""

import os
import sys
import torch
import json
import numpy as np
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.visualization import Visualizer

def test_directory_creation():
    """测试目录创建功能"""
    print("=" * 60)
    print("测试目录创建功能")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = f"test_results/run_{timestamp}"
    
    try:
        # 创建目录
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(os.path.join(test_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(test_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(test_dir, "logs"), exist_ok=True)
        
        print(f"✅ 测试目录创建成功: {test_dir}")
        
        # 测试文件写入权限
        test_file = os.path.join(test_dir, "test_permission.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("权限测试文件\n")
            f.write(f"创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print("✅ 文件写入权限测试通过")
        
        # 测试文件读取权限
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("✅ 文件读取权限测试通过")
        
        # 清理测试文件
        os.remove(test_file)
        print("✅ 文件删除权限测试通过")
        
        return test_dir
        
    except Exception as e:
        print(f"❌ 目录创建测试失败: {e}")
        return None

def test_model_saving(test_dir):
    """测试模型保存功能"""
    print("\n" + "=" * 60)
    print("测试模型保存功能")
    print("=" * 60)
    
    try:
        # 创建一个简单的模型用于测试
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        
        # 模拟训练数据
        omega2 = 3.1416
        k = 1
        loss_history = {
            'total_loss': [10.0, 5.0, 2.0, 1.0, 0.5],
            'F_loss': [8.0, 4.0, 1.5, 0.8, 0.4],
            'G_loss': [2.0, 1.0, 0.5, 0.2, 0.1],
            'omega2': [1.0, 2.0, 2.5, 3.0, 3.1416]
        }
        
        # 保存模型
        model_path = os.path.join(test_dir, "models", f"test_model_k{k}_omega2_{omega2:.4f}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'omega2': omega2,
            'k': k,
            'loss_history': loss_history
        }, model_path)
        
        print(f"✅ 模型保存成功: {model_path}")
        
        # 验证模型可以加载
        checkpoint = torch.load(model_path)
        loaded_model = SimpleModel()
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        
        print("✅ 模型加载验证通过")
        
        # 验证数据完整性
        assert abs(checkpoint['omega2'] - omega2) < 1e-6, "特征值不匹配"
        assert checkpoint['k'] == k, "参数k不匹配"
        assert len(checkpoint['loss_history']['total_loss']) == 5, "损失历史长度不匹配"
        
        print("✅ 数据完整性验证通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型保存测试失败: {e}")
        return False

def test_visualization_saving(test_dir):
    """测试可视化保存功能"""
    print("\n" + "=" * 60)
    print("测试可视化保存功能")
    print("=" * 60)
    
    try:
        # 创建可视化器
        visualizer = Visualizer(test_dir)
        
        # 测试单个解图像保存
        x_test = torch.linspace(0, 1, 100).view(-1, 1)
        
        # 创建一个简单的模型用于测试
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(1, 10)
                self.linear2 = torch.nn.Linear(10, 1)
            
            def forward(self, x):
                y1 = torch.sin(x)  # 模拟解
                y2 = torch.cos(x)  # 模拟一阶导数
                y3 = -torch.sin(x)  # 模拟二阶导数
                y4 = -torch.cos(x)  # 模拟三阶导数
                return y1, y2, y3, y4, None
        
        model = SimpleModel()
        
        # 测试单个解图像保存
        visualizer.plot_solution(model, x_test, 
                               title="测试解图像",
                               filename="test_solution.png")
        
        print("✅ 单个解图像保存成功")
        
        # 测试损失历史图像保存
        loss_history = {
            'total_loss': [10.0, 5.0, 2.0, 1.0, 0.5],
            'F_loss': [8.0, 4.0, 1.5, 0.8, 0.4],
            'G_loss': [2.0, 1.0, 0.5, 0.2, 0.1],
            'omega2': [1.0, 2.0, 2.5, 3.0, 3.1416]
        }
        
        visualizer.plot_loss_history(loss_history,
                                   title="测试损失历史",
                                   filename="test_loss_history.png")
        
        print("✅ 损失历史图像保存成功")
        
        # 测试特征值分布图像保存
        omega2_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        visualizer.plot_eigenvalue_distribution(omega2_values,
                                              title="测试特征值分布",
                                              filename="test_eigenvalue_distribution.png")
        
        print("✅ 特征值分布图像保存成功")
        
        # 验证文件确实存在
        expected_files = [
            "test_solution.png",
            "test_loss_history.png", 
            "test_eigenvalue_distribution.png"
        ]
        
        for filename in expected_files:
            file_path = os.path.join(test_dir, filename)
            if os.path.exists(file_path):
                print(f"✅ 文件存在: {filename}")
            else:
                print(f"❌ 文件不存在: {filename}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 可视化保存测试失败: {e}")
        return False

def test_json_saving(test_dir):
    """测试JSON文件保存功能"""
    print("\n" + "=" * 60)
    print("测试JSON文件保存功能")
    print("=" * 60)
    
    try:
        # 创建测试数据
        summary = {
            'total_solutions': 3,
            'omega2_values': [1.0, 2.0, 3.0],
            'ks': [1, 2, 3],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'test_data': {
                'array': [1, 2, 3],
                'nested': {'key': 'value'}
            }
        }
        
        # 保存JSON文件
        summary_path = os.path.join(test_dir, "test_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        
        print(f"✅ JSON文件保存成功: {summary_path}")
        
        # 验证JSON文件可以加载
        with open(summary_path, 'r', encoding='utf-8') as f:
            loaded_summary = json.load(f)
        
        print("✅ JSON文件加载验证通过")
        
        # 验证数据完整性
        assert loaded_summary['total_solutions'] == 3, "解数量不匹配"
        assert len(loaded_summary['omega2_values']) == 3, "特征值数量不匹配"
        assert 'timestamp' in loaded_summary, "时间戳缺失"
        
        print("✅ JSON数据完整性验证通过")
        
        return True
        
    except Exception as e:
        print(f"❌ JSON保存测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试训练结果保存功能...")
    
    # 运行所有测试
    test_dir = test_directory_creation()
    if not test_dir:
        print("❌ 目录创建测试失败，终止测试")
        return
    
    tests = [
        ("模型保存", test_model_saving, test_dir),
        ("可视化保存", test_visualization_saving, test_dir),
        ("JSON保存", test_json_saving, test_dir)
    ]
    
    results = []
    for test_name, test_func, arg in tests:
        try:
            result = test_func(arg)
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            results.append((test_name, False))
    
    # 输出测试结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！训练结果保存功能正常")
        print(f"测试文件保存在: {test_dir}")
    else:
        print("⚠️ 部分测试失败，请检查相关功能")
    print("=" * 60)

if __name__ == "__main__":
    main()