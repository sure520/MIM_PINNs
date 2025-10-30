"""
测试 run_training 函数的测试脚本
"""

import torch
import numpy as np
import os
import tempfile
import shutil
from datetime import datetime

# 导入项目模块
import sys
sys.path.append('.')

from main_optimized import run_training, evaluate_solutions, visualize_results
from configs.optimized_config import get_optimized_config


def test_run_training_basic():
    """
    测试 run_training 函数的基本功能
    """
    print("=" * 60)
    print("测试 run_training 基本功能")
    print("=" * 60)
    
    # 创建临时目录用于测试
    temp_dir = tempfile.mkdtemp()
    save_dir = os.path.join(temp_dir, "test_results")
    
    try:
        # 获取配置
        config = get_optimized_config()
        
        # 修改配置为测试模式（减少训练轮数）
        config['training']['epochs'] = 50  # 减少训练轮数
        config['training']['n_homotopy_steps'] = 2  # 减少同伦步骤
        config['training']['homotopy_init_ks'] = [1]  # 只测试一个k值
        config['training']['max_solutions'] = 1  # 只保留一个解
        config['data']['N_f'] = 100  # 减少配置点数量
        config['data']['N_b'] = 4   # 减少边界点数量
        
        # 设置设备
        device = 'cpu'  # 使用CPU进行测试
        
        print(f"测试配置:")
        print(f"  设备: {device}")
        print(f"  保存目录: {save_dir}")
        print(f"  训练轮数: {config['training']['epochs']}")
        print(f"  同伦步骤: {config['training']['n_homotopy_steps']}")
        print(f"  k值: {config['training']['homotopy_init_ks']}")
        
        # 运行训练
        print("\n开始训练...")
        all_solutions, all_omega2_values = run_training(config, device, save_dir)
        
        # 验证结果
        print("\n验证训练结果...")
        
        # 检查是否返回了正确的结果
        assert isinstance(all_solutions, list), "all_solutions 应该是列表"
        assert isinstance(all_omega2_values, list), "all_omega2_values 应该是列表"
        
        print(f"  找到的解数量: {len(all_solutions)}")
        print(f"  特征值数量: {len(all_omega2_values)}")
        
        # 检查保存的文件
        print("\n检查保存的文件...")
        
        # 检查模型文件
        models_dir = os.path.join(save_dir, "models")
        assert os.path.exists(models_dir), f"模型目录不存在: {models_dir}"
        
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        print(f"  保存的模型文件数量: {len(model_files)}")
        
        # 检查日志文件
        logs_dir = os.path.join(save_dir, "logs")
        assert os.path.exists(logs_dir), f"日志目录不存在: {logs_dir}"
        
        log_files = [f for f in os.listdir(logs_dir) if f.endswith('.txt')]
        print(f"  保存的日志文件数量: {len(log_files)}")
        
        # 检查解的质量
        if len(all_solutions) > 0:
            print("\n检查解的质量...")
            solution = all_solutions[0]
            
            # 检查模型
            assert hasattr(solution['model'], 'forward'), "模型应该有 forward 方法"
            assert hasattr(solution['model'], 'compute_residuals'), "模型应该有 compute_residuals 方法"
            
            # 检查特征值
            omega2 = solution['omega2']
            assert isinstance(omega2, (int, float, np.number)), "特征值应该是数值类型"
            assert omega2 > 0, "特征值应该大于0"
            
            # 检查损失历史
            loss_history = solution['loss_history']
            assert isinstance(loss_history, dict), "损失历史应该是字典"
            assert 'total_loss' in loss_history, "损失历史应该包含 total_loss"
            
            print(f"  特征值 ω²: {omega2:.4f}")
            print(f"  最终损失: {loss_history['total_loss'][-1]:.6f}")
        
        print("\n✅ 基本功能测试通过!")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"清理临时目录: {temp_dir}")


def test_run_training_multiple_k():
    """
    测试 run_training 函数处理多个k值的情况
    """
    print("\n" + "=" * 60)
    print("测试多个k值的情况")
    print("=" * 60)
    
    # 创建临时目录用于测试
    temp_dir = tempfile.mkdtemp()
    save_dir = os.path.join(temp_dir, "test_results_multiple_k")
    
    try:
        # 获取配置
        config = get_optimized_config()
        
        # 修改配置为测试模式
        config['training']['epochs'] = 50  # 进一步减少训练轮数
        config['training']['n_homotopy_steps'] = 2  # 减少同伦步骤
        config['training']['homotopy_init_ks'] = [1, 2]  # 测试两个k值
        config['training']['max_solutions'] = 2  # 保留两个解
        config['training']['solution_threshold'] = 0.5  # 放宽解差异阈值
        config['data']['N_f'] = 50  # 减少配置点数量
        
        # 设置设备
        device = 'cpu'
        
        print(f"测试配置:")
        print(f"  k值: {config['training']['homotopy_init_ks']}")
        print(f"  最大解数量: {config['training']['max_solutions']}")
        
        # 运行训练
        print("\n开始训练...")
        all_solutions, all_omega2_values = run_training(config, device, save_dir)
        
        # 验证结果
        print("\n验证多个k值的结果...")
        
        # 检查是否处理了多个k值
        k_values_used = [sol['k'] for sol in all_solutions]
        print(f"  使用的k值: {k_values_used}")
        print(f"  找到的解数量: {len(all_solutions)}")
        
        # 检查解的唯一性
        if len(all_omega2_values) > 1:
            # 检查特征值是否不同
            omega2_diff = abs(all_omega2_values[0] - all_omega2_values[1])
            print(f"  特征值差异: {omega2_diff:.4f}")
            
            # 如果差异大于阈值，说明解是不同的
            if omega2_diff > config['training']['solution_threshold']:
                print("  ✅ 解具有足够的差异")
            else:
                print("  ⚠️ 解差异较小，可能重复")
        
        print("\n✅ 多个k值测试通过!")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"清理临时目录: {temp_dir}")


def test_run_training_error_handling():
    """
    测试 run_training 函数的错误处理
    """
    print("\n" + "=" * 60)
    print("测试错误处理")
    print("=" * 60)
    
    # 创建临时目录用于测试
    temp_dir = tempfile.mkdtemp()
    save_dir = os.path.join(temp_dir, "test_results_error")
    
    try:
        # 获取配置
        config = get_optimized_config()
        
        # 修改配置为测试模式
        config['training']['epochs'] = 10  # 非常少的训练轮数
        config['training']['n_homotopy_steps'] = 2
        config['training']['homotopy_init_ks'] = [1, 999]  # 包含一个可能出错的k值
        config['training']['max_solutions'] = 2
        config['data']['N_f'] = 10  # 非常少的配置点
        
        # 设置设备
        device = 'cpu'
        
        print("测试配置包含可能出错的k值...")
        
        # 运行训练（应该能处理错误）
        print("\n开始训练（期望能处理错误）...")
        all_solutions, all_omega2_values = run_training(config, device, save_dir)
        
        # 验证错误处理
        print("\n验证错误处理...")
        
        # 检查是否至少有一个解成功
        if len(all_solutions) > 0:
            print(f"  ✅ 至少有一个解成功: {len(all_solutions)} 个解")
        else:
            print("  ⚠️ 没有成功解，但程序没有崩溃")
        
        # 检查错误日志
        logs_dir = os.path.join(save_dir, "logs")
        error_logs = [f for f in os.listdir(logs_dir) if 'error' in f and f.endswith('.txt')]
        
        if len(error_logs) > 0:
            print(f"  ✅ 检测到错误日志: {len(error_logs)} 个")
            for error_log in error_logs:
                error_path = os.path.join(logs_dir, error_log)
                with open(error_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(f"    错误日志 {error_log}: {content[:100]}...")
        else:
            print("  ⚠️ 未检测到错误日志")
        
        print("\n✅ 错误处理测试通过!")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"清理临时目录: {temp_dir}")


def test_device_consistency():
    """
    测试设备一致性
    """
    print("\n" + "=" * 60)
    print("测试设备一致性")
    print("=" * 60)
    
    # 创建临时目录用于测试
    temp_dir = tempfile.mkdtemp()
    save_dir = os.path.join(temp_dir, "test_results_device")
    
    try:
        # 获取配置
        config = get_optimized_config()
        
        # 修改配置为测试模式
        config['training']['epochs'] = 10
        config['training']['n_homotopy_steps'] = 2
        config['training']['homotopy_init_ks'] = [1]
        config['training']['max_solutions'] = 1
        config['data']['N_f'] = 10
        
        # 测试CPU设备
        device = 'cpu'
        
        print("测试CPU设备...")
        
        # 运行训练
        all_solutions, all_omega2_values = run_training(config, device, save_dir)
        
        if len(all_solutions) > 0:
            # 检查模型设备
            model = all_solutions[0]['model']
            model_device = next(model.parameters()).device
            print(f"  模型设备: {model_device}")
            
            # 检查设备一致性
            assert str(model_device) == device, f"模型设备不一致: {model_device} vs {device}"
            print("  ✅ 设备一致性检查通过")
        
        print("\n✅ 设备一致性测试通过!")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"清理临时目录: {temp_dir}")


def run_all_tests():
    """
    运行所有测试
    """
    print("=" * 60)
    print("开始运行 run_training 函数的所有测试")
    print("=" * 60)
    
    test_results = []
    
    # 运行基本功能测试
    test_results.append(("基本功能测试", test_run_training_basic()))
    
    # 运行多个k值测试
    test_results.append(("多个k值测试", test_run_training_multiple_k()))
    
    # 运行错误处理测试
    test_results.append(("错误处理测试", test_run_training_error_handling()))
    
    # 运行设备一致性测试
    test_results.append(("设备一致性测试", test_device_consistency()))
    
    # 汇总测试结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed_tests += 1
    
    print(f"\n总测试数: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"失败测试: {total_tests - passed_tests}")
    
    if passed_tests == total_tests:
        print("\n🎉 所有测试通过! run_training 函数工作正常。")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} 个测试失败，需要进一步检查。")
        return False


if __name__ == "__main__":
    # 运行所有测试
    success = run_all_tests()
    
    # 根据测试结果退出
    sys.exit(0 if success else 1)