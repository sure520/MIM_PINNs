#!/usr/bin/env python3
"""
运行测试脚本的辅助脚本
"""

import subprocess
import sys
import os

def run_test():
    """运行测试脚本"""
    try:
        # 切换到项目目录
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # 运行测试脚本
        result = subprocess.run([sys.executable, "test_direct_trainer.py"], 
                              capture_output=True, text=True)
        
        print("测试脚本输出:")
        print("=" * 50)
        print("STDOUT:")
        print(result.stdout)
        print("=" * 50)
        print("STDERR:")
        print(result.stderr)
        print("=" * 50)
        print(f"返回码: {result.returncode}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"运行测试时出错: {e}")
        return False

if __name__ == "__main__":
    success = run_test()
    if success:
        print("✅ 测试运行成功!")
    else:
        print("❌ 测试运行失败!")
    sys.exit(0 if success else 1)