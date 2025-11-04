"""
运行兼容性测试脚本
"""

import subprocess
import sys

def run_compatibility_test():
    """运行兼容性测试"""
    try:
        result = subprocess.run([
            sys.executable, 
            "test_direct_trainer_compatibility.py"
        ], capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        
        print("测试输出:")
        print(result.stdout)
        
        if result.stderr:
            print("错误信息:")
            print(result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"运行测试时出错: {e}")
        return False

if __name__ == "__main__":
    import os
    
    print("运行direct_trainer.py与MIMHomPINNFusion模型兼容性测试...")
    print("=" * 60)
    
    success = run_compatibility_test()
    
    if success:
        print("\n🎉 兼容性测试运行成功！")
    else:
        print("\n⚠️ 兼容性测试运行失败，请检查错误信息")
    
    sys.exit(0 if success else 1)