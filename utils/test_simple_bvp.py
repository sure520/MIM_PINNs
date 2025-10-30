import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

# 测试一个简单的二阶特征值问题：y'' + λy = 0, y(0)=y(1)=0
# 已知特征值：λ = (nπ)^2, n=1,2,3,...

def fun_simple(x, y, p):
    """二阶特征值问题：y'' + λy = 0"""
    lambda_val = p[0]
    y0, y1 = y
    return [y1, -lambda_val * y0]

def bc_simple(ya, yb, p):
    """边界条件：y(0)=0, y(1)=0"""
    return [ya[0], yb[0], 0]  # 2个状态边界条件 + 1个参数边界条件

def residual_simple(lambda_val):
    """残差函数"""
    x = np.linspace(0, 1, 50)
    y_guess = np.zeros((2, x.size))
    y_guess[0] = np.sin(np.pi * x)  # 初始猜测
    y_guess[1] = np.pi * np.cos(np.pi * x)
    
    sol = solve_bvp(fun_simple, bc_simple, x, y_guess, p=[lambda_val])
    
    if sol.success:
        max_val = np.max(np.abs(sol.y[0]))
        print(f"λ={lambda_val:.2f}, 最大解值={max_val:.6f}")
        return -max_val
    else:
        print(f"λ={lambda_val:.2f}, 求解失败")
        return 1e6

# 测试已知的特征值附近
print("测试简单二阶特征值问题:")
for lambda_val in [8, 9, 10, 11, 12]:
    residual_simple(lambda_val)

print("\n已知第一个特征值: λ = π² ≈", np.pi**2)

# 绘制第一个特征函数
x_fine = np.linspace(0, 1, 100)
y_guess_fine = np.zeros((2, x_fine.size))
y_guess_fine[0] = np.sin(np.pi * x_fine)
y_guess_fine[1] = np.pi * np.cos(np.pi * x_fine)

sol_final = solve_bvp(fun_simple, bc_simple, x_fine, y_guess_fine, p=[np.pi**2])

if sol_final.success:
    plt.figure(figsize=(10, 6))
    plt.plot(sol_final.x, sol_final.y[0], 'b-', linewidth=2, label=f'特征函数 (λ=π²)')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y(x)', fontsize=12)
    plt.title('二阶特征值问题的特征函数', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()
else:
    print("求解失败")