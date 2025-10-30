import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.optimize import root_scalar

# --------------------------
# 1. 定义微分方程和边界条件
# --------------------------
def fun(x, y, p):
    """定义一阶微分方程组（含参数ω²）
    y = [y0, y1, y2, y3] 其中：
    y0 = y, y1 = y', y2 = y'', y3 = y'''
    p = [omega_sq] 表示ω²（特征值参数）
    """
    omega_sq = p[0]
    y0, y1, y2, y3 = y  # 状态变量
    return [
        y1,                  # y0' = y1
        y2,                  # y1' = y2
        y3,                  # y2' = y3
        625*y2 + 50*y1 + omega_sq * y0  # y3' = 625y'' + 50y' + ω² y（原方程移项）
    ]

def bc(ya, yb, p):
    """定义边界条件：y(0)=y(1)=y''(0)=y''(1)=0
    ya = [y0(0), y1(0), y2(0), y3(0)]
    yb = [y0(1), y1(1), y2(1), y3(1)]
    p = [omega_sq] 参数
    """
    return [
        ya[0],    # y(0) = 0
        yb[0],    # y(1) = 0
        ya[2],    # y''(0) = 0
        yb[2],    # y''(1) = 0
        0         # 参数边界条件（通常设为0）
    ]

# --------------------------
# 2. 定义残差函数（用于搜索特征值）
# --------------------------
def residual(omega_sq):
    """对于给定的ω²，求解BVP并返回残差（判断是否存在非零解）"""
    # 离散化区间 [0,1]，取50个点
    x = np.linspace(0, 1, 50)
    # 初始猜测：使用更简单的多项式函数作为初始猜测
    y_guess = np.zeros((4, x.size))
    # 使用满足边界条件的多项式：y(x) = x*(1-x)，y(0)=y(1)=0
    y_guess[0] = x * (1 - x)          # y0猜测：x(1-x)
    y_guess[1] = 1 - 2*x              # y1猜测：y0' = 1-2x
    y_guess[2] = -2 * np.ones_like(x) # y2猜测：y0'' = -2
    y_guess[3] = np.zeros_like(x)     # y3猜测：y0''' = 0
    
    # 求解边值问题
    try:
        sol = solve_bvp(
            fun=fun,
            bc=bc,
            x=x,
            y=y_guess,
            p=[omega_sq],  # 传递参数ω²
            max_nodes=10000,  # 增加最大节点数，提高精度
            tol=1e-8         # 降低容差
        )
        
        # 残差定义：解的最大绝对值（非零解应使残差足够大，这里取负的便于根查找）
        if sol.success:
            max_solution = np.max(np.abs(sol.y[0]))
            print(f"ω²={omega_sq:.2f}, 成功求解, 最大解值={max_solution:.6f}")
            return -max_solution  # 负号：让非零解处残差为负
        else:
            print(f"ω²={omega_sq:.2f}, 求解失败")
            return 1e6  # 求解失败时返回大值
    except Exception as e:
        print(f"ω²={omega_sq:.2f}, 求解异常: {e}")
        return 1e6  # 异常时返回大值

# --------------------------
# 3. 搜索特征值（ω²）
# --------------------------
# 特征值可能存在的区间（对于四阶方程，特征值通常较小，这里测试1到1000）
# 注意：实际特征值可能需要调整搜索区间
interval = (1, 1000)

# 尝试使用直接有限差分法求解特征值问题
print("使用直接有限差分法求解特征值问题...")

# 导入直接求解方法
from scipy.sparse import diags
from scipy.linalg import eig

def solve_fourth_order_eigenvalue(N=200):
    """
    使用有限差分法求解四阶微分方程的特征值问题，精确处理边界条件
    
    Parameters:
    N: 网格点数
    
    Returns:
    eigenvalues: 特征值（ω²）
    eigenvectors: 特征函数
    """
    # 创建网格
    x = np.linspace(0, 1, N)
    h = x[1] - x[0]  # 网格间距
    
    # 我们需要处理4个边界条件：y(0)=0, y(1)=0, y''(0)=0, y''(1)=0
    # 内部点数量：N-4
    
    # 构建四阶导数矩阵（中心差分）
    # 使用五点模板：y'''' ≈ (y_{i-2} - 4y_{i-1} + 6y_i - 4y_{i+1} + y_{i+2}) / h^4
    
    # 主对角线：6/h^4
    main_diag = 6 * np.ones(N-4) / h**4
    
    # 次对角线：-4/h^4
    sub_diag1 = -4 * np.ones(N-5) / h**4
    sub_diag2 = 1 * np.ones(N-6) / h**4
    
    # 构建四阶导数矩阵（作用于内部点）
    D4 = diags([sub_diag2, sub_diag1, main_diag, sub_diag1, sub_diag2], 
               [-2, -1, 0, 1, 2], 
               shape=(N-4, N-4)).toarray()
    
    # 构建二阶导数矩阵（作用于内部点）
    # y'' ≈ (y_{i-1} - 2y_i + y_{i+1}) / h^2
    main_diag_2 = -2 * np.ones(N-2) / h**2
    sub_diag_2 = 1 * np.ones(N-3) / h**2
    
    D2 = diags([sub_diag_2, main_diag_2, sub_diag_2], 
               [-1, 0, 1], 
               shape=(N-2, N-2)).toarray()
    
    # 构建一阶导数矩阵（作用于内部点）
    # y' ≈ (y_{i+1} - y_{i-1}) / (2h)
    sub_diag_1_neg = -1 * np.ones(N-2) / (2*h)
    sub_diag_1_pos = 1 * np.ones(N-2) / (2*h)
    
    D1 = diags([sub_diag_1_neg, sub_diag_1_pos], 
               [-1, 1], 
               shape=(N-2, N-2)).toarray()
    
    # 扩展矩阵到内部点维度
    # D2和D1作用于N-2个点，D4作用于N-4个点
    # 我们需要将D2和D1扩展到与D4相同的维度
    D2_reduced = D2[1:-1, 1:-1]  # 去掉边界点
    D1_reduced = D1[1:-1, 1:-1]  # 去掉边界点
    
    # 构建完整的微分算子（作用于内部点）
    L = D4 - 625 * D2_reduced - 50 * D1_reduced
    
    # 质量矩阵（单位矩阵）
    M = np.eye(N-4)
    
    # 求解特征值问题：L * y = λ * y
    eigenvalues, eigenvectors = eig(L, M)
    
    # 排序特征值（按实部排序）
    idx = np.argsort(np.real(eigenvalues))
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 只保留实特征值（虚部很小）
    real_mask = np.abs(np.imag(eigenvalues)) < 1e-10
    eigenvalues = np.real(eigenvalues[real_mask])
    eigenvectors = np.real(eigenvectors[:, real_mask])
    
    # 构建完整的特征函数（包括边界点）
    full_eigenvectors = np.zeros((N, len(eigenvalues)))
    
    for i in range(len(eigenvalues)):
        # 内部点
        full_eigenvectors[2:-2, i] = eigenvectors[:, i]
        
        # 边界条件：y(0)=0, y(1)=0
        full_eigenvectors[0, i] = 0
        full_eigenvectors[-1, i] = 0
        
        # 边界条件：y''(0)=0, y''(1)=0
        # 使用更精确的边界条件处理方法
        # 对于左边界：y''(0) = 0，使用二阶导数的三点公式
        # y''(0) ≈ (2y_0 - 5y_1 + 4y_2 - y_3)/h^2 = 0
        # 由于y_0=0，我们有：-5y_1 + 4y_2 - y_3 = 0
        # 类似地处理右边界：y''(1) = 0
        
        # 左边界附近点：使用边界条件确定y_1
        # y_1 = (4y_2 - y_3)/5
        full_eigenvectors[1, i] = (4 * full_eigenvectors[2, i] - full_eigenvectors[3, i]) / 5
        
        # 右边界附近点：使用边界条件确定y_{N-2}
        # y_{N-2} = (4y_{N-3} - y_{N-4})/5
        full_eigenvectors[-2, i] = (4 * full_eigenvectors[-3, i] - full_eigenvectors[-4, i]) / 5
    
    # 归一化特征函数
    for i in range(full_eigenvectors.shape[1]):
        full_eigenvectors[:, i] = full_eigenvectors[:, i] / np.max(np.abs(full_eigenvectors[:, i]))
    
    return eigenvalues, full_eigenvectors, x

try:
    eigenvalues, eigenvectors, x_fine = solve_fourth_order_eigenvalue(100)
    
    # 取第一个特征值
    omega_sq = eigenvalues[0]
    omega = np.sqrt(omega_sq)
    
    print(f"找到第一个特征值：ω ≈ {omega:.4f}，ω² ≈ {omega_sq:.4f}")
    
    # 使用第一个特征函数作为解
    sol_final_y = eigenvectors[:, 0]
    
except Exception as e:
    print(f"直接求解失败: {e}")
    print("回退到原始方法...")
    
    # 用二分法查找第一个特征值（使残差为零的ω²）
    root_result = root_scalar(
        residual,
        method='bisect',
        bracket=interval,
        maxiter=1000,
        xtol=1e-6
    )
    
    if root_result.success:
        omega_sq = root_result.root
        omega = np.sqrt(omega_sq)
        print(f"找到特征值：ω ≈ {omega:.4f}，ω² ≈ {omega_sq:.4f}")
        
        # 求解对应特征值的非零特解
        x_fine = np.linspace(0, 1, 200)
        y_guess_fine = np.zeros((4, x_fine.size))
        y_guess_fine[0] = np.sin(np.pi * x_fine)  # 初始猜测

        sol_final = solve_bvp(
            fun=fun,
            bc=bc,
            x=x_fine,
            y=y_guess_fine,
            p=[omega_sq],
            max_nodes=10000
        )

        if not sol_final.success:
            raise RuntimeError("求解最终特解失败")
        
        sol_final_y = sol_final.y[0]
    else:
        raise ValueError("未找到特征值，请调整搜索区间")

# --------------------------
# 5. 绘图展示结果
# --------------------------
plt.figure(figsize=(10, 6))
plt.plot(x_fine, sol_final_y, 'b-', linewidth=2, label=f'特征函数 (ω≈{omega:.4f})')
plt.xlabel('x', fontsize=12)
plt.ylabel('y(x)', fontsize=12)
plt.title('四阶微分方程边值问题的非零特解', fontsize=14)
plt.grid(True)
plt.legend()
plt.show()