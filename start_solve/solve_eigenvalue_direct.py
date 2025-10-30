import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.sparse import diags

# 直接求解四阶微分方程的特征值问题
# y'''' = 625y'' + 50y' + ω²y
# 边界条件：y(0)=y(1)=y''(0)=y''(1)=0

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
        # 使用二阶导数的有限差分公式来设置边界附近的值
        # 对于左边界：y''(0) ≈ (2y_0 - 5y_1 + 4y_2 - y_3)/h^2 = 0
        # 由于y_0=0，我们有：-5y_1 + 4y_2 - y_3 = 0
        # 类似地处理右边界
        
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

# 求解特征值问题
print("求解四阶微分方程特征值问题...")
eigenvalues, eigenvectors, x = solve_fourth_order_eigenvalue(100)

# 显示前10个特征值
print("前10个特征值（ω²）:")
for i in range(min(10, len(eigenvalues))):
    print(f"λ{i+1} = {eigenvalues[i]:.6f}")

# 绘制前几个特征函数
plt.figure(figsize=(12, 8))
for i in range(min(4, len(eigenvalues))):
    plt.plot(x, eigenvectors[:, i], label=f'特征函数 {i+1} (ω²={eigenvalues[i]:.2f})')

plt.xlabel('x', fontsize=12)
plt.ylabel('y(x)', fontsize=12)
plt.title('四阶微分方程的特征函数', fontsize=14)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 验证边界条件
print("\n验证边界条件:")
for i in range(min(3, len(eigenvalues))):
    y = eigenvectors[:, i]
    print(f"特征函数 {i+1}:")
    print(f"  y(0) = {y[0]:.6f}")
    print(f"  y(1) = {y[-1]:.6f}")
    # 计算二阶导数在边界处的值（有限差分）
    h = x[1] - x[0]
    ypp_0 = (y[2] - 2*y[1] + y[0]) / h**2
    ypp_1 = (y[-1] - 2*y[-2] + y[-3]) / h**2
    print(f"  y''(0) ≈ {ypp_0:.6f}")
    print(f"  y''(1) ≈ {ypp_1:.6f}")
    print()