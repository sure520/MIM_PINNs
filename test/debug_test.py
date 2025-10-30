import torch
import torch.nn as nn

# 测试基本的PyTorch功能
print("PyTorch版本:", torch.__version__)

# 测试创建简单的线性层
try:
    linear = nn.Linear(1, 10)
    print("✓ 线性层创建成功")
except Exception as e:
    print(f"✗ 线性层创建失败: {e}")

# 测试创建ResNet块
try:
    class ResidualBlock(nn.Module):
        def __init__(self, width):
            super(ResidualBlock, self).__init__()
            self.linear1 = nn.Linear(width, width)
            self.linear2 = nn.Linear(width, width)
            self.activation = nn.Tanh()

        def forward(self, x):
            residual = x
            out = self.activation(self.linear1(x))
            out = self.linear2(out)
            out += residual
            out = self.activation(out)
            return out
    
    block = ResidualBlock(10)
    print("✓ ResNet块创建成功")
except Exception as e:
    print(f"✗ ResNet块创建失败: {e}")

# 测试创建MIM1模型
try:
    class MIM1(nn.Module):
        def __init__(self, width=30, depth=2):
            super(MIM1, self).__init__()
            self.width = width
            self.depth = depth
            
            self.input_layer = nn.Linear(1, width)
            self.residual_blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(depth)])
            self.output_layer = nn.Linear(width, 5)
            self.activation = nn.Tanh()

        def forward(self, x):
            out = self.activation(self.input_layer(x))
            out = self.residual_blocks(out)
            out = self.output_layer(out)
            
            y1 = out[:, 0]
            y2 = out[:, 1]
            y3 = out[:, 2]
            y4 = out[:, 3]
            omega2 = out[:, 4]
            
            boundary_factor = x[:, 0] * (1 - x[:, 0])
            y1 = y1 * boundary_factor
            y3 = y3 * boundary_factor
                
            return y1, y2, y3, y4, omega2
    
    model = MIM1(width=10, depth=1)
    print("✓ MIM1模型创建成功")
except Exception as e:
    print(f"✗ MIM1模型创建失败: {e}")

# 测试前向传播
try:
    x = torch.linspace(0, 1, 10).reshape(-1, 1)
    x.requires_grad = True
    y1, y2, y3, y4, omega2 = model(x)
    print(f"✓ 前向传播成功: y1.shape={y1.shape}, omega2.shape={omega2.shape}")
except Exception as e:
    print(f"✗ 前向传播失败: {e}")

print("基本测试完成")