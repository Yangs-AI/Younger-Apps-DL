#!/usr/bin/env python3
# -*- encoding=utf8 -*-

"""
简单的多层感知机模型示例
"""

import torch
import torch.nn as nn

# 从 younger_apps_dl 导入注册装饰器
from younger_apps_dl.models import register_model


@register_model('simple_mlp_example')
class SimpleMLP(nn.Module):
    """
    一个简单的多层感知机模型
    
    Args:
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
        output_dim: 输出维度
        dropout_rate: Dropout 比率
    """
    
    def __init__(
        self, 
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 10,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        print(f"[SimpleMLP] 初始化完成: {input_dim} -> {hidden_dim} -> {output_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, input_dim]
            
        Returns:
            输出张量 [batch_size, output_dim]
        """
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x


if __name__ == '__main__':
    # 测试模型
    print("测试 SimpleMLP 模型...")
    
    model = SimpleMLP(input_dim=10, hidden_dim=20, output_dim=5)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
    
    # 测试前向传播
    batch_size = 3
    x = torch.randn(batch_size, 10)
    print(f"输入形状: {x.shape}")
    
    output = model(x)
    print(f"输出形状: {output.shape}")
    
    assert output.shape == (batch_size, 5), "输出形状不正确"
    print("✓ 测试通过!")
