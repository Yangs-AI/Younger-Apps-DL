#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2026-02-01 20:31:04
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-02-02 02:21:25
# Copyright (c) 2026 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


"""
A simple Multi-Layer Perceptron (MLP) model example for Younger Apps DL.
"""

import torch
import torch.nn as nn

from younger_apps_dl.models import register_model


@register_model('simple_mlp_example')
class SimpleMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) model

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
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

        print(f"[SimpleMLP] Initialization complete: {input_dim} -> {hidden_dim} -> {output_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP model.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Output tensor [batch_size, output_dim]
        """
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    # Test the model
    print("Testing SimpleMLP model...")
    
    model = SimpleMLP(input_dim=10, hidden_dim=20, output_dim=5)
    print(f"Model parameters count: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass
    batch_size = 3
    x = torch.randn(batch_size, 10)
    print(f"Input shape: {x.shape}")
    
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (batch_size, 5), "Output shape is incorrect"
    print("✓ Test passed!")
