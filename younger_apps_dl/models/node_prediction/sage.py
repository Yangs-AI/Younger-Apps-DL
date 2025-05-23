#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-03-14 21:43:17
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-04-01 11:14:21
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import torch
from torch import nn
from torch.nn import Embedding
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv


class SAGE(nn.Module):

    def __init__(self, node_dict_size, node_dim, hidden_dim, dropout, output_embedding = False):
        super(SAGE_NP, self).__init__()
        self.output_embedding = output_embedding
        self.node_embedding_layer = Embedding(node_dict_size, node_dim)
        self.dropout = dropout
        self.layer_1 = SAGEConv(node_dim, hidden_dim)
        self.layer_2 = SAGEConv(hidden_dim, node_dict_size)
        self.initialize_parameters()

    def forward(self, x, edge_index, mask_x_position):
        x = self.node_embedding_layer(x).squeeze(1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer_1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layer_2(x, edge_index)
        if self.output_embedding:
            return torch.mean(x, dim=0).unsqueeze(0)
        return F.log_softmax(x[mask_x_position], dim=1)

    def initialize_parameters(self):
        nn.init.normal_(self.node_embedding_layer.weight, mean=0, std=self.node_embedding_layer.embedding_dim ** -0.5)