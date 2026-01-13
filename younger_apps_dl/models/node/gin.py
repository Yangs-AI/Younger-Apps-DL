#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-03-14 21:43:17
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-01-12 17:08:22
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


from typing import Literal

from torch import nn, mean
from torch.nn import Embedding
from torch.nn import functional as F
from torch_geometric.nn import GINConv

from younger_apps_dl.models import register_model


@register_model("gin_node_classification")
class GINNodeClassification(nn.Module):

    def __init__(self, node_emb_size, node_emb_dim, hidden_dim, dropout_rate, layer_number = 3, stage: Literal['encode', 'classify', 'embedding'] = 'classify'):
        super(GINNodeClassification, self).__init__()
        self.dropout_rate = dropout_rate
        self.stage = stage
        self.node_embedding_layer = Embedding(node_emb_size, node_emb_dim)

        self.layers = nn.ModuleList()
        self.layers.append(GIN_Conv(node_emb_dim, hidden_dim, hidden_dim))
        for i in range(layer_number):
            self.layers.append(GIN_Conv(hidden_dim, hidden_dim, hidden_dim))

        self.layers.append(GIN_Conv(hidden_dim, hidden_dim, node_emb_size))
        self.initialize_parameters()

    def forward(self, x, edge_index, x_position = None):
        x = self.node_embedding_layer(x).squeeze(1)
        for layer in self.layers:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x, edge_index)

        if self.stage == 'encode':
            raise NotImplementedError('GINNodeClassification does not support encoding stage currently.')
        if self.stage == 'classify':
            x = x if x_position is None else x[x_position]
        if self.stage == 'embedding':
            x = mean(x, dim=0).unsqueeze(0) if x_position is None else mean(x[x_position], dim=0).unsqueeze(0)
        return x

    def initialize_parameters(self):
        nn.init.normal_(self.node_embedding_layer.weight, mean=0, std=self.node_embedding_layer.embedding_dim ** -0.5)


class GIN_Conv(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, epsilon = 0):
        super(GIN_Conv, self).__init__()
        self.gnn = GINConv(nn.Sequential(nn.Identity()), eps=epsilon)
        self.lr1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.ac1 = nn.PReLU()
        self.lr2 = nn.Linear(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.ac2 = nn.PReLU()
        if input_dim == output_dim:
            self.res = nn.Identity()
        else:
            self.res = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x, edge_index):
        o = self.gnn(x, edge_index)
        o = self.lr1(o)
        o = self.bn1(o)
        o = self.ac1(o)
        o = self.lr2(o)
        o = self.bn2(o)
        o = self.ac2(o)
        o = o + self.res(x)
        return o
