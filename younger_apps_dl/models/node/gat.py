#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-03-14 21:43:17
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-01-12 17:08:03
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


from typing import Literal

from torch import nn, mean
from torch.nn import Embedding
from torch.nn import functional as F
from torch_geometric.nn import GATConv

from younger_apps_dl.models import register_model


@register_model('gat_node_classification')
class GATNodeClassification(nn.Module):
    def __init__(self, node_emb_size, node_emb_dim, hidden_dim, dropout_rate, stage: Literal['encode', 'classify', 'embedding'] = 'classify'):
        super(GATNodeClassification, self).__init__()
        self.stage = stage
        self.dropout_rate = dropout_rate
        self.node_embedding_layer = Embedding(node_emb_size, node_emb_dim)
        self.layer_1 = GATConv(node_emb_dim, hidden_dim, heads=8, dropout=dropout_rate, concat=False)
        self.layer_2 = GATConv(hidden_dim, node_emb_size, heads=8, dropout=dropout_rate, concat=False)
        self.initialize_parameters()

    def forward(self, x, edge_index, x_position = None):
        x = self.node_embedding_layer(x).squeeze(1)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.layer_1(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.layer_2(x, edge_index)
        if self.stage == 'encode':
            raise NotImplementedError('GATNodeClassification does not support "encode" stage yet.')
        if self.stage == 'classify':
            x = x if x_position is None else x[x_position]
        if self.stage == 'embedding':
            x = mean(x, dim=0).unsqueeze(0) if x_position is None else mean(x[x_position], dim=0).unsqueeze(0)
        return x

    def initialize_parameters(self):
        nn.init.normal_(self.node_embedding_layer.weight, mean=0, std=self.node_embedding_layer.embedding_dim ** -0.5)
