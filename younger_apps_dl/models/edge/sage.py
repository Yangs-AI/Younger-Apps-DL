#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-03-14 21:43:17
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-01-13 12:33:14
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


from torch import nn
from torch.nn import Embedding
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv

from younger_apps_dl.models import register_model


@register_model('sage_edge_classification')
class SAGEEdgeClassification(nn.Module):
    def __init__(self, node_emb_size, node_emb_dim, hidden_dim, output_dim, dropout_rate):
        super(SAGEEdgeClassification, self).__init__()
        self.dropout_rate = dropout_rate
        self.node_embedding_layer = Embedding(node_emb_size, node_emb_dim)
        self.layer_1 = SAGEConv(node_emb_dim, hidden_dim)
        self.layer_2 = SAGEConv(hidden_dim, output_dim)
        self.initialize_parameters()

    def encode(self, x, edge_index):
        x = self.node_embedding_layer(x).squeeze(1)
        x = F.relu(self.layer_1(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.layer_2(x, edge_index)
        return x

    def decode(self, z, edge_position):
        src = z[edge_position[0]]
        dst = z[edge_position[1]]
        r = (src * dst).sum(dim=-1)
        return r

    def forward(self, x, edge_index, edge_position):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_position)

    def initialize_parameters(self):
        nn.init.normal_(self.node_embedding_layer.weight, mean=0, std=self.node_embedding_layer.embedding_dim ** -0.5)
