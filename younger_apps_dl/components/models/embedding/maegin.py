#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-03-14 21:43:17
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-04-03 08:14:04
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import numpy

import torch.nn as nn
import torch.nn.functional as F

from pydantic import BaseModel, Field

from torch.nn import Embedding
from torch_geometric.nn import GINConv

from younger_apps_dl.components import BaseComponent, register_component


class MAEGINOptions(BaseModel):
    node_dict_size: int = Field(..., description='')
    node_dim: int = Field(..., description='')
    hidden_dim: int = Field(..., description='')
    dropout: float = Field(..., description='')
    layer_number: float = Field(3, description='')


@register_component('model', 'maegin')
class MAEGIN(BaseComponent[MAEGINOptions], nn.Module):
    _options_ = MAEGINOptions
    def __init__(self, configuration, node_dict_size, node_dim, hidden_dim, dropout, layer_number = 3):
        super(MAEGIN, self).__init__(configuration)
        self.encoder = MAEGINEncoder(node_dict_size, node_dim, hidden_dim, dropout, layer_number)
        self.project = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.decoder = MAEGINDecoder(node_dict_size, hidden_dim)

    def forward(self, x, edge_index):
        x = self.encoder(x, edge_index)
        x = self.project(x)
        x = self.decoder(x, edge_index)
        return x


class MAEGINEncoder(nn.Module):

    def __init__(self, node_dict_size, node_dim, hidden_dim, dropout, layer_number = 3):
        super(MAEGINEncoder, self).__init__()
        self.dropout = dropout

        self.node_embedding_layer = Embedding(node_dict_size, node_dim)

        self.layers = nn.ModuleList()

        dims = numpy.linspace(node_dim, hidden_dim, 1 + layer_number + 1 + 1, endpoint=True, dtype=int)

        self.layers.append(MAEGINConv(dims[0], dims[0], dims[1]))
        for i in range(1, 1 + layer_number):
            self.layers.append(MAEGINConv(dims[i], dims[i+1], dims[i+1]))

        self.layers.append(MAEGINConv(dims[1+layer_number], dims[1+layer_number], dims[1+layer_number+1]))

    def forward(self, x, edge_index):
        x = self.node_embedding_layer(x).squeeze(1)
        for layer in self.layers:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x, edge_index)
        return x


class MAEGINDecoder(nn.Module):

    def __init__(self, node_dict_size, hidden_dim):
        super(MAEGINDecoder, self).__init__()
        self.gnn = GINConv(nn.Sequential(nn.Identity()))
        middle_dim = hidden_dim + int((node_dict_size - hidden_dim) / 2)
        self.trn = nn.Linear(hidden_dim, middle_dim)
        self.prd = nn.Linear(middle_dim, node_dict_size)

    def forward(self, x, edge_index):
        x = self.gnn(x, edge_index)
        x = self.trn(x)
        x = self.prd(x)
        return x


class MAEGINConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MAEGINConv, self).__init__()
        self.gnn = GINConv(nn.Sequential(nn.Identity()))
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