#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-03-14 21:43:17
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-01-15 23:57:14
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


from torch import nn
from torch.nn import Embedding
from torch.nn import functional as F
from torch_geometric.nn import resolver, MLP, GINConv, global_mean_pool

from younger_apps_dl.models import register_model


@register_model('gin_performance_prediction')
class GINPerformancePrediction(nn.Module):
    # Neural Architecture Performance Prediction - GIN
    def __init__(
        self,
        node_emb_size: int,
        node_emb_dim: int,
        hidden_dim: int,
        readout_dim: int,
        dropout_rate: float,
    ):
        super(GINPerformancePrediction, self).__init__()
        self.activation_layer = resolver.activation_resolver('ELU')

        self.dropout_rate = dropout_rate
        self.node_embedding_layer = Embedding(node_emb_size, node_emb_dim)

        self.gnn_head_mp_layer = GINConv(
            nn=MLP(
                channel_list=[node_emb_dim, hidden_dim, hidden_dim],
                act='ELU',
            ),
            eps=0,
            train_eps=False
        )

        self.gnn_mid_mp_layer = GINConv(
            nn=MLP(
                channel_list=[hidden_dim, hidden_dim, hidden_dim],
                act='ELU',
            ),
            eps=0,
            train_eps=False
        )

        self.gnn_tail_mp_layer = GINConv(
            nn=MLP(
                channel_list=[hidden_dim, hidden_dim, hidden_dim],
                act='ELU',
            ),
            eps=0,
            train_eps=False
        )

        self.gnn_readout_layer = MLP(
            channel_list=[hidden_dim, hidden_dim, readout_dim],
            act='ELU',
            norm=None,
            dropout=0.5
        )

        # Now Try Regression
        self.reg_output_layer = MLP(
            channel_list=[readout_dim, 1],
            act=None,
            norm=None,
        )
        self.initialize_parameters()

    def forward(self, x, edge_index, batch):
        # x - [ total_nodes X num_node_features ] (Current Version: num_node_features=1)
        main_feature = x[:, 0]
        x = self.node_embedding_layer(main_feature)
        # x - [ total_nodes X node_emb_dim ]

        x = self.gnn_head_mp_layer(x, edge_index)
        x = self.activation_layer(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # x - [ total_nodes X hidden_dim ]
        x = self.gnn_mid_mp_layer(x, edge_index)
        x = self.activation_layer(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # x - [ total_nodes X hidden_dim ]
        x = self.gnn_tail_mp_layer(x, edge_index)
        x = self.activation_layer(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # x - [ total_nodes X hidden_dim ]
        x = global_mean_pool(x, batch)
        # x - [ batch_size X hidden_dim ]

        x = self.gnn_readout_layer(x)
        # x - [ batch_size X readout_dim ]

        reg_output = self.reg_output_layer(x)
        # reg_output - [ batch_size X 1 ]

        return reg_output

    def initialize_parameters(self):
        nn.init.normal_(self.node_embedding_layer.weight, mean=0, std=self.node_embedding_layer.embedding_dim ** -0.5)
