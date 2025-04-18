#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-03-14 21:43:17
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-04-01 11:16:50
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import torch

from typing import Literal

from torch.nn import Embedding, ReLU, Dropout, Linear

from torch_geometric.nn import GATConv, DenseGATConv, SAGPooling, DMoNPooling
from torch_geometric.utils import to_dense_adj, to_dense_batch


class DenseAggregation(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.gate_layer = Linear(in_channels, 1)
        self.transform_layer = Linear(in_channels, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [ batch_size X node_number X in_channels ]

        gated_x = self.gate_layer(x).squeeze(-1)
        # [ batch_size X node_number ]

        gated_x = torch.nn.functional.softmax(gated_x, dim=-1)
        # [ batch_size X node_number ]

        transformed_x = self.transform_layer(x)
        # [ batch_size X node_number X out_channels ]

        output = torch.matmul(gated_x.unsqueeze(1), transformed_x).squeeze(1)
        # [ batch_size X out_channels ]
        return output


class NAPPGATVaryV1(torch.nn.Module):
    # Neural Architecture Performance Prediction - GNN - Base Model
    def __init__(
        self,
        node_dict: dict,
        task_dict: dict,
        node_dim: int = 100,
        task_dim: int = 100,
        hidden_dim: int = 100,
        readout_dim: int = 100,
        cluster_num: int = 16,
    ) -> None:
        super().__init__()

        # Embedding
        self.node_embedding_layer = Embedding(len(node_dict), node_dim)
        self.task_embedding_layer = Embedding(len(task_dict), task_dim)

        # v GNN Message Passing Head
        self.mp_head_layer = GATConv(node_dim, hidden_dim, heads=4, concat=False, dropout=0.1)
        self.mp_head_activate = ReLU(inplace=False)
        # ^ GNN Message Passing Head

        # v GNN Message Passing Body
        self.mp_body_layer_1 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=0.1)
        self.mp_body_activate_1 = ReLU(inplace=False)
        self.mp_body_dropout_1 = Dropout(p=0.3)

        self.mp_body_sag_pooling = SAGPooling(hidden_dim, ratio=4000)

        self.mp_body_layer_2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=0.1)
        self.mp_body_activate_2 = ReLU(inplace=False)
        self.mp_body_dropout_2 = Dropout(p=0.3)
        # ^ GNN Message Passing Body

        # v GNN Whole Coarsening
        self.global_pooling_layer = DMoNPooling(hidden_dim, cluster_num)
        # ^ GNN Whole Coarsening

        # v GNN Message Passing Tail
        self.mp_tail_layer = DenseGATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=0.1)
        self.mp_tail_activate = ReLU(inplace=False)
        self.mp_tail_dropout = Dropout(p=0.3)
        # ^ GNN Message Passing Tail

        self.readout_layer = DenseAggregation(hidden_dim, readout_dim)

        self.task_transform_layer = Linear(task_dim, hidden_dim)
        self.task_transform_activate = ReLU(inplace=False)

        self.fuse_layer = Linear(hidden_dim + readout_dim, hidden_dim)
        self.fuse_activate = ReLU(inplace=False)

        self.output_layer = Linear(hidden_dim, 1)

        self.initialize_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, task: torch.Tensor):
        # x
        # total_node_number = sum(node_number_of_graph_{1}, ..., node_number_of_graph_{batch_size})
        # [ total_node_number X num_node_features ] (Current Version: num_node_features=1)

        x = self.node_embedding_layer(x)
        # [ total_node_number X node_dim ]

        x = self.mp_head_layer(x, edge_index)
        x = self.mp_head_activate(x)
        # [ total_node_number X hidden_dim ]

        x = self.mp_body_layer_1(x, edge_index)
        x = self.mp_body_activate_1(x)
        x = self.mp_body_dropout_1(x)
        # [ total_node_number X hidden_dim ]

        x = self.mp_body_layer_2(x, edge_index)
        x = self.mp_body_activate_2(x)
        x = self.mp_body_dropout_2(x)
        # [ total_pooled_node_number X hidden_dim ]

        x, edge_index, _, batch, _, _ = self.mp_body_sag_pooling(x, edge_index, batch=batch)
        # [ total_pooled_node_number X hidden_dim ]

        (dense_x, mask) = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch)
        # max_pooled_node_number = max(pooled_node_number_of_graph_{1}, ..., pooled_node_number_of_graph_{batch_size})
        # [ batch_size X max_pooled_node_number X hidden_dim ]

        _, dense_x, adj, spectral_loss, orthogonality_loss, cluster_loss = self.global_pooling_layer(dense_x, adj, mask)
        # [ batch_size X global_pooled_node_number X hidden_dim ]

        global_pooling_loss = spectral_loss + orthogonality_loss + cluster_loss

        dense_x = self.mp_tail_layer(dense_x, adj)
        dense_x = self.mp_tail_activate(dense_x)
        dense_x = self.mp_tail_dropout(dense_x)
        # [ batch_size X global_pooled_node_number X hidden_dim ]

        dense_x = self.readout_layer(dense_x)
        # [ batch_size X readout_dim ]

        task = self.task_embedding_layer(task)
        task = self.task_transform_layer(task)
        task = self.task_transform_activate(task)

        fuse = torch.concat([dense_x, task], dim=-1)
        fuse = self.fuse_layer(fuse)
        fuse = self.fuse_activate(fuse)

        output = self.output_layer(fuse)
        # output - [ batch_size X 1 ]

        return output, global_pooling_loss

    def initialize_parameters(self):
        torch.nn.init.normal_(self.node_embedding_layer.weight, mean=0, std=self.node_embedding_layer.embedding_dim ** -0.5)
        torch.nn.init.normal_(self.task_embedding_layer.weight, mean=0, std=self.task_embedding_layer.embedding_dim ** -0.5)