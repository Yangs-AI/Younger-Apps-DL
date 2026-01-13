#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-03-14 21:43:17
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-01-12 16:17:12
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


from typing import Literal

from torch import nn, mean, arange
from torch.nn import Embedding
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, GAE

from younger_apps_dl.models import register_model
from younger_apps_dl.commons.utils import gradient_computation_switch


class GAENodeClassificationEncoder(nn.Module):
    """
    Encoder for Graph Autoencoder - wrappable by torch_geometric.nn.GAE.
    """

    def __init__(self, node_emb_size, node_emb_dim, hidden_dim):
        super(GAENodeClassificationEncoder, self).__init__()
        self.node_embedding_layer = Embedding(node_emb_size, node_emb_dim)
        self.layer_1 = GCNConv(node_emb_dim, 2 * hidden_dim)
        self.layer_2 = GCNConv(2 * hidden_dim, hidden_dim)
        self.initialize_parameters()

    def forward(self, x, edge_index):
        x = self.node_embedding_layer(x).squeeze(1)
        x = F.relu(self.layer_1(x, edge_index))
        x = self.layer_2(x, edge_index)
        return x

    def initialize_parameters(self):
        nn.init.normal_(self.node_embedding_layer.weight, mean=0, std=self.node_embedding_layer.embedding_dim ** -0.5)


class GAENodeClassificationClassifier(nn.Module):
    """
    Linear Classifier for GAE Node Classification.
    """

    def __init__(self, hidden_dim, node_emb_size):
        super(GAENodeClassificationClassifier, self).__init__()
        self.layer_classify = nn.Linear(hidden_dim, node_emb_size)

    def forward(self, x, x_position = None, embedding = False):
        """
        x: Node embeddings from GAE encoder.
        x_position: Positions of nodes to classify. If None, classify all nodes.
        embedding: If True, return mean embedding of nodes at x_position; else return logits for nodes at x_position.
        """
        x = self.layer_classify(x)
        if embedding:
            x = mean(x, dim=0).unsqueeze(0) if x_position is None else mean(x[x_position], dim=0).unsqueeze(0)
        else:
            x = x if x_position is None else x[x_position]
        return x


@register_model('gae_node_classification')
class GAENodeClassification(nn.Module):
    """
    Unified GAE model with encoder and classifier.

    Supports multiple stages via `stage` parameters:
    1. Encoding stage: stage='encode' - only creates encoder, and forward method returns raw embeddings z
    2. Classification stage: stage='classify' - creates encoder + classifier, and forward method returns log-softmax scores at specified positions
    3. Embedding stage: stage='embedding' - creates encoder + classifier, and forward method returns mean embedding at specified positions
    """

    def __init__(self, node_emb_size: int, node_emb_dim: int, hidden_dim: int, stage: Literal['encode', 'classify', 'embedding'] = 'classify'):
        super(GAENodeClassification, self).__init__()
        self.stage = stage
        if stage == 'encode':
            self.encoder = GAE(GAENodeClassificationEncoder(node_emb_size, node_emb_dim, hidden_dim))
        if stage == 'classify':
            self.encoder = GAE(GAENodeClassificationEncoder(node_emb_size, node_emb_dim, hidden_dim))
            gradient_computation_switch(self.encoder, False)
            self.encoder.eval()
            self.classifier = GAENodeClassificationClassifier(hidden_dim, node_emb_size)
        if stage == 'embedding':
            self.encoder = GAE(GAENodeClassificationEncoder(node_emb_size, node_emb_dim, hidden_dim))
            self.classifier = GAENodeClassificationClassifier(hidden_dim, node_emb_size)

    def forward(self, x, edge_index, x_position = None):
        """
        Args:
            x: Node input [N, 1]
            edge_index: Graph connectivity [2, E]
            x_position: Indices of nodes to classify (required for 'classify' and 'embedding' stages)

        Returns:
            z if stage='encode'
            log_probs if stage='classify'
            embedding if stage='embedding'
        """

        if self.stage == 'encode':
            z = self.encoder.encode(x, edge_index)
            return z
        if self.stage == 'classify':
            z = self.encoder.encode(x, edge_index)
            # Detach to prevent gradient flow to encoder
            output = self.classifier(z.detach(), x_position, embedding=False)
            return output
        if self.stage == 'embedding':
            z = self.encoder.encode(x, edge_index)
            output = self.classifier(z, x_position, embedding=True)
            return output
