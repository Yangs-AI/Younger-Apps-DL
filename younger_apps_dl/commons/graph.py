#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2026-01-13 10:11:12
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-01-13 15:55:45
# Copyright (c) 2026 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import torch

from typing import Literal

from torch_geometric.utils import negative_sampling


def dag_edge_masking(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    mask_ratio: float,
    sample_ratio: float,
    device_descriptor: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mask edges in a DAG for self-supervised learning (some times used for link prediction).
    Only mantains a portion of edges for prior knowledge ( (1-mask_ratio) * num_edge ). The rest edges are masked out. Greater mask_ratio means less prior knowledge. Recommended range: 0.3~0.5 (prior knowledge is 0.5~0.7).
    Generates a certain ratio of positive and negative samples for the masked edges ( 2* sample_ratio * mask_ratio * num_edge ).
    Positive sampling is used to create positive edges among existing masked out edges.
    Negative sampling is used to create negative edges among non-existing edges.
    Thus:
      - the `x` returned is the same as input `x`.
      - the `edge_index` returned partially contains original edges.
      - the `edge_position` returned contains positive and negative sampled edges.

    So the term ``masking`` here refers to generating positive and negative samples for edges and reducing the original edge set.

    Args:
        x: Node feature tensor. Shape [num_nodes, num_features].
        edge_index: Edge index tensor representing graph connectivity. Shape [2, num_edge].
        mask_ratio: Ratio of edges to mask (0.0 to 1.0). It applies to the number of edges. Thus the returned edge_index contains (1 - mask_ratio) * num_edge edges.
        sample_ratio: Ratio of positive and negative samples to generate for the masked edges (0.0 to 1.0).
        device_descriptor: Device to perform computations on.
    Returns:
        x: Node feature tensor. The same as input x. Shape [num_nodes, num_features].
        edge_index: Edge index tensor representing graph connectivity. Different from input edge_index. Shape [2, (1 - mask_ratio) * num_edge ].
        edge_position: Edge positions tensor containing positive and negative sampled edges. Shape [2, 2 * sample_ratio * mask_ratio * num_edge ].
        edge_position_label: Edge labels for edge_position tensor indicating positive (1) and negative (0) samples. Shape [2 * sample_ratio * mask_ratio * num_edge ].
    """
    assert 0.0 <= mask_ratio <= 1.0, "mask_ratio must be between 0.0 and 1.0"
    assert 0.0 <= sample_ratio <= 1.0, "sample_ratio must be between 0.0 and 1.0"
    # ensure tensors are on the requested device
    x = x.clone().to(device_descriptor)
    edge_index = edge_index.clone().to(device_descriptor)

    num_edge = edge_index.size(1)
    num_keep = min(int(round((1.0 - mask_ratio) * num_edge)), num_edge) # number of edges to keep as prior knowledge
    num_mask = num_edge - num_keep
    num_smpl = min(int(round(sample_ratio * mask_ratio * num_edge)), num_mask) # number of positive and negative samples

    perm = torch.randperm(num_edge, device=device_descriptor)
    keep_idx = perm[:num_keep]
    mask_idx = perm[num_keep:]

    keep_edge_index = edge_index[:, keep_idx]
    mask_edge_index = edge_index[:, mask_idx]

    # Positive samples: sample from masked (original) edges
    # Negative samples: sample non-existing edges using torch_geometric utility
    if num_smpl > 0:
        pos_perm = torch.randperm(num_mask, device=device_descriptor)[:num_smpl]
        pos_edge = mask_edge_index[:, pos_perm]
        neg_edge = negative_sampling(
            edge_index=edge_index, num_nodes=len(x),
            num_neg_samples=num_smpl, method='sparse'
        ).to(device_descriptor)
        edge_position = torch.cat([pos_edge, neg_edge], dim=1)

        pos_edge_label = torch.ones(pos_edge.size(1), dtype=torch.long, device=device_descriptor)
        neg_edge_label = torch.zeros(neg_edge.size(1), dtype=torch.long, device=device_descriptor)
        edge_position_label = torch.cat([pos_edge_label, neg_edge_label], dim=0)
    else:
        edge_position = torch.empty((2, 0), dtype=torch.long, device=device_descriptor)
        edge_position_label = torch.empty((0,), dtype=torch.long, device=device_descriptor)

    return x, keep_edge_index, edge_position, edge_position_label


def dag_node_masking(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    x_dict: dict[str, int],
    mask_ratio: float,
    mask_method: Literal['Random', 'Purpose'],
    device_descriptor: torch.device,
    mode: Literal['BERT', 'PURE'] = 'BERT',
    ignore_index: int = -1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mask node features for self-supervised learning (BERT-style pre-training).

    Supports two masking strategies:
    - 'Random': Mask any node with uniform probability
    - 'Purpose': Only mask leaf nodes (nodes without successors in the DAG)

    Supports two modes:
    - 'BERT': Training mode uses BERT-style masking
        - 80% of masked nodes → [MASK] token
        - 10% of masked nodes → random token
        - 10% of masked nodes → keep original (helps model learn representations)
    - 'PURE': All masked nodes replaced with [MASK] token

    Args:
        x: Node feature tensor
        edge_index: Edge index tensor representing graph connectivity
        x_dict: Dictionary mapping token strings to indices (must contain '__MASK__')
        mask_ratio: Base ratio of nodes to mask (0.0 to 1.0)
        mask_method: 'Random' for uniform masking, 'Purpose' for leaf-only masking
        device_descriptor: Device to perform computations on
        mode: 'BERT' for BERT-style masking, 'PURE' for pure [MASK] replacement
        ignore_index: Label for non-masked nodes (-1 by default)
        
    Returns:
        Tuple of (masked_x, edge_index, label) where:
        - masked_x: Node features with masking applied
        - edge_index: Original edge indices (unchanged)
        - label: Ground truth labels (-1 for non-masked nodes)
    """
    # Clone to avoid modifying original tensors
    x = x.clone().to(device_descriptor)
    edge_index = edge_index.clone().to(device_descriptor)
    golden = x.clone()

    # === Step 1: Determine mask probability based on mask_method ===
    if mask_method == 'Purpose':
        # Strategy: Only mask leaf nodes (nodes without successors)
        source_index = edge_index[0]
        target_index = edge_index[1]

        # Find nodes that appear as targets but not as sources (leaf nodes)
        leaf_nodes = target_index[~torch.isin(target_index, source_index)]

        # Adjust mask_ratio to maintain expected number of masked nodes
        # If there are N total nodes and we want to mask mask_ratio * N nodes,
        # but only M leaf nodes exist, we need to mask (mask_ratio * N / M) of the leaf nodes
        unique_leaf_nodes = torch.unique(leaf_nodes)
        if unique_leaf_nodes.shape[0] > 0:
            adjusted_ratio = mask_ratio * len(x) / len(unique_leaf_nodes)
            adjusted_ratio = min(adjusted_ratio, 1.0)  # Cap at 1.0 to avoid invalid probability
        else:
            adjusted_ratio = 0.0  # No leaf nodes to mask

        # Create mask probability: only leaf nodes have non-zero probability
        mask_probability = torch.zeros_like(x, dtype=torch.float, device=device_descriptor)
        mask_probability[unique_leaf_nodes] = adjusted_ratio

    if mask_method == 'Random':
        # Strategy: Mask any node with uniform probability
        mask_probability = torch.full(x.shape, mask_ratio, dtype=torch.float, device=device_descriptor)

    # === Step 2: Sample which nodes to mask ===
    mask_indices = torch.bernoulli(mask_probability).bool()
    golden[~mask_indices] = ignore_index # Non-masked nodes have label -1 (ignored in loss)

    # === Step 3: Apply masking strategy ===
    if mode == 'PURE':
        # PURE mode: Replace all masked nodes with [MASK] token
        x[mask_indices] = x_dict['__MASK__']

    if mode == 'BERT':
        # BERT mode: BERT-style masking for better generalization
        # 80% of masked nodes → [MASK] token
        mask_with_mask = torch.bernoulli(
            torch.full(x.shape, 0.8, dtype=torch.float, device=device_descriptor)
        ).bool() & mask_indices
        x[mask_with_mask] = x_dict['__MASK__']

        # 10% of masked nodes → random token from vocabulary
        # (50% of the remaining 20% after [MASK] replacement)
        mask_with_optr = torch.bernoulli(
            torch.full(x.shape, 0.5, dtype=torch.float, device=device_descriptor)
        ).bool() & mask_indices & ~mask_with_mask
        x[mask_with_optr] = torch.randint(
            2, len(x_dict), x.shape, dtype=torch.long, device=device_descriptor
        )[mask_with_optr]

        # Remaining 10% keep original token (implicit, no operation needed)

    return x, edge_index, golden

