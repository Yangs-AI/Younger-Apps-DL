#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-01-14 10:00:57
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-01-13 10:12:38
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import torch
import numpy
import random

from typing import Any, Literal, Iterable


def make_reproducible(seed: int = 3407, mode: bool = True):
    assert 0 < seed, 'Seed must > 0 .'

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(mode)


def get_model_parameters_number(model: torch.nn.Module) -> int:
    parameters_number = dict()
    for name, parameters in model.named_parameters():
        root_name = name.split('.')[0]
        if root_name in parameters_number:
            parameters_number[root_name] += parameters.numel()
        else:
            parameters_number[root_name] = parameters.numel()

    return parameters_number


def get_device_descriptor(device: Literal['CPU', 'GPU'], index: int) -> torch.device:
    if device == 'CPU':
        device_name = 'cpu'

    if device == 'GPU':
        device_name = f'cuda:{index}'

    return torch.device(device_name)


def average_model_state_dict(model_state_dicts: list[dict], weights: list[float] | None = None) -> dict:
    if weights is None:
        weights = [1.0 / len(model_state_dicts)] * len(model_state_dicts)
    else:
        assert len(model_state_dicts) == len(weights)

    final_state_dict = dict()
    for model_state_dict, weight in zip(model_state_dicts, weights):
        for key, value in model_state_dict.items():
            if key in final_state_dict:
                final_state_dict[key] += value * weight
            else:
                final_state_dict[key] = value * weight
    return final_state_dict


def broadcast_object(value: Any, src: int = 0) -> bool:
    # Broadcast a picklable object to all processes in the distributed group.
    # Only the source process (src) provides the object, others receive it.
    object_list = [value] if torch.distributed.get_rank() == src else [None]
    torch.distributed.broadcast_object_list(object_list, src=src)
    return object_list[0]


def gradient_computation_switch(module: torch.nn.Module, switch: bool):
    """
    Utility method to set requires_grad for all parameters in a module.
    """
    for parameter in module.parameters():
        parameter.requires_grad = switch
