#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-01-14 10:00:57
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-02-23 16:41:29
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import os
import re
import torch
import numpy
import random
import pathlib

from typing import Any, Literal, Iterable


def shuffle_sequence(sequence: Iterable) -> Iterable:
    indices = list(range(len(sequence)))
    random.shuffle(indices)
    shuffled_sequence = ( sequence[index] for index in indices )
    return shuffled_sequence


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


def get_logging_metrics_str(metrics: dict[str, str]) -> str:
    metrics_str = ' '.join([f'[{metric_name}]={metric_value}' for metric_name, metric_value in metrics.items()])
    return metrics_str


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


def find_all_checkpoints(dirpath: pathlib.Path, basename: str = 'checkpoint', number: int = 1, metric: str | None = None) -> dict[int, pathlib.Path]:
    checkpoint_filename_pattern = re.compile(f'{basename}_Epoch_(?:\d+)_Step_(\d+)\.cp')
    checkpoints = dict()
    for path in dirpath.iterdir():
        if path.is_file():
            result = checkpoint_filename_pattern.fullmatch(path.name)
            if result is not None:
                position = int(result.group(1))
                checkpoints[position] = path
            else:
                continue
        else:
            continue

    return checkpoints


def load_checkpoint(load_path: pathlib.Path, basename: str = 'checkpoint', number: int = 1, metric: str | None = None) -> dict[str, Any] | None:
    checkpoint = None
    if load_path.is_file():
        checkpoint = torch.load(load_path, map_location=torch.device('cpu'))

    if load_path.is_dir():
        assert len(basename) != 0, f'Invalid checkpoint name.'
        checkpoints = find_all_checkpoints(load_path, basename, number, metric)

        if len(checkpoints) == 0:
            latest_checkpoint = None
        else:
            max_position = max(checkpoints.keys())
            latest_checkpoint_path = checkpoints[max_position]
            if latest_checkpoint_path.is_file():
                latest_checkpoint = torch.load(latest_checkpoint_path, map_location=torch.device('cpu'))
                assert max_position == latest_checkpoint['Step'], 'An Error occurred when loading checkpoint.'
            else:
                latest_checkpoint = None

        checkpoint = latest_checkpoint

    return checkpoint


def save_checkpoint(checkpoint, save_path: pathlib.Path, basename: str = 'checkpoint', number: int = 1, metric: str | None = None):
    if checkpoint_path.is_dir():
        assert len(checkpoint_name) != 0, f'Invalid checkpoint name.'
        position = checkpoint['Step']
        checkpoint_filename = f'{checkpoint_name}_Epoch_{checkpoint["Epoch"]}_Step_{checkpoint["Step"]}.cp'
        checkpoint_filepath = checkpoint_path.joinpath(checkpoint_filename)
        torch.save(checkpoint, checkpoint_filepath)

        checkpoints = find_all_checkpoints(checkpoint_path, checkpoint_name)
        positions = sorted(list(checkpoints.keys()), reverse=True)
        for position in positions[keep_number:]:
            remove_checkpoint(checkpoints[position])
    else:
        checkpoint_filepath = checkpoint_path
        torch.save(checkpoint, checkpoint_filepath)


def remove_checkpoint(checkpoint_path: pathlib.Path):
    if os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)
    else:
        raise IOError(f'Invalid address: {checkpoint_path}')


# def save_operator_embedding(save_dirpath: pathlib.Path, weights: NDArray, op_dict: dict[str, int]):
#     save_dirpath.mkdir(parents=True, exist_ok=False)
#     weights_filepath = save_dirpath.joinpath(f'weights.npy')
#     op_dict_filepath = save_dirpath.joinpath(f'op_dict.json')
#     numpy.save(weights_filepath, weights)
#     save_json(op_dict, op_dict_filepath, indent=2)


# def load_operator_embedding(load_dirpath: pathlib.Path):
#     weights_filepath = load_dirpath.joinpath(f'weights.npy')
#     op_dict_filepath = load_dirpath.joinpath(f'op_dict.json')
#     weights = numpy.load(weights_filepath)
#     op_dict = load_json(op_dict_filepath)
#     return weights, op_dict