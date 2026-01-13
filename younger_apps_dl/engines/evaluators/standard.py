#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-10-16 22:58:32
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-01-02 11:31:18
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import time
import torch
import pathlib

from typing import Any, Literal, Callable
from pydantic import BaseModel, Field

from younger_apps_dl.commons.utils import get_device_descriptor, no_operation
from younger_apps_dl.commons.logging import logger
from younger_apps_dl.commons.checkpoint import load_checkpoint

from younger_apps_dl.engines import BaseEngine, register_engine


class StandardEvaluatorOptions(BaseModel):
    # Checkpoint Options
    checkpoint_filepath: pathlib.Path  = Field(..., description='Path to load checkpoint.')

    # Device Options
    device_type: Literal['CPU', 'GPU'] = Field('GPU', description='Device type for model evaluation. Use CUDA_VISIBLE_DEVICES environment variable to control GPU selection.')

    # Iteration Options
    batch_size: int = Field(32, ge=1, description='Batch size for validation.')


@register_engine('evaluator', 'standard')
class StandardEvaluator(BaseEngine[StandardEvaluatorOptions]):
    OPTIONS = StandardEvaluatorOptions

    def log(self, metric_names: list[str], metric_values: list[torch.Tensor], metric_formats: list[Callable[[float], str]]) -> None:
        logs = list()
        for metric_name, metric_value, metric_format in zip(metric_names, metric_values, metric_formats):
            logs.append(f'[{metric_name}]={metric_format(float(metric_value / self.options.node_number))}')
        logger.info(f'Evaluation Results - {" ".join(logs)}')

    def run(
        self,
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        evaluate_fn: Callable[[Any], tuple[list[str], list[torch.Tensor | float], list[Callable[[float], str]]]],
        initialize_fn: Callable[[torch.nn.Module], None] = no_operation,
        dataloader_type: Literal['pth', 'pyg'] = 'pth',
    ) -> None:
        """
        Run the evaluation process.
        
        :param model: The model to be evaluated.
        :type model: torch.nn.Module
        :param dataset: The dataset used for evaluation.
        :type dataset: torch.utils.data.Dataset
        :param evaluate_fn: The function to evaluate the model.
        :type evaluate_fn: Callable[[Any], tuple[list[str], list[torch.Tensor | float], list[Callable[[float], str]]]]
        :param initialize_fn: The function to initialize the model before evaluation. Future it may also recieve other parameters if needed.
        :type initialize_fn: Callable[[torch.nn.Module], None]
        :param dataloader_type: The type of dataloader to use ('pth' for PyTorch, 'pyg' for PyTorch Geometric).
        :type dataloader_type: Literal[&#39pth&#39, &#39pyg&#39]
        """

        checkpoint = load_checkpoint(self.options.checkpoint_filepath)

        logger.info(f'-> Checkpoint from [Epoch/Step/Itr]@[{checkpoint.epoch}/{checkpoint.step}/{checkpoint.itr}].')

        logger.info(f'    v Loading Parameters ...')
        model.load_state_dict(checkpoint.model_state_dict)
        logger.info(f'    ^ Loaded.')

        self.evaluate(
            model,
            dataset,
            evaluate_fn,
            initialize_fn,
            dataloader_type
        )

    def evaluate(
        self,
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        evaluate_fn: Callable[[Any], tuple[list[str], list[torch.Tensor | float], list[Callable[[float], str]]]],
        initialize_fn: Callable[[torch.nn.Module], None] = no_operation,
        dataloader_type: Literal['pth', 'pyg'] = 'pth',
    ) -> None:
        device_descriptor = get_device_descriptor(self.options.device_type, 0)
        model.to(device=device_descriptor)
        logger.info(f'-> Using Device: {device_descriptor}')

        initialize_fn(model)

        if dataloader_type == 'pth':
            from torch.utils.data import DataLoader
        if dataloader_type == 'pyg':
            from torch_geometric.loader import DataLoader

        dataloader = DataLoader(dataset, batch_size=self.options.batch_size, shuffle=False)

        logger.info(f'-> Dataset Size: {len(dataset)} | Batch Size: {self.options.batch_size} | Iteration Size: {len(dataloader)}')

        logger.info(f'-> Evaluating ...')
        tic = time.time()
        model.eval()
        with torch.no_grad():
            metric_names, metric_values, metric_formats = evaluate_fn(dataloader)
            self.log(metric_names, metric_values, metric_formats)

        toc = time.time()

        logger.info(f'-> Finished. Overall Time Cost = {toc-tic:.2f}s)')
