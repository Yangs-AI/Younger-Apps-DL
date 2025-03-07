#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-10-16 22:58:32
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-02-25 16:42:29
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import tqdm
import time
import torch
import pathlib

from typing import Any, Literal, Callable
from pydantic import BaseModel, Field

from younger.commons.io import create_dir

from younger_apps_dl.commons.logging import equip_logger, logger
from younger_apps_dl.commons.checkpoint import load_checkpoint

from younger_apps_dl.engines import BaseEngine


class StandardEvaluatorOptions(BaseModel):
    # Main Options
    logging_filepath: str = Field('./standard_evaluator.log', description="Logging file path where logs will be saved, default to None, which may save to a default path that is determined by the Younger.")

    # Checkpoint Options
    checkpoint_filepath: str  = Field(..., description="Path to load checkpoint.")

    # Iteration Options
    batch_size: int = Field(32, ge=1, description="Batch size for validation.")


class StandardEvaluator(BaseEngine[StandardEvaluatorOptions]):
    _options_ = StandardEvaluatorOptions

    def __init__(
        self,
        configuration: dict,
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        evaluate_fn: Callable[[torch.nn.Module, Any], tuple[list[str], list[torch.Tensor], list[Callable[[float], str]]]],
        dataloader: Literal['pth', 'pyg'] = 'pth',
    ):
        super().__init__(configuration)
        self.model = model
        self.dataset = dataset
        self.evaluate_fn = evaluate_fn
        self.dataloader = dataloader

    def log(self, metric_names: list[str], metric_values: list[torch.Tensor], metric_formats: list[Callable[[float], str]]) -> None:
        logs = list()
        for metric_name, metric_value, metric_format in zip(metric_names, metric_values, metric_formats):
            logs.append(f'[{metric_name}]={metric_format(float(metric_value / self.options.node_number))}')
        logger.info(f'Evaluation Results - {' '.join(logs)}')

    def run(self):
        equip_logger(self.options.logging_filepath)
        checkpoint = load_checkpoint(pathlib.Path(self.options.checkpoint_filepath))

        logger.info(f'-> Checkpoint from [Epoch/Step/Itr]@[{checkpoint.epoch}/{checkpoint.step}/{checkpoint.itr}].')

        logger.info(f'    v Loading Parameters ...')
        self.model.load_state_dict(checkpoint.model_state_dict)
        logger.info(f'    ^ Loaded.')

        self.evaluate()

    def evaluate(self):
        if self.dataloader == 'pth':
            from torch.utils.data import DataLoader
        if self.dataloader == 'pyg':
            from torch_geometric.loader import DataLoader

        dataloader = DataLoader(self.dataset, batch_size=self.options.batch_size, shuffle=False)

        logger.info(f'-> Dataset Size: {len(self.dataset)} | Batch Size: {self.options.batch_size} | Iteration Size: {len(dataloader)}')

        logger.info(f'-> Evaluating ...')
        tic = time.time()
        self.model.eval()
        with torch.no_grad():
            metric_names, metric_values, metric_formats = self.evaluate_fn(self.model, dataloader)
            self.log(metric_names, metric_values, metric_formats)

        toc = time.time()

        logger.info(f'-> Finished. Overall Time Cost = {toc-tic:.2f}s)')

