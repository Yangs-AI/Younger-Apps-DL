#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) Jason Young (杨郑鑫).
#
# E-Mail: <AI.Jason.Young@outlook.com>
# 2024-05-19 18:30
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.


import tqdm
import torch
import numpy
import pathlib

from typing import Any, overload, Literal, Callable
from pydantic import BaseModel, Field

from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

from younger_apps_dl.models.node.gae import GAENodeClassification
from younger_apps_dl.tasks import BaseTask, register_task
from younger_apps_dl.engines import StandardPreprocessor, StandardPreprocessorOptions, StandardTrainer, StandardTrainerOptions, StandardEvaluator, StandardEvaluatorOptions, StandardPredictor, StandardPredictorOptions
from younger_apps_dl.datasets import DAGDataset, DAGData
from younger_apps_dl.models import GATEdgeClassification, GCNEdgeClassification, SAGEEdgeClassification
from younger_apps_dl.commons.graph import dag_edge_masking
from younger_apps_dl.commons.logging import logger


class ModelOptions(BaseModel):
    model_type: Literal['GAT', 'GCN', 'SAGE'] = Field(..., description='The identifier of the Model Type, e.g., GAT, GCN, SAGE.')
    node_emb_dim: int = Field(256, description='The Dimension of Node Embedding.')
    hidden_dim: int = Field(128, description='The Dimension of Hidden Layers.')
    dropout_rate: float = Field(0.5, description='The Dropout Rate.')
    output_dim: int = Field(64, description='The Output Dimension of the Model.')


class OptimizerOptions(BaseModel):
    lr: float = Field(0.0005, description='Learning rate used by the optimizer.')
    eps: float = Field(1e-8, description='Epsilon for numerical stability.')
    weight_decay: float = Field(0.01, description='L2 regularization (weight decay) coefficient.')
    amsgrad: bool = Field(False, description='Whether to use the AMSGrad variant of the Adam optimizer.')


class SchedulerOptions(BaseModel):
    step_size: int = Field(100, description='Period of learning rate decay.')
    gamma: float = Field(0.1, description='Multiplicative factor of learning rate decay.')


class DatasetOptions(BaseModel):
    meta_filepath: pathlib.Path = Field(..., description='Path to the metadata file that describes the dataset.')
    raw_dirpath: pathlib.Path = Field('', description='Directory containing raw input data files.')
    raw_filename: str = Field('', description='Filename of the raw input data.')
    processed_dirpath: pathlib.Path = Field('', description='Directory where processed dataset should be stored.')
    processed_filename: str = Field('', description='Filename of the processed dataset.')
    worker_number: int = Field(4, description='Number of workers for parallel data loading or processing.')


class BasicNodeClassificationOptions(BaseModel):
    # Main Options
    mask_ratio: float = Field(..., description='Ratio of edges to mask for self-supervised learning (0.0 to 1.0). (1 - mask_ratio) * num_edge edges are kept as prior knowledge.')
    sample_ratio: float = Field(..., description='Ratio of positive and negative samples to generate for the masked edges (0.0 to 1.0). 2 * sample_ratio * mask_ratio * num_edge edges are generated as samples.')

    trainer: StandardTrainerOptions
    evaluator: StandardEvaluatorOptions
    preprocessor: StandardPreprocessorOptions
    predictor: StandardPredictorOptions

    train_dataset: DatasetOptions
    valid_dataset: DatasetOptions
    test_dataset: DatasetOptions
    predict_dataset: DatasetOptions

    model: ModelOptions
    optimizer: OptimizerOptions
    scheduler: SchedulerOptions


@register_task('ir', 'basic_edge_classification')
class BasicEdgeClassification(BaseTask[BasicNodeClassificationOptions]):
    OPTIONS = BasicNodeClassificationOptions

    def preprocess(self):
        preprocessor = StandardPreprocessor(self.options.preprocessor)
        preprocessor.run()

    def train(self):
        self.train_dataset = self._build_dataset_(
            self.options.train_dataset.meta_filepath,
            self.options.train_dataset.raw_dirpath,
            self.options.train_dataset.raw_filename,
            self.options.train_dataset.processed_dirpath,
            self.options.train_dataset.processed_filename,
            'train',
            self.options.train_dataset.worker_number
        )
        self.valid_dataset = self._build_dataset_(
            self.options.valid_dataset.meta_filepath,
            self.options.valid_dataset.raw_dirpath,
            self.options.valid_dataset.raw_filename,
            self.options.valid_dataset.processed_dirpath,
            self.options.valid_dataset.processed_filename,
            'valid',
            self.options.valid_dataset.worker_number
        )
        self.model = self._build_model_(
            self.options.model.model_type,
            len(self.train_dataset.dicts['i2t']),
            self.options.model.node_emb_dim,
            self.options.model.hidden_dim,
            self.options.model.output_dim,
            self.options.model.dropout_rate,
        )
        self.optimizer = self._build_optimizer_(
            self.model,
            self.options.optimizer.lr,
            self.options.optimizer.eps,
            self.options.optimizer.weight_decay,
            self.options.optimizer.amsgrad,
        )
        self.scheduler = self._build_scheduler_(
            self.optimizer,
            self.options.scheduler.step_size,
            self.options.scheduler.gamma,
        )
        self.dicts = self.train_dataset.dicts

        trainer = StandardTrainer(self.options.trainer)
        trainer.run(
            self.model,
            self.optimizer,
            self.scheduler,
            self.train_dataset,
            self.valid_dataset,
            self._train_fn_,
            self._valid_fn_,
            initialize_fn=self._initialize_fn_,
            dataloader_type='pyg',
        )

    def evaluate(self):
        self.test_dataset = self._build_dataset_(
            self.options.test_dataset.meta_filepath,
            self.options.test_dataset.raw_dirpath,
            self.options.test_dataset.raw_filename,
            self.options.test_dataset.processed_dirpath,
            self.options.test_dataset.processed_filename,
            'test',
            self.options.test_dataset.worker_number,
        )
        self.model = self._build_model_(
            self.options.model.model_type,
            len(self.train_dataset.dicts['i2t']),
            self.options.model.node_emb_dim,
            self.options.model.hidden_dim,
            self.options.model.output_dim,
            self.options.model.dropout_rate,
        )
        self.dicts = self.test_dataset.dicts
        evaluator = StandardEvaluator(self.options.evaluator)
        evaluator.run(
            self.model,
            self.test_dataset,
            self._test_fn_,
            initialize_fn=self._initialize_fn_,
            dataloader_type='pyg'
        )

    @overload
    def _build_model_(
        self,
        model_type: Literal["GAT"],
        node_emb_size: int,
        node_emb_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float = 0.5,
    ) -> GATEdgeClassification: ...

    @overload
    def _build_model_(
        self,
        model_type: Literal["GCN"],
        node_emb_size: int,
        node_emb_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float = 0.5,
    ) -> GCNEdgeClassification: ...

    @overload
    def _build_model_(
        self,
        model_type: Literal["SAGE"],
        node_emb_size: int,
        node_emb_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float = 0.5,
    ) -> SAGEEdgeClassification: ...

    def _build_model_(self,
        model_type: Literal['GAT', 'GCN', 'SAGE'],
        node_emb_size: int,
        node_emb_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float = 0.5,
    ) -> GATEdgeClassification | GCNEdgeClassification | SAGEEdgeClassification:
        if model_type == 'GAT':
            model = GATEdgeClassification(
                node_emb_size,
                node_emb_dim,
                hidden_dim,
                output_dim,
                dropout_rate
            )
        if model_type == 'GCN':
            model = GCNEdgeClassification(
                node_emb_size,
                node_emb_dim,
                hidden_dim,
                output_dim,
                dropout_rate
            )
        if model_type == 'SAGE':
            model = SAGEEdgeClassification(
                node_emb_size,
                node_emb_dim,
                hidden_dim,
                output_dim,
                dropout_rate
            )
        return model

    def _build_dataset_(self, meta_filepath: pathlib.Path, raw_dirpath: pathlib.Path, raw_filename: str, processed_dirpath: pathlib.Path, processed_filename: str, split: Literal['train', 'valid', 'test'], worker_number: int) -> DAGDataset:
        dataset = DAGDataset(
            meta_filepath,
            raw_dirpath,
            raw_filename,
            processed_dirpath,
            processed_filename,
            split=split,
            worker_number=worker_number
        )
        return dataset

    def _build_optimizer_(
        self,
        model: torch.nn.Module,
        lr: float,
        eps: float,
        weight_decay: float,
        amsgrad: bool
    ) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        return optimizer

    def _build_scheduler_(
        self,
        optimizer: torch.optim.Optimizer,
        step_size: int,
        gamma: float,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=gamma)
        return scheduler

    def _initialize_fn_(self, model: GATEdgeClassification | GCNEdgeClassification | SAGEEdgeClassification) -> None:
        self.model = model
        self.device_descriptor = next(self.model.parameters()).device

    def _train_fn_(self, minibatch: DAGData) -> tuple[list[str], list[torch.Tensor], list[Callable[[float], str]]]:
        minibatch = minibatch.to(self.device_descriptor)
        x, edge_index, edge_position, edge_position_label = self._mask_(
            minibatch,
            self.options.mask_ratio,
            self.options.sample_ratio,
        )
        output = self.model(x, edge_index, edge_position)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, edge_position_label)

        return self._compute_metrics_(loss.item())

    def _valid_fn_(self, dataloader: torch.utils.data.DataLoader) -> tuple[list[str], list[torch.Tensor], list[Callable[[float], str]]]:
        outputs = list()
        goldens = list()
        with torch.no_grad():
            with tqdm.tqdm(total=len(dataloader)) as progress_bar:
                for index, minibatch in enumerate(dataloader, start=1):
                    minibatch: DAGData = minibatch.to(self.device_descriptor)
                    x, edge_index, edge_position, edge_position_label = self._mask_(minibatch, self.options.mask_ratio, self.options.sample_ratio)
                    output = self.model(x, edge_index, edge_position)

                    outputs.append(output)
                    goldens.append(edge_position_label)
                    progress_bar.update(1)

        outputs = torch.cat(outputs)
        goldens = torch.cat(goldens)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, edge_position_label)

        score = outputs.sigmoid().cpu().numpy()
        pred = (score > 0.5).astype(int)
        gold = goldens.cpu().numpy()

        # For Debug Logging
        logger.info(f'pred[:5]: {pred[:5]}')
        logger.info(f'gold[:5]: {gold[:5]}')

        return self._compute_metrics_(loss.item(), pred=pred, gold=gold, score=score)

    def _test_fn_(self, dataloader: torch.utils.data.DataLoader) -> tuple[list[str], list[torch.Tensor], list[Callable[[float], str]]]:
        return self._valid_fn_(dataloader)

    def _compute_metrics_(
        self,
        loss: float,
        pred: numpy.ndarray | None = None,
        gold: numpy.ndarray | None = None,
        score: numpy.ndarray | None = None,
        logger_prefix: str = ''
    ) -> tuple[list[str], list[torch.Tensor], list[Callable[[float], str]]]:
        """
        Compute evaluation metrics.

        Args:
            loss: Loss value (scalar) - required
            pred: Predicted labels [N] - optional
            gold: Ground truth labels [N] - optional
            score: Prediction probabilities [N, V] - optional
            logger_prefix: Prefix for logging output - optional

        Returns:
            (metric_names, metric_values, metric_formats)
        """

        if len(logger_prefix) != 0:
            logger.info(f'{logger_prefix}')

        metrics = [
            ('loss', loss, lambda x: f'{x:.4f}'),
        ]

        if pred is not None and gold is not None:
            f1 = f1_score(gold, pred, average='binary', zero_division=0)
            metrics.extend([
                ('f1', f1, lambda x: f'{x:.4f}'),
            ])

        if gold is not None and score is not None:
            ap = average_precision_score(gold, score)
            auc = roc_auc_score(gold, score)
            metrics.extend([
                ('ap', ap, lambda x: f'{x:.4f}'),
                ('auc', auc, lambda x: f'{x:.4f}')
            ])

        # Unpack tuples: (name, value, format) -> three separate lists
        metric_names_list, metric_values_list, metric_formats_list = zip(*metrics)
        return list(metric_names_list), list(metric_values_list), list(metric_formats_list)

    def _mask_(
        self,
        minibatch: DAGData,
        mask_ratio: float,
        sample_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, edge_index, edge_position, edge_position_label = dag_edge_masking(minibatch.x, minibatch.edge_index, mask_ratio, sample_ratio, self.device_descriptor)
        return x, edge_index, edge_position, edge_position_label
