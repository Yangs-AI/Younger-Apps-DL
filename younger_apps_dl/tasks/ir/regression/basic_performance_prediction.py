#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-03-31 21:20:17
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-01-27 14:55:43
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import tqdm
import torch
import pathlib

from typing import overload, Literal, Callable
from pydantic import BaseModel, Field

from younger_apps_dl.tasks import BaseTask, register_task
from younger_apps_dl.engines import StandardPreprocessor, StandardPreprocessorOptions, StandardTrainer, StandardTrainerOptions, StandardEvaluator, StandardEvaluatorOptions, StandardPredictor, StandardPredictorOptions
from younger_apps_dl.datasets import DAGWithLabelsDataset, DAGData
from younger_apps_dl.models import GATPerformancePrediction, GINPerformancePrediction
from younger_apps_dl.commons.logging import logger


class ModelOptions(BaseModel):
    model_type: Literal['GIN', 'GAT'] = Field('GIN', description='The identifier of the model type, e.g., \'GIN\', etc.')
    node_emb_dim: int = Field(512, description='Node embedding dimensionality.')
    hidden_dim: int = Field(256, description='Hidden layer dimensionality within the model.')
    head_number: int = Field(4, description='Number of attention heads (applicable for GAT model).')
    readout_dim: int = Field(256, description='Dimensionality of the readout layer.')
    output_dim: int = Field(1, description='Number of output labels to predict (e.g., 1 for single metric, N for multiple metrics). It should be the same as the number of label keys specified in the dataset options.')
    dropout_rate: float = Field(0.5, description='Dropout probability used for regularization.')


class OptimizerOptions(BaseModel):
    lr: float = Field(0.001, description='Learning rate used by the optimizer.')
    eps: float = Field(1e-8, description='Epsilon for numerical stability.')
    weight_decay: float = Field(0.01, description='L2 regularization (weight decay) coefficient.')
    amsgrad: bool = Field(False, description='Whether to use the AMSGrad variant of the Adam optimizer.')


class SchedulerOptions(BaseModel):
    start_factor: float = Field(0.1, description='Initial learning rate multiplier for warm-up.')
    warmup_steps: int = Field(500, description='Number of warm-up steps at the start of training using linear learning rate scaling.')
    total_steps: int = Field(60000, description='Total number of training steps for cosine annealing decay after warmup.')
    last_step: int = Field(-1, description='The last step index when resuming training. Use -1 to start fresh.')


class DatasetOptions(BaseModel):
    meta_filepath: pathlib.Path = Field(..., description='Path to the metadata file that describes the dataset.')
    raw_dirpath: pathlib.Path = Field(..., description='Directory containing raw input data files.')
    raw_filename: str = Field(..., description='Filename of the raw input data.')
    processed_dirpath: pathlib.Path = Field(..., description='Directory where processed dataset should be stored.')
    processed_filename: str = Field(..., description='Filename of the processed dataset.')
    worker_number: int = Field(4, description='Number of workers for parallel data loading or processing.')

    label_keys: list[str] = Field(..., description='List of keys identifying the labels to be predicted. (e.g., [\'download\', \'accuracy\'])')


class BasicPerformancePredictionOptions(BaseModel):
    # Main Options

    # Stage-specific options
    trainer: StandardTrainerOptions | None = Field(None, description='Trainer options, required for training stage.')
    evaluator: StandardEvaluatorOptions | None = Field(None, description='Evaluator options, required for evaluation stage.')
    preprocessor: StandardPreprocessorOptions | None = Field(None, description='Preprocessor options, required for preprocessing stage.')
    predictor: StandardPredictorOptions | None = Field(None, description='Predictor options, required for prediction stage.')

    train_dataset: DatasetOptions | None = Field(None, description='Training dataset options, required for training stage.')
    valid_dataset: DatasetOptions | None = Field(None, description='Validation dataset options, required for training stage.')
    test_dataset: DatasetOptions | None = Field(None, description='Test dataset options, required for evaluation stage.')
    predict_dataset: DatasetOptions | None = Field(None, description='Prediction dataset options, required for prediction stage.')

    model: ModelOptions | None = Field(None, description='Model options, required for training/evaluation/prediction stages.')
    optimizer: OptimizerOptions | None = Field(None, description='Optimizer options, required for training stage.')
    scheduler: SchedulerOptions | None = Field(None, description='Scheduler options, required for training stage.')


# Self-Supervised Learning for Node Prediction
@register_task('ir', 'basic_performance_prediction')
class BasicPerformancePrediction(BaseTask[BasicPerformancePredictionOptions]):
    """
    Basic Performance Prediction Task for Directed Acyclic Graphs (DAGs) using supervised learning.
    Implements a BERT-style masked node prediction framework with optional scheduled sampling.
    The model learns to predict masked nodes based on their context within the graph structure.
    It is a foundational task for various IR applications such as code generation, dag generation, etc.
    The task only supports one-step generation during training and validation stage, and generate full DAG based on several first level nodes and the topological structure during testing.
    One can copy and modify this class to implement more advanced generation tasks as needed.
    """
    OPTIONS = BasicPerformancePredictionOptions
    STAGE_REQUIRED_OPTION = {
        'preprocess': ['preprocessor'],
        'train': ['train_dataset', 'valid_dataset', 'model', 'optimizer', 'scheduler', 'trainer'],
        'evaluate': ['test_dataset', 'model', 'evaluator'],
        'predict': ['predict_dataset', 'model', 'predictor'],
        'postprocess': [],
    }

    def _preprocess_(self):
        preprocessor = StandardPreprocessor(self.options.preprocessor)
        preprocessor.run()

    def _train_(self):
        assert self.options.model.output_dim == len(self.options.train_dataset.label_keys), f"Model output_dim ({self.options.model.output_dim}) must match the number of label keys ({len(self.options.train_dataset.label_keys)}) in the training dataset."

        self.train_dataset = self._build_dataset_(
            self.options.train_dataset.meta_filepath,
            self.options.train_dataset.raw_dirpath,
            self.options.train_dataset.raw_filename,
            self.options.train_dataset.processed_dirpath,
            self.options.train_dataset.processed_filename,
            'train',
            self.options.train_dataset.worker_number,
            label_keys=self.options.train_dataset.label_keys,
        )
        self.valid_dataset = self._build_dataset_(
            self.options.valid_dataset.meta_filepath,
            self.options.valid_dataset.raw_dirpath,
            self.options.valid_dataset.raw_filename,
            self.options.valid_dataset.processed_dirpath,
            self.options.valid_dataset.processed_filename,
            'valid',
            self.options.valid_dataset.worker_number,
            label_keys=self.options.valid_dataset.label_keys,
        )
        self.model = self._build_model_(
            self.options.model.model_type,
            len(self.train_dataset.dicts['i2t']),
            self.options.model.node_emb_dim,
            self.options.model.hidden_dim,
            self.options.model.head_number,
            self.options.model.readout_dim,
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
            self.options.scheduler.start_factor,
            self.options.scheduler.warmup_steps,
            self.options.scheduler.total_steps,
            self.options.scheduler.last_step,
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

    def _evaluate_(self):
        self.test_dataset = self._build_dataset_(
            self.options.test_dataset.meta_filepath,
            self.options.test_dataset.raw_dirpath,
            self.options.test_dataset.raw_filename,
            self.options.test_dataset.processed_dirpath,
            self.options.test_dataset.processed_filename,
            'test',
            self.options.test_dataset.worker_number,
            self.options.test_dataset.label_keys,
        )
        self.model = self._build_model_(
            self.options.model.model_type,
            len(self.train_dataset.dicts['i2t']),
            self.options.model.node_emb_dim,
            self.options.model.hidden_dim,
            self.options.model.head_number,
            self.options.model.readout_dim,
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

    def _predict_(self):
        """Predict implementation (currently not used in this task)."""
        raise NotImplementedError("Prediction is not implemented for BasicPerformancePrediction task")

    def _postprocess_(self):
        """Postprocess implementation (currently not used in this task)."""
        raise NotImplementedError("Postprocessing is not implemented for BasicPerformancePrediction task")

    @overload
    def _build_model_(
        self,
        model_type: Literal["GAT"],
        node_emb_size: int,
        node_emb_dim: int,
        hidden_dim: int,
        head_number: int,
        readout_dim: int,
        output_dim: int,
        dropout_rate: float = 0.5,
    ) -> GATPerformancePrediction: ...

    @overload
    def _build_model_(
        self,
        model_type: Literal["GIN"],
        node_emb_size: int,
        node_emb_dim: int,
        hidden_dim: int,
        head_number: int,
        readout_dim: int,
        output_dim: int,
        dropout_rate: float = 0.5,
    ) -> GINPerformancePrediction: ...

    def _build_model_(self,
        model_type: Literal['GAT', 'GIN'],
        node_emb_size: int,
        node_emb_dim: int,
        hidden_dim: int,
        head_number: int,
        readout_dim: int,
        output_dim: int,
        dropout_rate: float = 0.5,
    ) -> GATPerformancePrediction | GINPerformancePrediction:
        if model_type == 'GAT':
            model = GATPerformancePrediction(
                node_emb_size,
                node_emb_dim,
                hidden_dim,
                head_number,
                readout_dim,
                output_dim,
                dropout_rate
            )
        if model_type == 'GIN':
            model = GINPerformancePrediction(
                node_emb_size,
                node_emb_dim,
                hidden_dim,
                readout_dim,
                output_dim,
                dropout_rate
            )
        return model

    def _build_dataset_(self, meta_filepath: pathlib.Path, raw_dirpath: pathlib.Path, raw_filename: str, processed_dirpath: pathlib.Path, processed_filename: str, split: Literal['train', 'valid', 'test'], worker_number: int, label_keys: list[str]) -> DAGWithLabelsDataset:
        dataset = DAGWithLabelsDataset(
            meta_filepath,
            raw_dirpath,
            raw_filename,
            processed_dirpath,
            processed_filename,
            split=split,
            worker_number=worker_number,
            label_keys=label_keys,
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
        start_factor: float,
        warmup_steps: int,
        total_steps: int,
        last_step: int,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        warmup_lr_schr = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=start_factor,
            total_iters=warmup_steps,
            last_epoch=last_step,
        )
        cosine_lr_schr = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            last_epoch=last_step,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_schr, cosine_lr_schr],
            milestones=[warmup_steps],
            last_epoch=last_step,
        )
        return scheduler

    def _initialize_fn_(self, model: GATPerformancePrediction | GINPerformancePrediction) -> None:
        self.model = model
        self.device_descriptor = next(self.model.parameters()).device

    def _train_fn_(self, minibatch: DAGData) -> tuple[list[str], list[torch.Tensor], list[Callable[[float], str]]]:
        minibatch = minibatch.to(self.device_descriptor)
        goldens = minibatch.y.reshape(len(minibatch), -1)
        # goldens: [batch_size, output_dim] where output_dim == len(label_keys)

        outputs = self.model(minibatch.x, minibatch.edge_index, minibatch.batch)

        return self._compute_metrics_(outputs, goldens)

    def _valid_fn_(self, dataloader: torch.utils.data.DataLoader) -> tuple[list[str], list[torch.Tensor], list[Callable[[float], str]]]:
        outputs = list()
        goldens = list()

        with torch.no_grad():
            with tqdm.tqdm(total=len(dataloader)) as progress_bar:
                for index, minibatch in enumerate(dataloader, start=1):
                    minibatch: DAGData = minibatch.to(self.device_descriptor)
                    golden = minibatch.y.reshape(len(minibatch), -1)
                    # golden: [batch_size, output_dim] where output_dim == len(label_keys)
                    output = self.model(minibatch.x, minibatch.edge_index, minibatch.batch)

                    outputs.append(output)
                    goldens.append(golden)
                    progress_bar.update(1)

        outputs = torch.cat(outputs)
        goldens = torch.cat(goldens)

        return self._compute_metrics_(outputs, goldens)

    def _test_fn_(self, dataloader: torch.utils.data.DataLoader) -> tuple[list[str], list[torch.Tensor], list[Callable[[float], str]]]:
        return self._valid_fn_(dataloader)

    def _compute_metrics_(
        self,
        pred: torch.Tensor,
        gold: torch.Tensor,
        logger_prefix: str = ''
    ) -> tuple[list[str], list[torch.Tensor], list[Callable[[float], str]]]:
        """
        Compute evaluation metrics for regression.

        Args:
            pred: Model predictions [batch_size] or [batch_size, output_dim]
            gold: Ground truth values [batch_size] or [batch_size, output_dim]
            logger_prefix: Prefix for logging output - optional

        Returns:
            (metric_names, metric_values, metric_formats)
        """

        if len(logger_prefix) != 0:
            logger.info(f'{logger_prefix}')

        # Reshape to ensure consistent dimensions
        pred = pred.reshape(-1, pred.shape[-1] if pred.dim() > 1 else 1)
        gold = gold.reshape(-1, gold.shape[-1] if gold.dim() > 1 else 1)

        # Compute regression metrics
        mae = torch.nn.functional.l1_loss(pred, gold, reduction='mean')
        mse = torch.nn.functional.mse_loss(pred, gold, reduction='mean')
        rmse = torch.sqrt(mse)

        metrics = [
            ('MAE', mae.item(), lambda x: f'{x:.4f}'),
            ('MSE', mse.item(), lambda x: f'{x:.4f}'),
            ('RMSE', rmse.item(), lambda x: f'{x:.4f}'),
        ]

        # Unpack tuples: (name, value, format) -> three separate lists
        metric_names_list, metric_values_list, metric_formats_list = zip(*metrics)
        return list(metric_names_list), list(metric_values_list), list(metric_formats_list)
