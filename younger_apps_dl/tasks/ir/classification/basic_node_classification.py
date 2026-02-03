#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-12-25 02:26:46
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-02-03 15:45:04
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import tqdm
import torch
import numpy
import pathlib

from typing import Literal, Callable
from pydantic import BaseModel, Field

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, top_k_accuracy_score

from younger_apps_dl.tasks import BaseTask, register_task
from younger_apps_dl.engines import StandardPreprocessor, StandardPreprocessorOptions, StandardTrainer, StandardTrainerOptions, StandardEvaluator, StandardEvaluatorOptions, StandardPredictor, StandardPredictorOptions
from younger_apps_dl.datasets import DAGDataset, DAGData
from younger_apps_dl.models import GAENodeClassification, GATNodeClassification, GCNNodeClassification, GINNodeClassification, SAGENodeClassification, VGAENodeClassification
from younger_apps_dl.commons.graph import dag_node_masking
from younger_apps_dl.commons.logging import logger


class ModelOptions(BaseModel):
    model_type: Literal['GAE', 'GAT', 'GCN', 'GIN', 'SAGE', 'VGAE'] = Field(..., description='The identifier of the Model Type, e.g., GAE, GAT, GCN, GIN, SAGE, VGAE.')
    node_emb_dim: int = Field(512, description='The Dimension of Node Embedding.')
    hidden_dim: int = Field(256, description='The Dimension of Hidden Layers.')
    dropout_rate: float = Field(0.5, description='The Dropout Rate.')
    layer_number: int = Field(3, description='The Number of GNN Layers.')


class OptimizerOptions(BaseModel):
    lr: float = Field(0.001, description='Learning rate used by the optimizer.')
    eps: float = Field(1e-8, description='Epsilon for numerical stability.')
    weight_decay: float = Field(0.01, description='L2 regularization (weight decay) coefficient.')
    amsgrad: bool = Field(False, description='Whether to use the AMSGrad variant of the Adam optimizer.')


class SchedulerOptions(BaseModel):
    factor: float = Field(0.2, description='Factor by which the learning rate will be reduced.')
    min_lr: float = Field(5e-5, description='A minimum learning rate value.')


class DatasetOptions(BaseModel):
    meta_filepath: pathlib.Path = Field(..., description='Path to the metadata file that describes the dataset.')
    raw_dirpath: pathlib.Path = Field('', description='Directory containing raw input data files.')
    raw_filename: str = Field('', description='Filename of the raw input data.')
    processed_dirpath: pathlib.Path = Field('', description='Directory where processed dataset should be stored.')
    processed_filename: str = Field('', description='Filename of the processed dataset.')
    worker_number: int = Field(4, description='Number of workers for parallel data loading or processing.')


class BasicNodeClassificationOptions(BaseModel):
    # Main Options
    mask_ratio: float = Field(..., description='Ratio of nodes to mask during training for self-supervised learning (0.0 to 1.0).')
    mask_method: Literal['Random', 'Purpose'] = Field(..., description='Node masking strategy: \'Random\' masks any node uniformly; \'Purpose\' only masks leaf nodes. It is recommended to use \'Purpose\' for DAGs with scheduled sampling.')
    mask_mode: Literal['BERT', 'PURE'] = Field('BERT', description='Masking mode: \'BERT\' for training with BERT-style masking; \'PURE\' for pure masking without random/no-change strategies.')

    stage: Literal['encode', 'classify', 'embedding'] = Field('classify', description='The Stage of the Model: \'encode\' for encoder only; \'classify\' for classification; \'embedding\' for embedding extraction. This option only required when using GAE or VGAE. Please leave it as default for other models.')

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


@register_task('ir', 'basic_node_classification')
class BasicNodeClassification(BaseTask[BasicNodeClassificationOptions]):
    """
    """
    OPTIONS = BasicNodeClassificationOptions
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
        self.train_dataset = self._build_dataset_(
            self.options.train_dataset.meta_filepath,
            self.options.train_dataset.raw_dirpath,
            self.options.train_dataset.raw_filename,
            self.options.train_dataset.processed_dirpath,
            self.options.train_dataset.processed_filename,
            'train',
            self.options.train_dataset.worker_number,
        )
        self.valid_dataset = self._build_dataset_(
            self.options.valid_dataset.meta_filepath,
            self.options.valid_dataset.raw_dirpath,
            self.options.valid_dataset.raw_filename,
            self.options.valid_dataset.processed_dirpath,
            self.options.valid_dataset.processed_filename,
            'valid',
            self.options.valid_dataset.worker_number,
        )
        self.model = self._build_model_(
            self.options.model.model_type,
            len(self.train_dataset.dicts['i2t']),
            self.options.model.node_emb_dim,
            self.options.model.hidden_dim,
            dropout_rate=self.options.model.dropout_rate,
            layer_number=self.options.model.layer_number,
            stage=self.options.stage,
        )
        self.optimizer = self._build_optimizer_(
            self.model,
            lr=self.options.optimizer.lr,
            eps=self.options.optimizer.eps,
            weight_decay=self.options.optimizer.weight_decay,
            amsgrad=self.options.optimizer.amsgrad
        )
        self.scheduler = self._build_scheduler_(
            self.optimizer,
            factor=self.options.scheduler.factor,
            min_lr=self.options.scheduler.min_lr,
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
            on_update_fn=self._on_update_fn_,
            on_lfresh_fn=self._on_lfresh_fn_,
            dataloader_type='pyg'
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
        )
        self.model = self._build_model_(
            self.options.model.model_type,
            len(self.test_dataset.dicts['i2t']),
            self.options.model.node_emb_dim,
            self.options.model.hidden_dim,
            dropout_rate=self.options.model.dropout_rate,
            layer_number=self.options.model.layer_number,
            stage=self.options.stage,
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
        """
        The filepath of the meta file used during prediction must be the same as that used during training.
        """
        meta = DAGDataset.load_meta(self.options.predict_dataset.meta_filepath)
        self.dicts = DAGDataset.load_dicts(meta)
        self.model = self._build_model_(
            self.options.model.model_type,
            len(self.dicts['i2t']),
            self.options.model.node_emb_dim,
            self.options.model.hidden_dim,
            dropout_rate=self.options.model.dropout_rate,
            layer_number=self.options.model.layer_number,
            stage=self.options.stage,
        )

        self.dicts = self.predict_dataset.dicts

        predictor = StandardPredictor(self.options.predictor)
        predictor.run(
            self.model,
            predict_raw_fn=self._predict_raw_fn_,
            initialize_fn=self._initialize_fn_,
        )

    def _postprocess_(self):
        """Postprocess implementation (currently not used in this task)."""
        raise NotImplementedError("Postprocessing is not implemented for BasicNodeClassification task")

    def _build_model_(self,
        model_type: Literal['GAE', 'GAT', 'GCN', 'GIN', 'SAGE', 'VGAE'],
        node_emb_size: int,
        node_emb_dim: int,
        hidden_dim: int,
        dropout_rate: float = 0.5,
        layer_number: int = 3,
        stage: Literal['encode', 'classify', 'embedding'] = 'classify',
    ) -> GAENodeClassification | GATNodeClassification | GCNNodeClassification | GINNodeClassification | SAGENodeClassification | VGAENodeClassification:
        if model_type == 'GAE':
            model = GAENodeClassification(
                node_emb_size,
                node_emb_dim,
                hidden_dim,
                stage=stage,
            )
        if model_type == 'GAT':
            model = GATNodeClassification(
                node_emb_size,
                node_emb_dim,
                hidden_dim,
                dropout_rate,
                stage=stage,
            )
        if model_type == 'GCN':
            model = GCNNodeClassification(
                node_emb_size,
                node_emb_dim,
                hidden_dim,
                dropout_rate,
                stage=stage,
            )
        if model_type == 'GIN':
            model = GINNodeClassification(
                node_emb_size,
                node_emb_dim,
                hidden_dim,
                dropout_rate,
                layer_number=layer_number,
                stage=stage,
            )
        if model_type == 'SAGE':
            model = SAGENodeClassification(
                node_emb_size,
                node_emb_dim,
                hidden_dim,
                dropout_rate,
                stage=stage,
            )
        if model_type == 'VGAE':
            model = VGAENodeClassification(
                node_emb_size,
                node_emb_dim,
                hidden_dim,
                stage=stage,
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
        factor: float,
        min_lr: float,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, min_lr=min_lr)
        return scheduler

    def _initialize_fn_(self, model: GAENodeClassification | GATNodeClassification | GCNNodeClassification | GINNodeClassification | SAGENodeClassification | VGAENodeClassification) -> None:
        self.model = model
        self.device_descriptor = next(self.model.parameters()).device

    def _train_fn_(self, minibatch: DAGData) -> tuple[list[str], list[torch.Tensor], list[Callable[[float], str]]]:
        """
        Train Function for Basic Node Classification Task.
        Only Support GAE, VGAE, GAT, GCN, GIN, SAGE Models.
        Embedding stage is not supported during training.
        """
        minibatch = minibatch.to(self.device_descriptor)
        x, edge_index, golden = self._mask_(minibatch, self.dicts['t2i'], self.options.mask_ratio, self.options.mask_method, mode=self.options.mask_mode, ignore_index=-1)
        if self.options.stage in ['encode', 'classify']:
            if self.options.stage == 'encode':
                assert self.options.model.model_type in ['GAE', 'VGAE'], f'Only GAE and VGAE support encoding stage.'
                z = self.model(x, edge_index)
                if self.options.model.model_type == 'GAE':
                    loss = self.model.encoder.recon_loss(z, edge_index)
                if self.options.model.model_type == 'VGAE':
                    loss = self.model.encoder.recon_loss(z, edge_index) + 0.001 * self.model.encoder.kl_loss()

            if self.options.stage == 'classify':
                output = self.model(x, edge_index)
                loss = torch.nn.functional.cross_entropy(output, golden.squeeze(1), ignore_index=-1)
        else:
            raise ValueError(f'Unsupported Stage During Training: {self.options.stage}')

        # Backward pass - delegate to train_fn for flexibility
        loss.backward()

        return self._compute_metrics_(loss.item())

    def _valid_fn_(self, dataloader: torch.utils.data.DataLoader) -> tuple[list[str], list[torch.Tensor], list[Callable[[float], str]]]:
        if self.options.stage == 'encode':
            assert self.options.model.model_type in ['GAE', 'VGAE'], f'Only GAE and VGAE support encoding stage.'
            losses = list()
            with torch.no_grad():
                with tqdm.tqdm(total=len(dataloader)) as progress_bar:
                    for index, minibatch in enumerate(dataloader, start=1):
                        minibatch = minibatch.to(self.device_descriptor)
                        x, edge_index, golden = self._mask_(minibatch, self.dicts['t2i'], self.options.mask_ratio, self.options.mask_method, mode=self.options.mask_mode, ignore_index=-1)
                        z = self.model(x, edge_index)
                        if self.options.model.model_type == 'GAE':
                            batch_loss = self.model.encoder.recon_loss(z, edge_index)
                        else:
                            batch_loss = self.model.encoder.recon_loss(z, edge_index) + 0.001 * self.model.encoder.kl_loss()
                        losses.append(batch_loss)
                        progress_bar.update(1)
            loss = torch.stack(losses).mean()
            metrics = self._compute_metrics_(loss.item())

        if self.options.stage == 'classify':
            outputs = list()
            goldens = list()
            with torch.no_grad():
                with tqdm.tqdm(total=len(dataloader)) as progress_bar:
                    for index, minibatch in enumerate(dataloader, start=1):
                        minibatch = minibatch.to(self.device_descriptor)
                        x, edge_index, golden = self._mask_(minibatch, self.dicts['t2i'], self.options.mask_ratio, self.options.mask_method, mode=self.options.mask_mode, ignore_index=-1)
                        output = self.model(x, edge_index)

                        outputs.append(output)
                        goldens.append(golden)
                        progress_bar.update(1)

            outputs = torch.cat(outputs)
            goldens = torch.cat(goldens).squeeze(1)
            loss = torch.nn.functional.cross_entropy(outputs, goldens, ignore_index=-1)

            val_indices = goldens != -1
            outputs = outputs[val_indices]
            goldens = goldens[val_indices]

            score = torch.softmax(outputs, dim=-1).cpu().numpy()
            pred = outputs.max(1)[1].cpu().numpy()
            gold = goldens.cpu().numpy()

            # For Debug Logging
            logger.info(f'pred[:5]: {pred[:5]}')
            logger.info(f'gold[:5]: {gold[:5]}')

            metrics = self._compute_metrics_(loss.item(), pred=pred, gold=gold, score=score)

        if self.options.stage == 'embedding':
            raise ValueError('_valid_fn_ does not support Embedding stage.')

        return metrics

    def _test_fn_(self, dataloader: torch.utils.data.DataLoader) -> tuple[list[str], list[torch.Tensor], list[Callable[[float], str]]]:
        return self._valid_fn_(dataloader)

    def _on_update_fn_(self) -> None:
        """
        Callback function for each parameter update during training.
        Users can customize this function to update parameters.
        """
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _on_lfresh_fn_(self, metrics: tuple[list[str], list[torch.Tensor], list[Callable[[float], str]]]) -> None:
        """
        Callback function for each parameter update during training.
        Some built-in schedulers (e.g., ReduceLROnPlateau) may require validation metrics to step.
        Users can customize this function to adjust learning rate schedulers, or perform other actions based on metrics.
        One may use validation metrics to adjust learning rates here.
        Or one can leave it empty if not needed.
        It is convenient to use _on_step_end_fn_ or _on_epoch_end_fn_ for epoch-level scheduler stepping.
        """
        metric_names, metric_values, _ = metrics
        metric_dict = dict(zip(metric_names, metric_values))
        if 'loss' in metric_dict:
            self.scheduler.step(metric_dict['loss'])

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
            acc = accuracy_score(gold, pred)
            macro_p = precision_score(gold, pred, average='macro', zero_division=0)
            macro_r = recall_score(gold, pred, average='macro', zero_division=0)
            macro_f1 = f1_score(gold, pred, average='macro', zero_division=0)
            micro_f1 = f1_score(gold, pred, average='micro', zero_division=0)
            metrics.extend([
                ('acc', acc, lambda x: f'{x:.4f}'),
                ('macro_p', macro_p, lambda x: f'{x:.4f}'),
                ('macro_r', macro_r, lambda x: f'{x:.4f}'),
                ('macro_f1', macro_f1, lambda x: f'{x:.4f}'),
                ('micro_f1', micro_f1, lambda x: f'{x:.4f}'),
            ])

        if gold is not None and score is not None:
            top3_acc = top_k_accuracy_score(gold, score, k = 3, labels=range(score.shape[1]))
            top5_acc = top_k_accuracy_score(gold, score, k = 5, labels=range(score.shape[1]))
            metrics.extend([
                ('top3_acc', top3_acc, lambda x: f'{x:.4f}'),
                ('top5_acc', top5_acc, lambda x: f'{x:.4f}')
            ])

        # Unpack tuples: (name, value, format) -> three separate lists
        metric_names_list, metric_values_list, metric_formats_list = zip(*metrics)
        return list(metric_names_list), list(metric_values_list), list(metric_formats_list)

    def _mask_(
        self,
        minibatch: DAGData,
        x_dict: dict[str, int],
        mask_ratio: float,
        mask_method: Literal['Random', 'Purpose'],
        mode: Literal['BERT', 'PURE'] = 'BERT',
        ignore_index: int = -1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, edge_index, golden = dag_node_masking(minibatch.x, minibatch.edge_index, x_dict, mask_ratio, mask_method, self.device_descriptor, mode, ignore_index)
        return x, edge_index, golden

    def _predict_raw_fn_(self, load_dirpath: pathlib.Path, save_dirpath: pathlib.Path) -> torch.Tensor:
        pass
