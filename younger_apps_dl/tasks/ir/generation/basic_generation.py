#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-01-14 18:07:43
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-01-21 20:51:55
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
from younger.commons.io import create_dir

from younger_apps_dl.tasks import BaseTask, register_task
from younger_apps_dl.engines import StandardPreprocessor, StandardPreprocessorOptions, StandardTrainer, StandardTrainerOptions, StandardEvaluator, StandardEvaluatorOptions, StandardPredictor, StandardPredictorOptions
from younger_apps_dl.datasets import DAGDataset, DAGData
from younger_apps_dl.models import MAEGIN
from younger_apps_dl.commons.graph import dag_node_masking
from younger_apps_dl.commons.logging import logger


class ModelOptions(BaseModel):
    model_type: Literal['MAEGIN'] = Field('MAEGIN', description='The identifier of the model type, e.g., \'MAEGIN\', etc.')
    node_emb_dim: int = Field(512, description='Node embedding dimensionality.')
    hidden_dim: int = Field(256, description='Hidden layer dimensionality within the model.')
    dropout_rate: float = Field(0.5, description='Dropout probability used for regularization.')
    layer_number: int = Field(3, description='Number of layers (e.g., message-passing rounds for GNNs).')


class OptimizerOptions(BaseModel):
    lr: float = Field(0.001, description='Learning rate used by the optimizer.')
    eps: float = Field(1e-8, description='Epsilon for numerical stability.')
    weight_decay: float = Field(0.01, description='L2 regularization (weight decay) coefficient.')
    amsgrad: bool = Field(False, description='Whether to use the AMSGrad variant of the Adam optimizer.')


class SchedulerOptions(BaseModel):
    start_factor: float = Field(0.1, description='Initial learning rate multiplier for warm-up.')
    warmup_steps: int = Field(1500, description='Number of warm-up steps at the start of training.')
    total_steps: int = Field(150000, description='Total number of training steps for the scheduler to plan the learning rate schedule.')
    last_step: int = Field(-1, description='The last step index when resuming training. Use -1 to start fresh.')


class DatasetOptions(BaseModel):
    meta_filepath: pathlib.Path = Field(..., description='Path to the metadata file that describes the dataset.')
    raw_dirpath: pathlib.Path = Field(..., description='Directory containing raw input data files.')
    raw_filename: str = Field(..., description='Filename of the raw input data.')
    processed_dirpath: pathlib.Path = Field(..., description='Directory where processed dataset should be stored.')
    processed_filename: str = Field(..., description='Filename of the processed dataset.')
    worker_number: int = Field(4, description='Number of workers for parallel data loading or processing.')


class BasicGenerationOptions(BaseModel):
    # Main Options
    mask_ratio: float = Field(..., description='Ratio of nodes to mask during training for self-supervised learning (0.0 to 1.0).')
    mask_method: Literal['Random', 'Purpose'] = Field(..., description='Node masking strategy: \'Random\' masks any node uniformly; \'Purpose\' only masks leaf nodes. It is recommended to use \'Purpose\' for DAGs with scheduled sampling.')
    mask_mode: Literal['BERT', 'PURE'] = Field('BERT', description='Masking mode: \'BERT\' for training with BERT-style masking; \'PURE\' for pure masking without random/no-change strategies.')

    scheduled_sampling_enable: bool = Field(False, description='Whether to enable scheduled sampling during training.')
    scheduled_sampling_level: int = Field(5, description='Number of hierarchical levels for scheduled sampling.')
    scheduled_sampling_mode: Literal['Sigmoid', 'Smooth'] = Field('Smooth', description='Mode of scheduled sampling probability increase over time.')
    scheduled_sampling_prob: float = Field(0.5, description='Probability of scheduled sampling at each level (0.0 to 1.0).')
    scheduled_sampling_fix: bool = Field(False, description='If True, use a fixed scheduled sampling probability throughout training; if False, increase over time.')
    scheduled_sampling_mu: float = Field(1000.0, description='Controls the rate of increase in scheduled sampling probability over training epoch.')
    scheduled_sampling_k: float = Field(1.0, description='Steepness parameter for the sigmoid function controlling scheduled sampling probability.')
    scheduled_sampling_supervise: bool = Field(False, description='If True, provide supervision for sampled nodes during scheduled sampling; if False, do not supervise sampled nodes.')

    generation_initial_level: int = Field(0, ge=0, description='Number of initial levels (L) to use as ground truth in autoregressive generation. Nodes in levels 0 to L-1 are kept unchanged; levels L and beyond are masked and predicted.')

    # Stage-specific options (Optional fields based on which stage is being run)
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
@register_task('ir', 'basic_generation')
class BasicGeneration(BaseTask[BasicGenerationOptions]):
    """
    Basic Generation Task for Directed Acyclic Graphs (DAGs) using self-supervised learning.
    Implements a BERT-style masked node prediction framework with optional scheduled sampling.
    The model learns to predict masked nodes based on their context within the graph structure.
    It is a foundational task for various IR applications such as code generation, dag generation, etc.
    The task only supports one-step generation during training and validation stage, and generate full DAG based on several first level nodes and the topological structure during testing.
    One can copy and modify this class to implement more advanced generation tasks as needed.
    """
    OPTIONS = BasicGenerationOptions

    @property
    def required_option_names_by_stage(self) -> dict[str, list[str]]:
        return {
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
            self.options.model.dropout_rate,
            self.options.model.layer_number
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
            on_step_begin_fn=self._on_step_begin_fn_,
            on_step_end_fn=self._on_step_end_fn_,
            on_epoch_begin_fn=self._on_epoch_begin_fn_,
            on_epoch_end_fn=self._on_epoch_end_fn_,
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
            self.options.test_dataset.worker_number
        )
        self.model = self._build_model_(
            self.options.model.model_type,
            len(self.test_dataset.dicts['i2t']),
            self.options.model.node_emb_dim,
            self.options.model.hidden_dim,
            self.options.model.dropout_rate,
            self.options.model.layer_number
        )
        self.dicts = self.test_dataset.dicts

        evaluator = StandardEvaluator(self.options.evaluator)
        evaluator.run(
            self.model,
            self.test_dataset,
            self._test_fn_,
            initialize_fn=self._initialize_fn_,
            dataloader_type='pyg',
        )

    def _predict_(self):
        """
        The filepath of the meta file used during prediction must be the same as that used during training.
        """
        meta = DAGDataset.load_meta(self.options.predict_dataset.meta_filepath)
        self.dicts = DAGDataset.load_dicts(meta)
        self.model = self._build_model_(
            self.options.model.model_type,
            len(self.test_dataset.dicts['i2t']),
            self.options.model.node_emb_dim,
            self.options.model.hidden_dim,
            self.options.model.dropout_rate,
            self.options.model.layer_number
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
        raise NotImplementedError("Postprocessing is not implemented for BasicGeneration task")

    def _build_model_(self, model_type: Literal['MAEGIN'], node_emb_size: int, node_emb_dim: int, hidden_dim: int, dropout_rate: float, layer_number: int) -> MAEGIN:
        if model_type == 'MAEGIN':
            model = MAEGIN(
                node_emb_size,
                node_emb_dim,
                hidden_dim,
                dropout_rate,
                layer_number
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
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
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

    def _scheduled_sampling_probability_(self, epoch: int) -> float:
        """
        Compute the scheduled sampling probability for the given epoch.
        """
        if self.options.scheduled_sampling_enable:
            if self.options.scheduled_sampling_fix:
                scheduled_sampling_prob = self.options.scheduled_sampling_prob
            else:
                if self.options.scheduled_sampling_mode == 'Sigmoid':
                    # Gradually increase sampling probability over epochs using sigmoid curve
                    # Formula: p(epoch) = 1 / (1 + exp(-k * (epoch - mu)))
                    # where k controls steepness, mu is the inflection point
                    k = self.options.scheduled_sampling_k
                    mu = self.options.scheduled_sampling_mu
                    scheduled_sampling_prob = 1.0 / (1.0 + torch.exp(-k * (epoch - mu)))
                if self.options.scheduled_sampling_mode == 'Smooth':
                    # Gradually increase sampling probability over epochs using smooth curve
                    # Formula: p(epoch) = 1 / (1 + mu * exp(-0.1 * (epoch / mu)))
                    # where mu controls the rate of increase
                    mu = self.options.scheduled_sampling_mu
                    scheduled_sampling_prob = 1.0 / (1.0 + mu * torch.exp(-0.1 * (epoch / mu)))
        else:
            scheduled_sampling_prob = 0.0
        return scheduled_sampling_prob

    def _initialize_fn_(self, model: MAEGIN) -> None:
        self.model = model
        self.device_descriptor = next(self.model.parameters()).device

    def _train_fn_(self, minibatch: DAGData) -> tuple[list[str], list[torch.Tensor], list[Callable[[float], str]]]:
        """
        Self-supervised pre-training with optional scheduled sampling.

        Strategy:
        1. Apply masking to mask nodes according to mask_ratio and mask_method
        2. If scheduled_sampling is enabled:
           - Identify non-masked nodes (original/random tokens kept from BERT masking)
           - Probabilistically sample these non-masked nodes based on scheduled_sampling_prob
           - Replace sampled nodes with model predictions (autoregressive step)
           - This simulates a curriculum where model gradually learns to use its own predictions
        3. Forward pass through the model on masked graph
        4. Compute cross-entropy loss, ignoring non-masked positions
        5. Backward pass (optimizer stepping handled by trainer based on update_period)

        Args:
            minibatch: graph data batch containing node features and graph topology

        Returns:
            metrics: tuple of (metric_names, metric_values, metric_formats) for training metrics
        """
        minibatch = minibatch.to(self.device_descriptor)

        # Apply BERT-style masking
        x, edge_index, golden = self._mask_(minibatch, self.dicts['t2i'], self.options.mask_ratio, self.options.mask_method, mode=self.options.mask_mode, ignore_index=-1)

        # Apply scheduled sampling if enabled: replace non-masked nodes based on probability
        if self.options.scheduled_sampling_enable:
            # all_nodes shape: torch.Size([x.shape[0]]) = [x.shape[0]]
            all_nodes = torch.arange(x.shape[0], device=self.device_descriptor)

            # Get indices of masked nodes
            # masked_nodes shape: torch.Size([num_masked_nodes]) = [num_masked_nodes]
            masked_nodes = (x == self.dicts['t2i']['__MASK__']).nonzero(as_tuple=True)[0]

            # Get indices of non-masked nodes
            # non_masked_nodes shape: torch.Size([num_non_masked_nodes]) = [num_non_masked_nodes]
            non_masked_nodes = all_nodes[~torch.isin(all_nodes, masked_nodes)]

            if len(non_masked_nodes) != 0:
                # Randomly select which non-masked nodes to replace based on scheduled_sampling_prob
                # sampling_mask shape: torch.Size([num_non_masked_nodes]) = [num_non_masked_nodes]
                sampling_mask = torch.bernoulli(
                    torch.full(non_masked_nodes.shape, self.scheduled_sampling_prob, device=self.device_descriptor)
                ).bool()
                sampling_nodes = non_masked_nodes[sampling_mask]

                if len(sampling_nodes) != 0:
                    x_to_sample_golden = x.clone()
                    x_to_sample = x.clone()
                    x_to_sample[sampling_nodes] = self.dicts['t2i']['__MASK__']
                    # Get model predictions for all nodes in eval mode
                    self.model.eval()
                    with torch.no_grad():
                        outputs = torch.softmax(self.model(x_to_sample, edge_index), dim=-1) # [N, V]
                        # Sample from predicted distribution for selected non-masked nodes
                        sampled = torch.multinomial(outputs[sampling_nodes], num_samples=1).squeeze(-1) # [|sampling_nodes|]
                    self.model.train()
                    x[sampling_nodes, 0] = sampled
                    if self.options.scheduled_sampling_supervise:
                        golden[sampling_nodes, 0] = x_to_sample_golden[sampling_nodes, 0]
                    x = x.detach() # cut any accidental graph ties; sampling is non-differentiable anyway

        # Forward pass
        output = self.model(x, edge_index)
        loss = torch.nn.functional.cross_entropy(output, golden.squeeze(1), ignore_index=-1)

        # Backprop here; optimizer/scheduler stepping is handled by the trainer according to update_period.
        loss.backward()

        return self._compute_metrics_(loss.item())

    def _valid_fn_(self, dataloader: torch.utils.data.DataLoader) -> tuple[list[str], list[torch.Tensor], list[Callable[[float], str]]]:
        """
        Evaluate model on validation set with ground truth labels.

        Strategy:
        1. Apply masking (controlled by mask_ratio and mask_method) to create masked graphs
        2. Forward pass through the model
        3. Compute predictions for masked positions
        4. Evaluate against ground truth using standard metrics (accuracy, F1, precision, recall, top-k accuracy)
        5. This is identical to training-time masking to maintain distribution consistency

        Note: This function performs standard masked prediction WITHOUT autoregressive generation.

        For autoregressive evaluation (predicting multiple levels sequentially), see _test_fn_.

        Args:
            dataloader: dataLoader containing validation batches with labels

        Returns:
            metrics: tuple of (metric_names, metric_values, metric_formats) for validation metrics
        """
        outputs = list()
        goldens = list()
        # Evaluate standard masked prediction
        with torch.no_grad():
            with tqdm.tqdm(total=len(dataloader)) as progress_bar:
                for index, minibatch in enumerate(dataloader, start=1):
                    minibatch: DAGData = minibatch.to(self.device_descriptor)
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

        return self._compute_metrics_(loss.item(), pred=pred, gold=gold, score=score, logger_prefix='Formatted Validation Results (For Debug):')

    def _test_fn_(self, dataloader: torch.utils.data.DataLoader) -> tuple[list[str], list[torch.Tensor], list[Callable[[float], str]]]:
        """
        Autoregressive generation with level-wise constraints.

        Strategy:
        1. Use the full graph topology (DAG structure) as scaffold
        2. Keep the first L levels (generation_initial_levels) as ground truth inputs
        3. Mask all nodes in levels L and deeper
        4. Predict nodes iteratively from level L to the deepest level using autoregressive generation
        5. At each level:
           - Use model predictions from previous levels (not ground truth) as input
           - Mask current level nodes
           - Get model predictions for current level
           - Use greedy decoding (argmax) to select node types (maybe beam search in future)
           - Update input with predictions for next level
        6. Evaluate final predictions against ground truth using standard metrics

        This simulates the real inference scenario where we have:
        - Full graph structure (DAG topology is known)
        - Initial node types for the first L levels (ground truth available)
        - Task: Fill in node types for levels L and deeper using only the graph structure and initial nodes

        Args:
            dataloader: dataLoader containing test batches (with ground truth labels for evaluation)

        Returns:
            metrics: tuple of (metric_names, metric_values, metric_formats) for test metrics
        """

        outputs = list()
        goldens = list()
        with torch.no_grad():
            with tqdm.tqdm(total=len(dataloader)) as progress_bar:
                for index, minibatch in enumerate(dataloader, start=1):
                    minibatch: DAGData = minibatch.to(self.device_descriptor)
                    x = minibatch.x.clone()
                    edge_index = minibatch.edge_index.clone()
                    level = minibatch.level.clone()
                    golden = minibatch.x.clone()

                    # Use autoregressive step-by-step generation with initial levels
                    output = self._step_by_step_(x, edge_index, level)

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

        return self._compute_metrics_(loss.item(), pred=pred, gold=gold, score=score, logger_prefix='Formatted Test Results (For Debug):')

    def _on_step_begin_fn_(self, step: int) -> None:
        # Reserved for per-step custom logic (e.g., gradient scaling housekeeping)
        return

    def _on_step_end_fn_(self, step: int) -> None:
        # Reserved for per-step custom logic
        return

    def _on_epoch_begin_fn_(self, epoch: int) -> None:
        # Reserved for per-epoch custom logic (e.g., reset accumulators)
        """Adjust scheduled sampling probability at the beginning of each epoch"""
        self.scheduled_sampling_prob = self._scheduled_sampling_probability_(epoch)
        logger.info(f"Epoch {epoch}: Scheduled sampling probability = {self.scheduled_sampling_prob:.4f}")

    def _on_epoch_end_fn_(self, epoch: int) -> None:
        # Reserved for per-epoch custom logic
        return

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
                ('loss', loss, lambda x: f'{x:.4f}'),
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

    def _step_by_step_(self, x: torch.Tensor, edge_index: torch.Tensor, level: torch.Tensor) -> torch.Tensor:
        """
        Predict node types autoregressively level-by-level.

        This is the core autoregressive generation method reusable across _test_fn_, _predict_raw_fn_, etc, for Evaluator and Predictor.

        Level convention: levels are non-positive; leaf nodes have level 0, and ancestor levels 
        move toward more negative values (-1, -2, ...). When `initial_levels=L`, we keep the L 
        smallest (most negative) levels as fixed inputs and predict the remaining levels 
        autoregressively from the deepest fixed level toward the leaf level.

        One may need to call `Preprocessor.mark_node_levels` to compute node levels before using this method.

        Strategy:
        1. Extract graph structure (x, edge_index, level) from minibatch
        2. Keep nodes in levels [0, initial_levels) as ground truth inputs (unchanged)
        3. Mask all nodes in levels [initial_levels, max_level]
        4. For each level from initial_levels to max_level:
           a. Create subgraph containing nodes up to current level
           b. Mask current level nodes (set to [MASK] token)
           c. Forward pass through model
           d. Use greedy decoding (argmax) to get predicted node types (maybe beam search in future)
           e. Update x with predictions for next level iteration
        5. Return predicted probabilities

        Args:
            x: Node features tensor [num_nodes, 1]
            edge_index: Edge indices tensor [2, num_edges]
            level: Level tensor indicating hierarchical level of each node [num_nodes]
        Returns:
            global_outputs: [num_nodes, vocab_size] probability distributions for all nodes, if some dimensions are ignored during evaluation, their values are all -1
        """
        initial_level = self.options.generation_initial_levels
        # Initialize global prediction tensor for all nodes
        global_outputs = torch.zeros(x.shape[0], len(self.dicts['t2i']), device=self.device_descriptor)

        # Determine kept levels: L smallest (most negative) levels are treated as fixed inputs
        unique_levels = torch.unique(level).tolist()
        kept_levels = set(unique_levels[:initial_level])
        kept_mask = torch.tensor([lvl in kept_levels for lvl in level.tolist()], device=self.device_descriptor)
        global_outputs[kept_mask] = -1 # ignore kept levels during evaluation

        # Process remaining levels from leaves to root
        for current_level in unique_levels:
            if current_level in kept_levels:
                logger.info(f'Skipping level {current_level} (kept as ground truth input)')
                continue

            current_x_oid = torch.where(level <= current_level)[0] # Nodes up to current level
            current_p_oid = torch.where(level == current_level)[0] # Nodes to predict at current level
            oid2nid = torch.zeros(x.shape[0], dtype=torch.long, device=self.device_descriptor)
            oid2nid[current_x_oid] = torch.arange(current_x_oid.shape[0], device=self.device_descriptor) # Original ID to New ID mapping for x_nodes (subgraph)

            current_x_mask = torch.zeros(x.shape[0], dtype=torch.bool, device=self.device_descriptor)
            current_x_mask[current_x_oid] = True

            current_edge_index_mask = current_x_mask[edge_index[0]] & current_x_mask[edge_index[1]]
            current_edge_index = edge_index[:, current_edge_index_mask]
            current_edge_index = oid2nid[current_edge_index]
            current_x = x[current_x_oid].clone()
            current_x[oid2nid[current_p_oid]] = self.dicts['t2i']['__MASK__']

            current_outputs = self.model(current_x, current_edge_index)
            current_predict = torch.argmax(current_outputs, dim=-1, keepdim=True)

            x[current_p_oid] = current_predict[oid2nid[current_p_oid]]
            global_outputs[current_p_oid] = current_outputs[oid2nid[current_p_oid]]

        return global_outputs

    def _predict_raw_fn_(self, load_dirpath: pathlib.Path, save_dirpath: pathlib.Path):
        """
        TODO::This method may contain some bugs and need further testing and fixing.

        Generate and save DAG embeddings for all LogicX files in the specified directory.
        """
        import pandas
        from torch_geometric.loader import NeighborLoader
        from younger_logics_ir.modules import LogicX

        logicx_filepaths = [logicx_filepath for logicx_filepath in load_dirpath.joinpath('logicxs').iterdir()]

        create_dir(save_dirpath)

        dag_hashes = list()
        dag_embeddings = list()
        for logicx_filepath in logicx_filepaths:
            logicx = LogicX()
            logicx.load(logicx_filepath)
            dag_hashes.append(LogicX.hash(logicx))

            data = DAGDataset.process_dag_data(logicx, {'dicts': self.dicts})
            loader = NeighborLoader(
                data,
                num_neighbors=[-1] * len(self.model.encoder.layers),
                batch_size=512,
                input_nodes=None,
                subgraph_type="directional",
                directed=True
            )

            embedding_dim = self.model.encoder.node_.embedding_layer.embedding_dim
            dag_embedding = torch.zeros(embedding_dim, device=self.device_descriptor)
            node_count = 0
            for batch in loader:
                batch: DAGData = batch.to(self.device_descriptor)
                out = self.model.encoder(batch.x, batch.edge_index)               # shape: [total_nodes_in_batch, dim]
                center_embeddings = out[:batch.batch_size]                   # shape: [batch_size, dim]
                dag_embedding += center_embeddings.sum(dim=0)
                node_count += center_embeddings.shape[0]
            assert len(logicx.dag) == node_count
            dag_embedding = (dag_embedding/node_count).detach().cpu().numpy().tolist()
            dag_embeddings.append(dag_embedding)

        hsh_df = pandas.DataFrame(dag_hashes, columns=["logicx_hash"])
        hsh_df.to_csv(save_dirpath.joinpath("dag_hashes.csv"), index=False)

        emb_df = pandas.DataFrame(dag_embeddings, columns=[str(i) for i in range(embedding_dim)])
        emb_df.to_csv(save_dirpath.joinpath("dag_embeddings.csv"), index=False)
