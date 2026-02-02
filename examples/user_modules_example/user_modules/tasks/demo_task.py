#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2026-02-01 20:31:04
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-02-02 09:19:27
# Copyright (c) 2026 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


"""
A simple demo task example.
"""

import torch
from typing import Tuple, List, Callable
from pydantic import BaseModel, Field

from younger_apps_dl.tasks import BaseTask, register_task
from younger_apps_dl.engines import StandardTrainer, StandardTrainerOptions, StandardEvaluator, StandardEvaluatorOptions, StandardPredictor, StandardPredictorOptions
from younger_apps_dl.commons.logging import logger

from ..models import SimpleMLP


class SimpleDemoModelOptions(BaseModel):
    """Model configuration options (flat, reusable)."""
    input_dim: int = Field(128, description='Model input dimension')
    hidden_dim: int = Field(256, description='Model hidden layer dimension')
    output_dim: int = Field(10, description='Model output dimension')


class SimpleDemoTaskOptions(BaseModel):
    """SimpleDemoTask configuration options."""
    """Training configuration options (flat, reusable)."""
    learning_rate: float = Field(0.001, description='Learning rate')
    batch_size: int = Field(32, description='Batch size')
    num_epochs: int = Field(10, description='Number of epochs to train')
    train_samples: int = Field(256, description='Number of synthetic training samples')
    valid_samples: int = Field(64, description='Number of synthetic validation samples')

    # Model configuration
    model: SimpleDemoModelOptions = Field(default_factory=SimpleDemoModelOptions, description='Model configuration options')

    # Trainer configuration
    trainer: StandardTrainerOptions | None = Field(None, description='Standard trainer options (optional)')
    evaluator: StandardEvaluatorOptions | None = Field(None, description='Standard evaluator options (optional)')
    predictor: StandardPredictorOptions | None = Field(None, description='Standard predictor options (optional)')
    # Other configuration
    message: str = Field('Hello from SimpleDemoTask!', description='Demo message')


@register_task('optional', 'simple_demo')
class SimpleDemoTask(BaseTask[SimpleDemoTaskOptions]):
    """
    A simple demo task

    This task demonstrates:
    1. Defining task configuration
    2. Using a custom model
    3. Implementing logic for various stages
    """

    OPTIONS = SimpleDemoTaskOptions

    STAGE_REQUIRED_OPTION = {
        'preprocess': [],
        'train': ['trainer', 'model'],
        'evaluate': ['evaluator', 'model'],
        'predict': ['predictor'],
        'postprocess': [],
    }

    def _preprocess_(self):
        """Preprocessing stage."""
        logger.info("Starting preprocessing...")
        logger.info(f"Message: {self.options.message}")
        logger.info("Preprocessing completed!")

    def _train_(self):
        """Training stage."""
        self.train_dataset = self._build_dataset_(self.options.train_samples)
        self.valid_dataset = self._build_dataset_(self.options.valid_samples)
        self.model = self._build_model_()
        self.optimizer = self._build_optimizer_(self.model)
        self.scheduler = self._build_scheduler_(self.optimizer)

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
            dataloader_type='pth'
        )

    def _evaluate_(self):
        """Evaluation stage."""
        logger.info("Starting evaluation...")
        self.test_dataset = self._build_dataset_(self.options.valid_samples)
        self.model = self._build_model_()
        evaluator = StandardEvaluator(self.options.evaluator)
        evaluator.run(
            self.model,
            self.test_dataset,
            self._test_fn_,
            initialize_fn=self._initialize_fn_,
            dataloader_type='pth'
        )

        logger.info("Evaluation completed!")

    def _predict_(self):
        """Prediction stage."""
        logger.info("Starting prediction...")
        self.model = self._build_model_()
        predictor = StandardPredictor(self.options.predictor)
        predictor.run(
            self.model,
            predict_raw_fn=self._predict_raw_fn_,
            initialize_fn=self._initialize_fn_,
        )
        logger.info("Prediction completed!")

    def _postprocess_(self):
        """Postprocess stage."""
        logger.info("Starting postprocess...")
        logger.info("Postprocess completed!")

    def _build_model_(self) -> torch.nn.Module:
        """Build model helper."""

        logger.info("Building model: SimpleMLP")

        model = SimpleMLP(
            input_dim=self.options.model.input_dim,
            hidden_dim=self.options.model.hidden_dim,
            output_dim=self.options.model.output_dim,
        )

        logger.info(f"Model built: {type(model).__name__}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model

    def _build_optimizer_(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=self.options.learning_rate)
        return optimizer

    def _build_scheduler_(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-6)
        return scheduler

    def _build_dataset_(self, sample_count: int) -> torch.utils.data.TensorDataset:
        inputs = torch.randn(sample_count, self.options.model.input_dim)
        targets = torch.randint(0, self.options.model.output_dim, (sample_count,))
        return torch.utils.data.TensorDataset(inputs, targets)

    def _initialize_fn_(self, model: torch.nn.Module) -> None:
        self.model = model

    def _train_fn_(self, minibatch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[List[str], List[torch.Tensor], List[Callable[[float], str]]]:
        inputs, targets = minibatch
        output = self.model(inputs)
        loss = torch.nn.functional.cross_entropy(output, targets)
        loss.backward()
        return self._compute_metrics_(loss.item())

    def _valid_fn_(self, dataloader: torch.utils.data.DataLoader) -> Tuple[List[str], List[torch.Tensor], List[Callable[[float], str]]]:
        losses = list()
        with torch.no_grad():
            for inputs, targets in dataloader:
                output = self.model(inputs)
                loss = torch.nn.functional.cross_entropy(output, targets)
                losses.append(loss)
        loss = torch.stack(losses).mean()
        return self._compute_metrics_(loss.item())

    def _test_fn_(self, dataloader: torch.utils.data.DataLoader) -> Tuple[List[str], List[torch.Tensor], List[Callable[[float], str]]]:
        return self._valid_fn_(dataloader)

    def _predict_raw_fn_(self, *args, **kwargs):
        logger.info("Predict raw fn is not implemented for SimpleDemoTask.")
        raise NotImplementedError("Predict raw fn is not implemented for SimpleDemoTask")

    def _on_update_fn_(self, metrics: Tuple[List[str], List[torch.Tensor], List[Callable[[float], str]]]) -> None:
        self.optimizer.step()
        self.optimizer.zero_grad()
        metric_names, metric_values, _ = metrics
        metric_dict = dict(zip(metric_names, metric_values))
        if self.scheduler is not None and 'loss' in metric_dict:
            self.scheduler.step(metric_dict['loss'])

    def _compute_metrics_(self, loss: float) -> Tuple[List[str], List[torch.Tensor], List[Callable[[float], str]]]:
        metrics = [
            ('loss', loss, lambda x: f'{x:.4f}'),
        ]
        metric_names_list, metric_values_list, metric_formats_list = zip(*metrics)
        return list(metric_names_list), list(metric_values_list), list(metric_formats_list)


if __name__ == '__main__':
    # Simple test for SimpleDemoTask
    print("Test SimpleDemoTask...")

    # Create 
    options = SimpleDemoTaskOptions(
        input_dim=10,
        hidden_dim=20,
        output_dim=5,
    )

    task = SimpleDemoTask(options)
    # Test each stage
    print("\n--- Test Preprocess ---")
    task.preprocess()
    print("\n--- Test Train ---")
    task.train()
    print("\n--- Test Evaluate ---")
    task.evaluate()
    print("\n--- Test Predict ---")
    task.predict()
    print("\n--- Test Postprocess ---")
    task.postprocess()
    print("\n✓ All Test Passed!")
