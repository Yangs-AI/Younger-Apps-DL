#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-03-31 21:20:17
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-01-15 02:09:17
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

import torch
import torch.utils.data

from typing import Any, Callable
from collections import OrderedDict
from torch_geometric.data import Batch, Data

from younger.datasets.modules import Instance, Network
from younger.datasets.utils.translation import get_complete_attributes_of_node

from younger.applications.models import NAPPGATVaryV1
from younger.applications.datasets import ArchitectureDataset
from younger.applications.tasks.base_task import YoungerTask


def infer_cluster_num(dataset: ArchitectureDataset) -> int:
    total_node_num = 0
    for data in dataset:
        total_node_num += data.num_nodes
    return int(total_node_num / len(dataset))


class PerformancePrediction(YoungerTask):
    def __init__(self, custom_config: dict, device_descriptor: torch.device) -> None:
        super().__init__(custom_config, device_descriptor)
        self.build_config(custom_config)
        self.build()

    def build_config(self, custom_config: dict):
        # Dataset
        dataset_config = dict()
        custom_dataset_config = custom_config.get('dataset', dict())
        dataset_config['train_dataset_dirpath'] = custom_dataset_config.get('train_dataset_dirpath', None)
        dataset_config['valid_dataset_dirpath'] = custom_dataset_config.get('valid_dataset_dirpath', None)
        dataset_config['test_dataset_dirpath'] = custom_dataset_config.get('test_dataset_dirpath', None)
        dataset_config['metric_feature_get_type'] = custom_dataset_config.get('metric_feature_get_type', 'mean') # 'none','mean','rand'
        dataset_config['worker_number'] = custom_dataset_config.get('worker_number', 4) 
        dataset_config['node_dict_size'] = custom_dataset_config.get('node_dict_size', None) 
        dataset_config['task_dict_size'] = custom_dataset_config.get('task_dict_size', None) 

        # Model
        model_config = dict()
        custom_model_config = custom_config.get('model', dict())
        model_config['node_dim'] = custom_model_config.get('node_dim', 512)
        model_config['task_dim'] = custom_model_config.get('task_dim', 512)
        model_config['hidden_dim'] = custom_model_config.get('hidden_dim', 512)
        model_config['readout_dim'] = custom_model_config.get('readout_dim', 256)
        model_config['cluster_num'] = custom_model_config.get('cluster_num', None)

        # Optimizer
        optimizer_config = dict()
        custom_optimizer_config = custom_config.get('optimizer', dict())
        optimizer_config['learning_rate'] = custom_optimizer_config.get('learning_rate', 1e-3)
        optimizer_config['weight_decay'] = custom_optimizer_config.get('weight_decay', 1e-1)

        # API
        api_config = dict()
        custom_api_config = custom_config.get('api', dict())
        api_config['meta_filepath'] = custom_api_config.get('meta_filepath', None)
        api_config['onnx_model_dirpath'] = custom_api_config.get('onnx_model_dirpath', list())

        config = dict()
        config['dataset'] = dataset_config
        config['model'] = model_config
        config['optimizer'] = optimizer_config
        config['api'] = api_config
        self.config = config

    def build(self):
        if self.config['dataset']['train_dataset_dirpath']:
            self.train_dataset = ArchitectureDataset(
                self.config['dataset']['train_dataset_dirpath'],
                task_dict_size=self.config['dataset']['task_dict_size'],
                node_dict_size=self.config['dataset']['node_dict_size'],
                metric_feature_get_type=self.config['dataset']['metric_feature_get_type'],
                worker_number=self.config['dataset']['worker_number']
            )
        else:
            self.train_dataset = None
        if self.config['dataset']['valid_dataset_dirpath']:
            self.valid_dataset = ArchitectureDataset(
                self.config['dataset']['valid_dataset_dirpath'],
                task_dict_size=self.config['dataset']['task_dict_size'],
                node_dict_size=self.config['dataset']['node_dict_size'],
                metric_feature_get_type=self.config['dataset']['metric_feature_get_type'],
                worker_number=self.config['dataset']['worker_number']
            )
        else:
            self.valid_dataset = None
        if self.config['dataset']['test_dataset_dirpath']:
            self.test_dataset = ArchitectureDataset(
                self.config['dataset']['test_dataset_dirpath'],
                task_dict_size=self.config['dataset']['task_dict_size'],
                node_dict_size=self.config['dataset']['node_dict_size'],
                metric_feature_get_type=self.config['dataset']['metric_feature_get_type'],
                worker_number=self.config['dataset']['worker_number']
            )
        else:
            self.test_dataset = None

        if self.train_dataset:
            self.logger.info(f'-> Nodes Dict Size: {len(self.train_dataset.x_dict["n2i"])}')
            self.logger.info(f'-> Tasks Dict Size: {len(self.train_dataset.y_dict["t2i"])}')

        if self.config['model']['cluster_num'] is None:
            self.config['model']['cluster_num'] = infer_cluster_num(self.train_dataset)
            self.logger.info(f'Cluster Number Not Specified! Infered Number: {self.config["model"]["cluster_num"]}')
        else:
            self.logger.info(f'-> Cluster Number: {self.config["model"]["cluster_num"]}')

        self.model = NAPPGATVaryV1(
            node_dict=self.train_dataset.x_dict['n2i'],
            task_dict=self.train_dataset.y_dict['t2i'],
            node_dim=self.config['model']['node_dim'],
            task_dim=self.config['model']['task_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            readout_dim=self.config['model']['readout_dim'],
            cluster_num=self.config['model']['cluster_num'],
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['optimizer']['learning_rate'], weight_decay=self.config['optimizer']['weight_decay'])

    def train(self, minibatch: Any) -> tuple[torch.Tensor, OrderedDict[str, tuple[torch.Tensor, Callable | None]]]:
        minibatch = minibatch.to(self.device_descriptor)
        y = minibatch.y.reshape(len(minibatch), -1)
        tasks = y[:, 0].long()
        metrics = y[:, 1]
        outputs, global_pooling_loss = self.model(minibatch.x[:, 0], minibatch.edge_index, minibatch.batch, tasks)
        main_loss = torch.nn.functional.mse_loss(outputs.reshape(-1), metrics)
        loss = main_loss + global_pooling_loss
        logs = OrderedDict({
            'Total-Loss': (loss, lambda x: f'{x:.4f}'),
            'REG-Loss (MSE)': (main_loss, lambda x: f'{x:.4f}'),
            'Cluster-loss': (global_pooling_loss, lambda x: f'{x:.4f}'),
        })
        return loss, logs

    def eval(self, minibatch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        # Return Output & Golden
        minibatch = minibatch.to(self.device_descriptor)
        y = minibatch.y.reshape(len(minibatch), -1)
        tasks = y[:, 0].long()
        metrics = y[:, 1]
        outputs, _ = self.model(minibatch.x[:, 0], minibatch.edge_index, minibatch.batch, tasks)
        return outputs.reshape(-1), metrics

    def eval_calculate_logs(self, all_outputs: list[torch.Tensor], all_goldens: list[torch.Tensor]) -> OrderedDict[str, tuple[torch.Tensor, Callable | None]]:
        all_outputs = torch.stack(all_outputs).reshape(-1)
        all_goldens = torch.stack(all_goldens).reshape(-1)
        mae = torch.nn.functional.l1_loss(all_outputs, all_goldens, reduction='mean')
        mse = torch.nn.functional.mse_loss(all_outputs, all_goldens, reduction='mean')
        rmse = torch.sqrt(mse)
        logs = OrderedDict({
            'MAE': (mae, lambda x: f'{x:.4f}'),
            'MSE': (mse, lambda x: f'{x:.4f}'),
            'RMSE': (rmse, lambda x: f'{x:.4f}'),
        })
        return logs

    def api(self, **kwargs):
        meta_filepath = self.config['api']['meta_filepath']
        onnx_model_dirpath = self.config['api']['onnx_model_dirpath']
        assert meta_filepath, f'No Meta File.'
        assert onnx_model_dirpath, f'No ONNX Dir.'

        self.logger.info(f'  v Loading Meta ...')
        meta = ArchitectureDataset.load_meta(meta_filepath)
        x_dict = ArchitectureDataset.get_x_dict(meta, node_dict_size=self.config['dataset']['node_dict_size'])
        y_dict = ArchitectureDataset.get_y_dict(meta, task_dict_size=self.config['dataset']['task_dict_size'])
        self.logger.info(f'    -> Tasks Dict Size: {len(x_dict)}')
        self.logger.info(f'    -> Nodes Dict Size: {len(y_dict)}')
        self.logger.info(f'  ^ Built.')

        self.logger.info(f'  v Loading ONNX Models')
        datas = list()
        onnx_model_filenames = list()
        for onnx_model_filepath in onnx_model_dirpath.iterdir():
            onnx_model_filenames.append(onnx_model_filepath.name)
            instance = Instance(onnx_model_filepath)
            standardized_graph = Network.standardize(instance.network.graph)
            for node_index in standardized_graph.nodes():
                operator = standardized_graph.nodes[node_index]['features']['operator']
                attributes = standardized_graph.nodes[node_index]['features']['attributes']
                standardized_graph.nodes[node_index]['features']['attributes'] = get_complete_attributes_of_node(attributes, operator['op_type'], operator['domain'], meta['max_inclusive_version'])
            standardized_graph.graph.clear()
            data = ArchitectureDataset.get_data(standardized_graph, x_dict, y_dict, feature_get_type='none')
            datas.append(data)
        minibatch = Batch.from_data_list(datas)
        self.logger.info(f'  ^ Loaded. Total - {len(datas)}.')

        self.model.eval()
        self.logger.info(f'  -> Interact Test Begin ...')

        with torch.no_grad():
            minibatch: Data = minibatch.to(self.device_descriptor)
            output, _ = self.model(minibatch.x, minibatch.edge_index, minibatch.batch)

            for onnx_model_filename, output_value in zip(onnx_model_filenames, output):
                self.logger.info(f'  -> Result - {onnx_model_filename}: {output_value}')