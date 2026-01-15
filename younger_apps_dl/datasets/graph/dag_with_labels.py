#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-03-14 21:43:17
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-01-14 22:01:15
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import torch

from typing import Any, Callable, Literal

from younger_logics_ir.modules import LogicX

from younger_apps_dl.datasets import register_dataset

from .dag import DAGDataset, DAGData


@register_dataset('dag_with_labels')
class DAGWithLabelsDataset(DAGDataset):
    def __init__(
        self,
        meta_filepath: str,
        raw_dirpath: str,
        raw_filename: str,
        processed_dirpath: str,
        processed_filename: str,
        name: str = 'YADL-DAG',
        split: Literal['train', 'valid', 'test'] = 'train',
        worker_number: int = 4,
        root: str | None = None,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        log: bool = True,
        force_reload: bool = False,
        label_keys: list[str] = ['download', 'like'],
    ):
        """
        The DAGWithLabelsDataset is a dataset class for loading and processing Directed Acyclic Graphs (DAGs) with associated labels.

        All parameters are inherited from DAGDataset except for `label_keys`.

        :param label_keys: The keys of labels to be used for each graph. Defaults to ['download', 'like']. One can customize this list based on the available labels in the dataset (e.g., ['download', 'like', 'task', 'accuracy']).
        :type label_keys: list[str]
        """
        self.label_keys = label_keys
        super(DAGWithLabelsDataset, self).__init__(
            meta_filepath=meta_filepath,
            raw_dirpath=raw_dirpath,
            raw_filename=raw_filename,
            processed_dirpath=processed_dirpath,
            processed_filename=processed_filename,
            name=name,
            split=split,
            worker_number=worker_number,
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            log=log,
            force_reload=force_reload,
        )

    @property
    def arguments(self) -> dict[str, Any]:
        args = super(DAGWithLabelsDataset, self).arguments
        args.update({
            'label_keys': self.label_keys,
        })
        return args

    @classmethod
    def process_dag_data(
        cls,
        logicx: LogicX,
        **arguments,
    ) -> DAGData:
        """
        Extends parent's process_dag_data by adding label information (y) to the DAGData.
        Requires 'label_keys' in arguments.
        """
        dag_data = super(DAGWithLabelsDataset, cls).process_dag_data(logicx, **arguments)
        dag_data.y = cls.process_dag_y(logicx, arguments['label_keys'])
        return dag_data

    @classmethod
    def process_dag_y(cls, logicx: LogicX, label_keys: list[str]) -> torch.Tensor:
        label_values = list()
        for label_key in label_keys:
            if label_key in logicx.dag.graph['labels']:
                label_values.append(logicx.dag.graph['labels'][label_key])
        y = torch.tensor(label_values, dtype=torch.float)
        return y
