#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-03-14 21:43:17
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-01-26 16:18:22
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import os
import torch
import pathlib
import networkx
import multiprocessing

from typing import Any, Callable, Literal

from torch_geometric.data import Data, Dataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr, GlobalStorage
from torch_geometric.utils import is_sparse

from younger.commons.io import load_json, load_pickle, create_dir
from younger.commons.utils import split_sequence
from younger.commons.progress import MultipleProcessProgressManager

from younger_logics_ir.modules import LogicX

from younger_apps_dl.datasets import register_dataset
from younger_apps_dl.commons.cache import YADL_CACHE_ROOT
from younger_apps_dl.commons.logging import logger


class DAGData(Data):
    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == 'level': # Directed Acyclic Graph Generation Level
            return 0

        if is_sparse(value) and 'adj' in key:
            return (0, 1)
        elif 'index' in key or key == 'face':
            return -1
        else:
            return 0

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == 'level': # Directed Acyclic Graph Generation Level
            return 0

        if 'batch' in key and isinstance(value, torch.Tensor):
            return int(value.max()) + 1
        elif 'index' in key or key == 'face':
            return self.num_nodes
        else:
            return 0


torch.serialization.add_safe_globals([DAGData, DataEdgeAttr, DataTensorAttr, GlobalStorage])


@register_dataset('dag')
class DAGDataset(Dataset):

    @property
    def raw_path(self):
        return self.raw_paths[0]

    @property
    def processed_path(self):
        return self.processed_paths[0]

    @property
    def raw_file_names(self):
        return [f'{self.raw_filename}']
        # return [f'{hash}' for hash in self.hashs]

    @property
    def processed_file_names(self):
        return [f'{self.processed_filename}']
        # return [f'{hash}' for hash in self.hashs]

    @property
    def raw_dir(self) -> str:
        return self.raw_dirpath

    @property
    def processed_dir(self) -> str:
        return self.processed_dirpath

    def len(self) -> int:
        return len(self.hashs)

    def get(self, index: int) -> DAGData:
        dag_data = self.all_dag_data[index]
        return dag_data

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
    ):
        # The class will not use `root`.
        # The `self.download()` will not be implemented.
        # This dataset assumes that the raw files are already present in `raw_dirpath`.
        # All data with LogicX format will be processed into DAGData and saved in `processed_dirpath`.

        self.meta_filepath = meta_filepath
        self.raw_dirpath = raw_dirpath
        self.raw_filename = raw_filename
        self.processed_dirpath = processed_dirpath
        self.processed_filename = processed_filename

        self.name = name
        self.split = split
        self.worker_number = worker_number

        self.cache_dirpath = YADL_CACHE_ROOT.joinpath('datasets', 'graph', 'dag', self.name, self.split)
        create_dir(self.cache_dirpath)

        self.meta = self.__class__.load_meta(self.meta_filepath)
        self.dicts = self.__class__.load_dicts(self.meta)
        self.hashs = self.__class__.load_hashs(self.meta)

        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)

        logger.info('Loading Processed File')
        self.all_dag_data: list[DAGData] = torch.load(self.processed_path)
        logger.info(f'Loaded {len(self.all_dag_data)} samples for split="{self.split}".')

    @property
    def arguments(self) -> dict[str, Any]:
        """
        This method returns a dictionary of arguments used for processing DAG data: `self.__class__.process_dag_data(dag, **self.arguments)`.
        """
        return {
            'dicts': self.dicts,
        }

    @classmethod
    def load_meta(cls, meta_filepath: str) -> dict[str, Any]:
        meta: dict[str, Any] = load_json(meta_filepath)
        return meta

    @classmethod
    def load_dicts(cls, meta: dict[str, Any]) -> dict[Literal['i2t', 't2i'], dict[int, str] | dict[str, int]]:
        assert 'node_types' in meta
        assert isinstance(meta['node_types'], list)

        dicts: dict[Literal['i2t', 't2i'], dict[int, str] | dict[str, int]] = dict()
        dicts['i2t'] = dict()
        dicts['t2i'] = dict()

        node_types = ['__UNK__'] + ['__MASK__'] + meta['node_types']
        for i, t in enumerate(node_types):
            dicts['i2t'][i] = t
            dicts['t2i'][t] = i
        return dicts

    @classmethod
    def load_hashs(cls, meta: dict[str, Any]) -> list[str]:
        assert 'item_names' in meta
        assert isinstance(meta['item_names'], list)
        hashs = sorted(meta['item_names'])
        return hashs

    def _process_chunk_(self, parameter: tuple[list[str], MultipleProcessProgressManager]) -> pathlib.Path:
        sdags_chunk, progress_manager = parameter
        dag_datas_chunk: list[DAGData] = list()
        for sdag in sdags_chunk:
            dag = LogicX.loads_dag(sdag)
            dag_data = self.__class__.process_dag_data(dag, **self.arguments)
            dag_datas_chunk.append(dag_data)
            progress_manager.update(1)

        chunk_filepath = self.cache_dirpath.joinpath(f'_process_chunk_{os.getpid()}_{multiprocessing.current_process()._identity[0]}.pt')
        torch.save(dag_datas_chunk, chunk_filepath)
        progress_manager.done()
        return chunk_filepath

    def process(self):
        progress_manager = MultipleProcessProgressManager(percent=0.1)
        hash2sdag: dict[str, str] = load_pickle(self.raw_path)
        sdags = [hash2sdag[hash] for hash in self.hashs]
        chunk_count = self.worker_number
        sdags_chunks: list[list[str]] = split_sequence(sdags, chunk_count)
        chunks = [(sdags_chunk, progress_manager) for sdags_chunk in sdags_chunks]
        dag_datas: list[DAGData] = list()
        chunk_filepaths: list[pathlib.Path] = []

        with progress_manager.progress(total=len(self.hashs), chunks=len(chunks), desc="Processing DAGs"):
            with multiprocessing.Pool(self.worker_number) as pool:
                for chunk_filepath in pool.imap_unordered(self._process_chunk_, chunks):
                    chunk_filepaths.append(chunk_filepath)
                    dag_datas.extend(torch.load(chunk_filepath))
                    os.remove(chunk_filepath)

        torch.save(dag_datas, self.processed_path)

    @classmethod
    def process_dag_data(
        cls,
        dag: networkx.DiGraph,
        **arguments,
    ) -> DAGData:
        """
        This method processes a LogicX DAG into a DAGData object.
        Key 'dicts' must contained in arguments: dict[Literal['i2t', 't2i'], dict[int, str] | dict[str, int]]

        """
        nxids = sorted(list(dag.nodes)) # NetworkX IDs
        pgids = list(range(dag.number_of_nodes())) # PyGeometric IDs
        nxid2pgid = dict(zip(nxids, pgids))
        # [B, A, C, D]
        # >>> print(list(sorted(dag.nodes)))
        # [A, B, C, D]
        # >>> print(nxid2pgid)
        # {A: 0, B: 1, C: 2, D: 3}

        x = cls.process_dag_x(dag, arguments['dicts'], nxid2pgid)
        edge_index = cls.process_dag_edge_index(dag, nxid2pgid)
        if dag.graph['level']:
            level = cls.process_dag_level(dag, nxid2pgid)
            dag_data = DAGData(x=x, edge_index=edge_index, level=level)
        else:
            dag_data = DAGData(x=x, edge_index=edge_index)
        return dag_data

    @classmethod
    def process_dag_x(cls, dag: networkx.DiGraph, dicts: dict[Literal['i2t', 't2i'], dict[int, str] | dict[str, int]], nxid2pgid: dict[str, int]) -> torch.Tensor:
        # Shape: [#Node, 1]

        # ID in DAG
        # NX Node ID sorted by PG Node ID
        node_indices_in_dag: list[str] = sorted(list(dag.nodes), key=lambda x: nxid2pgid[x])

        # ID in Dict
        node_indices_in_dict = list()
        for node_index_in_dag in node_indices_in_dag:
            node_uuid = dag.nodes[node_index_in_dag]['node_uuid']
            if node_uuid in dicts['t2i']:
                node_index_in_dict = [dicts['t2i'][node_uuid]]
            else:
                node_index_in_dict = [dicts['t2i']['__UNK__']]
            node_indices_in_dict.append(node_index_in_dict)
        x = torch.tensor(node_indices_in_dict, dtype=torch.long)

        return x

    @classmethod
    def process_dag_edge_index(cls, dag: networkx.DiGraph, nxid2pgid: dict[str, int]) -> torch.Tensor:
        # Shape: [2, #Edge]
        edge_index = torch.empty((2, dag.number_of_edges()), dtype=torch.long)
        for index, (src, dst) in enumerate(list(dag.edges)):
            edge_index[0, index] = nxid2pgid[src]
            edge_index[1, index] = nxid2pgid[dst]
        return edge_index

    @classmethod
    def process_dag_level(cls, dag: networkx.DiGraph, nxid2pgid: dict[str, int]) -> torch.Tensor:
        # Shape: [#Node, 1]

        level = list()
        node_indices_in_dag: list[str] = sorted(list(dag.nodes), key=lambda x: nxid2pgid[x])
        for index, node_index_in_dag in enumerate(node_indices_in_dag):
            level.append([dag.nodes[node_index_in_dag]['level']])
        level = torch.tensor(level, dtype=torch.long)
        return level
