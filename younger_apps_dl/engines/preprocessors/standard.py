#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-03-14 21:43:17
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-01-24 22:34:22
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import os
import tqdm
import numpy
import random
import pathlib
import networkx
import collections
import multiprocessing

from typing import Any, Literal
from pydantic import BaseModel, Field

from younger.commons.io import save_json, save_pickle
from younger.commons.utils import split_sequence
from younger.commons.progress import MultipleProcessProgressManager

from younger_logics_ir.modules import LogicX

from younger_apps_dl.commons.logging import logger

from younger_apps_dl.engines import BaseEngine, register_engine


class StandardPreprocessorOptions(BaseModel):
    load_dirpath: pathlib.Path = Field(..., description='Directory path to load LogicX\'s.')
    save_dirpath: pathlib.Path = Field(..., description='Directory path to save LogicX\'s.')

    # Subgraph Extraction Control
    subgraph: bool = Field(True, description='Whether to extract subgraphs from DAGs. '
                                             'If False, full DAGs will be used directly for Training/Validation/Test splitting.')

    # Subgraph Extraction Parameters (only used when subgraph=True)
    # Begin Subgraph Extraction Parameters
    method: Literal['Random', 'Cascade', 'RandomFull', 'CascadeFull', 'Window', 'MixBasic', 'MixSuper'] | None = Field(None, description='DAG splitting method. '
                                                                                               '\'Random\' selects a random node as center; BFS is used to expand the subgraph, retaining a random subset of nodes at each depth. '
                                                                                               '\'RandomFull\' is similar, but retains all nodes at each BFS depth. '
                                                                                               '\'Cascade\' restricts the expansion to ancestors or descendants of the center node, retaining a random subset at each depth. '
                                                                                               '\'CascadeFull\' is the full-retention version of \'Cascade\', preserving all nodes at each BFS depth.'
                                                                                               '\'Window\' randomly selects a graph and identifies nodes at a specific level; then performs a `split_scales`-step backward traversal, incorporating all traversed nodes and edges into the subgraph.'
                                                                                               '\'MixBasic\' uniformly samples one of the following methods for each subgraph: \'Random\', \'RandomFull\', \'Cascade\', or \'CascadeFull\'. '
                                                                                               '\'MixSuper\' extends MixBasic by additionally including \'Window\' in the set of candidate methods, sampling uniformly among all five.')

    split_scales: list[int] | None = Field(None, description='List of node counts to include in subgraph splits expanded from central nodes. Each value specifies a different subgraph split scale to generate.')
    split_count: int | None = Field(None, description='Number of subgraph splits to generate per central node.')
    split_tries: int | None = Field(None, description='Maximum number of attempts to generate `split_count` valid subgraphs (e.g., avoiding duplicates or undersized splits).')
    split_limit: int | None = Field(None, description='Maximum allowed size (in number of nodes) for a subgraph split. If a candidate subgraph exceeds this size, it will be discarded. '
                                                      'This limit only applies to methods that involve the \'Window\' extraction strategy, including \'Window\' and mixed strategies \'MixSuper\'.')
    # End Subgraph Extraction Parameters

    training_dataset_size: int = Field(..., description='Number of items to include in the training set.')
    validation_dataset_size: int = Field(..., description='Number of items to include in the validation set.')
    test_dataset_size: int = Field(..., description='Number of items to include in the test set.')

    min_dag_size: int | None = Field(None, ge=0, description='Minimum number of nodes a full dag must have to be considered for processing. '
                                                             'DAGs smaller than this value will be excluded. '
                                                             'Set to `None` to disable this filter.')
    max_dag_size: int | None = Field(None, ge=0, description='Maximum number of nodes a full dag must have to be considered for processing. '
                                                             'DAGs larger than this value will be excluded. '
                                                             'Set to `None` to disable this filter.')

    level: bool = Field(True, description='Whether to mark levels for nodes in DAGs.')

    uuid_threshold: int | None = Field(None, ge=0, description='Occurence threshold to ignore uuid, lower than threshold will be discarded.')
    seed: int = Field(16861, ge=0, description='Random seed for deterministic behavior during sampling.')
    worker_number: int = Field(1, ge=1, description='Number of parallel worker processes. Set to 1 for single-process mode.')


# Global variables for sharing data across worker processes
global_valid_logicx_filepaths = None
global_all_nid2nod = None
global_all_nod2nids = None

def initialize(valid_logicx_filepaths, all_nid2nod, all_nod2nids):
    """Initialize global variables in worker processes."""
    global global_valid_logicx_filepaths, global_all_nid2nod, global_all_nod2nids
    global_valid_logicx_filepaths = valid_logicx_filepaths
    global_all_nid2nod = all_nid2nod
    global_all_nod2nids = all_nod2nids


@register_engine('preprocessor', 'standard')
class StandardPreprocessor(BaseEngine[StandardPreprocessorOptions]):
    OPTIONS = StandardPreprocessorOptions

    @staticmethod
    def _load_logicx_by_chunk_(parameters: tuple[list[pathlib.Path], int, 'MultipleProcessProgressManager', int | None, int | None]) -> tuple[list[pathlib.Path], list[str], dict[str, dict[int, set[str]]], list[dict[str, int]], list[dict[int, list[str]]]]:
        """
        Load and filter a chunk of LogicX files in parallel.

        Parameters:
        parameters: tuple containing:
            - logicx_filepaths: list of LogicX file paths to load
            - seed: random seed for deterministic behavior
            - progress_manager: Progress manager for sending updates
            - min_dag_size: minimum DAG size filter
            - max_dag_size: maximum DAG size filter

        Returns:
            - list of file paths to load valid LogicX objects
            - list of valid LogicX hashes
            - dict of all UUID positions across loaded LogicX objects
            - dict mapping logicx index to node index to node order
            - dict mapping logicx index to node order to list of node indices
        """
        logicx_filepaths, seed, progress_manager, min_dag_size, max_dag_size = parameters
        random.seed(seed)
        numpy.random.seed(seed)

        valid_logicx_filepaths: list[pathlib.Path] = list() # [path1, path2, ...] - paths to LogicX files
        valid_logicx_hashes: list[str] = list() # [logicx1_hash, logicx2_hash, ...]
        all_uuid_positions: dict[str, dict[int, set[str]]] = dict() # {uuid: {local_index: set[node_index]}}
        all_nid2nod: list[dict[str, int]] = list() # per-logicx mapping  list[{node_index: order}]
        all_nod2nids: list[dict[int, list[str]]] = list() # per-logicx mapping  list[{order: list[node_index]}]

        for logicx_filepath in logicx_filepaths:
            progress_manager.update(1)
            logicx = LogicX()
            logicx.load(logicx_filepath)

            dag_size = len(logicx.dag)
            if min_dag_size is not None and dag_size < min_dag_size:
                continue
            if max_dag_size is not None and dag_size > max_dag_size:
                continue

            local_index = len(valid_logicx_filepaths)
            valid_logicx_filepaths.append(logicx_filepath)
            valid_logicx_hashes.append(logicx_filepath.name)

            # Generate UUID Positions
            for node_index in logicx.dag.nodes:
                # Generate UUID Positions and Node Orders
                # {uuid: {logicx_index: set[node_index]}}
                # uuid is the operator type
                # logicx_index is the index of logicx in logicxs
                # node_index is the index of node in logicx.dag
                # e.g., {'uuid1': {0: {'n1', 'n2'}, 2: {'n3'}}, 'uuid2': {1: {'n4'}}}
                # Thus, uuid1 appears in logicx 0 at nodes n1 and n2, and in logicx 2 at node n3; uuid2 appears in logicx 1 at node n4.
                uuid = logicx.dag.nodes[node_index]['node_uuid']
                uuid_positions = all_uuid_positions.get(uuid, dict())
                node_indices = uuid_positions.get(local_index, set())
                node_indices.add(node_index)
                uuid_positions[local_index] = node_indices
                all_uuid_positions[uuid] = uuid_positions

            # From Topological Order, Generate Node Index to Node Order & Node Order to Node Indices
            # Node Index: The Original Node Index in DAG
            # Node Order: The Longest Path from Root to Node
            all_nid2nod.append(dict())
            all_nod2nids.append(dict())
            for node_index in networkx.topological_sort(logicx.dag):
                predecessors = logicx.dag.predecessors(node_index)
                all_nid2nod[local_index][node_index] = max([all_nid2nod[local_index][predecessor] + 1 for predecessor in predecessors] + [0])
                all_nod2nids[local_index].setdefault(all_nid2nod[local_index][node_index], list()).append(node_index)

        progress_manager.done()
        return valid_logicx_filepaths, valid_logicx_hashes, all_uuid_positions, all_nid2nod, all_nod2nids

    @staticmethod
    def _extract_subgraphs_for_uuid_(
        parameters: tuple[ str, dict[int, set[str]],
            list[int], int, int, int,
            Literal['Random', 'Cascade', 'RandomFull', 'CascadeFull', 'Window', 'MixBasic', 'MixSuper'] | None,
            bool, int, 'MultipleProcessProgressManager'
        ]
    ) -> tuple[ str, dict[int, dict[str, LogicX]], dict[int, dict[str, list[str]]] ]:
        """Extract subgraphs for a single UUID across all split scales.

        Arguments:
        parameters: tuple containing:
            - uuid: the operator UUID
            - uuid_positions: dict mapping logicx_index to node indices
            - split_scales: list of split scale values
            - split_count: number of splits to generate
            - split_tries: maximum attempts to generate splits
            - split_limit: maximum subgraph size limit
            - method: extraction method name
            - level: whether to mark levels
            - seed: seed for randomization
            - progress_manager: Progress manager for sending updates (None for single-process mode)

        Note: This function accesses global variables valid_logicx_filepaths, all_nid2nod, and all_nod2nids, which are initialized via pool initializer to avoid passing large data structures.

        Returns:
            - tuple of (uuid, uuid_dags, uuid_dag_hashes)
        """
        (
            uuid, uuid_positions,
            split_scales, split_count, split_tries, split_limit,
            method,
            level, seed, progress_manager
        ) = parameters

        random.seed(seed)
        numpy.random.seed(seed)

        uuid_dags: dict[int, dict[str, networkx.DiGraph]] = {split_scale: dict() for split_scale in split_scales} # {split_scale: {dag_hash: dag}}
        uuid_dag_hashes: dict[int, dict[str, list[str]]] = {split_scale: dict() for split_scale in split_scales} # {split_scale: {uuid: list[dag_hash]}}

        total = len(split_scales)
        # For Each Split Size:
        for split_scale in split_scales:
            progress_manager.update(1)
            # For Each Operator:
            current_tries: int = 0
            current_split_count: int = 0
            candidate_logicx_indices: set[int] = set(uuid_positions.keys())

            # Generate Subgraph Split Repeatedly
            while len(candidate_logicx_indices) != 0 and current_split_count < split_count:
                if not (current_tries < split_tries):
                    break

                current_tries += 1

                selected_logicx_index: int = int(numpy.random.choice(list(candidate_logicx_indices)))
                selected_node_index: str = str(numpy.random.choice(list(uuid_positions[selected_logicx_index])))

                active_method = method
                if method == 'MixBasic':
                    active_method = random.choice(['Random', 'Cascade', 'RandomFull', 'CascadeFull'])
                if method == 'MixSuper':
                    active_method = random.choice(['Random', 'Cascade', 'RandomFull', 'CascadeFull', 'Window'])

                if active_method == 'Window':
                    selected_node_order: int = global_all_nid2nod[selected_logicx_index][selected_node_index]
                    if selected_node_order < split_scale - 1:
                        continue
                    selected_node_indices: list[str] = global_all_nod2nids[selected_logicx_index][selected_node_order]
                    split = StandardPreprocessor.retrieve_split(
                        global_valid_logicx_filepaths[selected_logicx_index], selected_node_indices, 
                        split_scale, split_limit, active_method
                    )
                    split_size = split_scale
                else:
                    split = StandardPreprocessor.retrieve_split(
                        global_valid_logicx_filepaths[selected_logicx_index], [selected_node_index], 
                        split_scale, split_limit, active_method
                    )
                    split_size = len(split.dag)

                if level:
                    StandardPreprocessor.mark_node_levels(split.dag)
                    split.dag.graph['level'] = True
                else:
                    split.dag.graph['level'] = False
                if split_size not in split_scales:
                    continue

                dag_hash = LogicX.hash(split)
                if dag_hash in uuid_dags[split_size]:
                    continue

                split.dag.graph['origin'] = selected_logicx_index
                uuid_dags[split_size][dag_hash] = split.dag
                uuid_dag_hashes[split_size].setdefault(uuid, list()).append(dag_hash)
                current_split_count += 1

        progress_manager.done()
        return (uuid, uuid_dags, uuid_dag_hashes)

    @staticmethod
    def _mark_levels_by_chunk_(parameters: tuple[list[pathlib.Path], bool, 'MultipleProcessProgressManager']) -> list[networkx.DiGraph]:
        """
        Mark levels for a chunk of LogicX objects.

        Arguments:
        parameters: tuple containing:
            - valid_logicx_filepaths_chunk: list of filepath for each LogicX object to process
            - level_flag: whether to mark levels
            - progress_manager: Progress manager for sending updates (None for single-process mode)

        Returns:
            - list of processed DiGraph(DAG) objects
        """
        valid_logicx_filepaths_chunk, level_flag, progress_manager = parameters

        dags_chunk = list()
        total = len(valid_logicx_filepaths_chunk)
        for valid_logicx_filepath in valid_logicx_filepaths_chunk:
            progress_manager.update(1)
            if level_flag:
                logicx = LogicX()
                logicx.load(valid_logicx_filepath)
                StandardPreprocessor.mark_node_levels(logicx.dag)
                logicx.dag.graph['level'] = True
            else:
                logicx.dag.graph['level'] = False
            dags_chunk.append(logicx.dag)

        progress_manager.done()
        return dags_chunk

    def run(self) -> None:
        random.seed(self.options.seed)
        numpy.random.seed(self.options.seed)

        logger.info(f'Reading LogicX Files from {self.options.load_dirpath.absolute()} ... ')
        logger.info(f'Random Seed = {self.options.seed}')
        logger.info(f'Subgraph Extraction: {"Enabled" if self.options.subgraph else "Disabled"}')

        # Step 1: Load and Filter LogicX files
        logicx_filepaths = sorted([logicx_filepath for logicx_filepath in self.options.load_dirpath.iterdir()])
        # For Debugging Purpose, Limit to 1234 Files
        # logicx_filepaths = logicx_filepaths[:1234]
        logger.info(f'Scanning and Loading LogicX files ...')
        logger.info(f'Using {self.options.worker_number} worker(s)')

        progress_manager = MultipleProcessProgressManager(percent=0.1)

        # Prepare chunks for loading
        # Return file paths instead of full objects to avoid pipe buffer overflow
        logicx_filepaths_chunks = split_sequence(logicx_filepaths, self.options.worker_number)
        chunks = [(logicx_filepaths_chunk, self.options.seed, progress_manager, self.options.min_dag_size, self.options.max_dag_size) for logicx_filepaths_chunk in logicx_filepaths_chunks]

        valid_logicx_filepaths: list[pathlib.Path] = list()  # Paths to LogicX files
        valid_logicx_hashes: list[str] = list()
        all_uuid_positions: dict[str, dict[int, set[str]]] = dict()
        all_nid2nod: list[dict[str, int]] = list()
        all_nod2nids: list[dict[int, list[str]]] = list()
        current_global_index = 0

        with progress_manager.progress(total=len(logicx_filepaths), chunks=len(chunks), desc='Loading LogicX files'):
            with multiprocessing.Pool(processes=self.options.worker_number) as pool:
                # chunk_count = 0
                for result in pool.imap(StandardPreprocessor._load_logicx_by_chunk_, chunks):
                    chunk_valid_logicx_filepaths, chunk_valid_logicx_hashes, chunk_all_uuid_positions, chunk_all_nid2nod, chunk_all_nod2nids = result
                    # chunk_count += 1
                    # logger.info(f'[DEBUG] Main process received chunk {chunk_count}/{len(chunks)}, size={len(chunk_valid_logicx_filepaths)}')

                    # Build mapping from local (chunk) indices to new global contiguous indices
                    index_map = {local_idx: current_global_index + local_idx for local_idx in range(len(chunk_valid_logicx_filepaths))}

                    # Extend paths and hashes
                    valid_logicx_filepaths.extend(chunk_valid_logicx_filepaths)
                    valid_logicx_hashes.extend(chunk_valid_logicx_hashes)

                    # Remap and merge uuid positions
                    # {uuid: {local_index: set[node_index]}}
                    for uuid, positions in chunk_all_uuid_positions.items():
                        remapped_positions = {
                            index_map[local_idx]: nodes for local_idx, nodes in positions.items()
                        }
                        if uuid in all_uuid_positions:
                            all_uuid_positions[uuid].update(remapped_positions)
                        else:
                            all_uuid_positions[uuid] = remapped_positions

                    # Merge nid2nod and nod2nids
                    # nid2nod per-logicx mapping  list[{node_index: order}]
                    # nod2nids per-logicx mapping  list[{order: list[node_index]}]
                    all_nid2nod.extend(chunk_all_nid2nod)
                    all_nod2nids.extend(chunk_all_nod2nids)

                    current_global_index += len(chunk_valid_logicx_filepaths)

                # logger.info(f'[DEBUG] Main process finished consuming all {chunk_count} chunks')
                # logger.info(f'[DEBUG] About to exit pool context manager...')
        logger.info(f'Loaded {len(valid_logicx_filepaths)} DAGs matching size criteria.')

        # Step 2: Calculate UUID Statistics
        uuid_occurrence: dict[str, int] = dict()
        for uuid, uuid_positions in all_uuid_positions.items():
            uuid_occurrence[uuid] = sum([len(node_indices) for logicx_index, node_indices in uuid_positions.items()])
        logger.info(f'Total {len(uuid_occurrence)} Different Operators')

        ignored = set(uuid for uuid, occurrence in uuid_occurrence.items() if self.options.uuid_threshold is not None and self.options.uuid_threshold <= occurrence)
        logger.info(f'After Ignore: {len(uuid_occurrence) - len(ignored)} Different Operators')

        logger.info(f'User Specified # of Items - Training/Validation/Test = {self.options.training_dataset_size} / {self.options.validation_dataset_size} / {self.options.test_dataset_size}')

        # Step 3: Process Based on Extraction Mode
        if self.options.subgraph:
            # Subgraph Extraction Mode: Extract Subgraphs From Loaded DAGs
            # Estimate Overall Number of Splits
            # For Each Operator, For Each Split Size, Generate Specific Number of Subgraph Splits
            # Thus, Overall Number of Splits = # of Operators * # of Split Sizes * # of Splits per Operator
            # However, the actual number of splits may be lower due to duplicate splits or failed attempts.
            # Therefore, the estimated overall number of splits serves as an upper bound.
            # The actual number of splits will be determined during the splitting process.
            # Finally, the Training/Validation/Test Dataset Sizes Are Capped By The Estimated Overall Number Of Splits.
            # Note: Validation and Test sizes may be zero if the estimated overall number of splits is less than or equal to the training size.
            expected_overall_split_count = len(uuid_occurrence)*len(self.options.split_scales)*min(self.options.split_count, self.options.split_tries)
            expected_training_dataset_size = min(expected_overall_split_count, self.options.training_dataset_size)
            expected_validation_dataset_size = min(max(0, expected_overall_split_count-self.options.training_dataset_size), self.options.validation_dataset_size)
            expected_test_dataset_size = min(max(0, expected_overall_split_count-self.options.training_dataset_size-self.options.validation_dataset_size), self.options.test_dataset_size)
            logger.info(f'Expected Overal # of Splits = {expected_overall_split_count}')
            logger.info(f'Expected # of Splits - Training/Validation/Test = {expected_training_dataset_size} / {expected_validation_dataset_size} / {expected_test_dataset_size}')

            # For Each Split Size, For Each Operator, Generate Specific Number of Subgraph Splits
            logger.info(f'Splitting ...')
            dags: dict[int, dict[str, LogicX]] = {split_scale: dict() for split_scale in self.options.split_scales} # {split_scale: {dag_hash: dag}}
            dag_hashes: dict[int, dict[str, list[str]]] = {split_scale: dict() for split_scale in self.options.split_scales} # {split_scale: {uuid: list[dag_hash]}}

            # Build tasks for all UUIDs
            # Use global variables via pool initializer to avoid passing large data structures repeatedly
            progress_manager = MultipleProcessProgressManager(percent=0.1)
            tasks = [
                (
                    uuid, uuid_positions,
                    self.options.split_scales, self.options.split_count, self.options.split_tries, self.options.split_limit,
                    self.options.method,
                    self.options.level, self.options.seed, progress_manager
                )
                for uuid, uuid_positions in all_uuid_positions.items()
            ]

            logger.info(f'Using {self.options.worker_number} Worker(s) for Subgraph Extraction')
            results: list[tuple[str, dict[int, dict[str, networkx.DiGraph]], dict[int, dict[str, list[str]]]]] = list()
            with progress_manager.progress(total=len(tasks)*len(self.options.split_scales), chunks=len(tasks), desc='Extracting subgraphs'):
                # Use initializer to share data across workers, avoiding repeated data transmission
                with multiprocessing.Pool(
                    processes=self.options.worker_number,
                    initializer=initialize,
                    initargs=(valid_logicx_filepaths, all_nid2nod, all_nod2nids)
                ) as pool:
                    for result in pool.imap_unordered(StandardPreprocessor._extract_subgraphs_for_uuid_, tasks):
                        results.append(result)
            logger.info(f'Subgraph Extraction Completed.')

            # split.dag
            # {split_scale: {dag_hash: dag}}
            # {split_scale: {uuid: list[dag_hash]}}
            # Merge Results and Restore Origin Hashes (Common for Both Modes)
            for uuid, uuid_dags, uuid_dag_hashes in results:
                for split_scale in self.options.split_scales:
                    for dag_hash, dag in uuid_dags[split_scale].items():
                        # Restore actual origin filename
                        origin = int(dag.graph['origin'])
                        dag.graph['origin'] = valid_logicx_hashes[origin]
                        dags[split_scale][dag_hash] = dag
                    if uuid_dag_hashes[split_scale]:
                        dag_hashes[split_scale].update(uuid_dag_hashes[split_scale])

            items_with_hashes = [
                (dag_hash, dags[split_scale][dag_hash])
                for split_scale, dag_hashes_at_split_scale in dag_hashes.items()
                for uuid, uuid_dag_hashes_at_split_scale in dag_hashes_at_split_scale.items()
                for dag_hash in uuid_dag_hashes_at_split_scale
            ]
        else:
            # Full DAG Mode: Use Full DAGs Directly Without Subgraph Extraction
            logger.info(f'Marking levels for full DAGs ...')
            progress_manager = MultipleProcessProgressManager(percent=0.1)

            # Prepare chunks for marking
            chunk_number = self.options.worker_number * 4
            valid_logicx_filepaths_chunks = split_sequence(valid_logicx_filepaths, chunk_number)
            chunks = [(valid_logicx_filepaths_chunk, self.options.level, progress_manager) for valid_logicx_filepaths_chunk in valid_logicx_filepaths_chunks]

            dags = list()
            with progress_manager.progress(total=len(valid_logicx_filepaths), chunks=len(chunks), desc='Marking levels'):
                with multiprocessing.Pool(processes=self.options.worker_number) as pool:
                    for dags_chunk in pool.imap(StandardPreprocessor._mark_levels_by_chunk_, chunks):
                        dags.extend(dags_chunk)

            items_with_hashes = [(dag_hash, dag) for dag_hash, dag in zip(dag_hashes, dags)]

        # Step 4: Shuffle and Split into Train/Validation/Test
        random.shuffle(items_with_hashes)

        exact_training_dataset_size = min(self.options.training_dataset_size, len(items_with_hashes))
        exact_validation_dataset_size = min(self.options.validation_dataset_size, len(items_with_hashes) - exact_training_dataset_size)
        exact_test_dataset_size = min(self.options.test_dataset_size, len(items_with_hashes) - exact_training_dataset_size - exact_validation_dataset_size)

        logger.info(f'Exact # of Items - Training/Validation/Test = {exact_training_dataset_size} / {exact_validation_dataset_size} / {exact_test_dataset_size}')

        # Step 5: Save Datasets
        training_dataset_save_dirpath = self.options.save_dirpath.joinpath('training')
        logger.info(f'Saving \'Training\' Dataset into {training_dataset_save_dirpath.absolute()} ... ')
        self.__class__.save_dataset(uuid_occurrence, items_with_hashes[:exact_training_dataset_size], training_dataset_save_dirpath, ignored)

        validation_dataset_save_dirpath = self.options.save_dirpath.joinpath('validation')
        logger.info(f'Saving \'Validation\' Dataset into {validation_dataset_save_dirpath.absolute()} ... ')
        self.__class__.save_dataset(uuid_occurrence, items_with_hashes[exact_training_dataset_size:exact_training_dataset_size + exact_validation_dataset_size], validation_dataset_save_dirpath, ignored)
        test_dataset_save_dirpath = self.options.save_dirpath.joinpath('test')
        logger.info(f'Saving \'Test\' Dataset into {test_dataset_save_dirpath.absolute()} ... ')
        self.__class__.save_dataset(uuid_occurrence, items_with_hashes[exact_training_dataset_size + exact_validation_dataset_size:exact_training_dataset_size + exact_validation_dataset_size + exact_test_dataset_size], test_dataset_save_dirpath, ignored)

    @classmethod
    def retrieve_split(cls, logicx_filepath: pathlib.Path, center_node_indices: list[str], split_scale: int, split_limit: int, method: Literal['Random', 'Cascade', 'RandomFull', 'CascadeFull', 'Window']) -> LogicX:
        # Direction: Literal[0, 1, -1] Center Node: 0; Successor: 1; Predecessors: -1;
        logicx = LogicX()
        logicx.load(logicx_filepath)
        bfs_flags = set(center_node_indices)
        bfs_queue = collections.deque([(center_node_index, 0) for center_node_index in center_node_indices])
        if method == 'Window':
            current_level = 0
            while len(bfs_queue) != 0 and current_level > -split_scale:
                prev_level_size = len(bfs_queue)
                for _ in range(prev_level_size):
                    node_index, _ = bfs_queue.popleft()
                    neighbors = [predecessor for predecessor in logicx.dag.predecessors(node_index)]
                    numpy.random.shuffle(neighbors)
                    for neighbor in neighbors:
                        if len(bfs_flags) < split_limit and neighbor not in bfs_flags:
                            bfs_flags.add(neighbor)
                            bfs_queue.append((neighbor, 0))
                current_level = current_level - 1
        else:
            while len(bfs_queue) != 0 and len(bfs_flags) < split_scale:
                current_node_index, direction = bfs_queue.popleft()
                next_levels = list()

                if method in ['Random', 'RandomFull']:
                    for neighbor in networkx.function.all_neighbors(logicx.dag, current_node_index):
                        next_levels.append((neighbor, 0))

                if method in ['Cascade', 'CascadeFull']:
                    if 0 <= direction:
                        for neighbor in logicx.dag.successors(current_node_index):
                            next_levels.append((neighbor, 1))

                    if direction <= 0:
                        for neighbor in logicx.dag.predecessors(current_node_index):
                            next_levels.append((neighbor, -1))

                if len(next_levels) == 0:
                    continue

                numpy.random.shuffle(next_levels)
                if method in ['Random', 'Cascade']:
                    limit = numpy.random.randint(1, len(next_levels) + 1)
                    for neighbor, direction in next_levels[:limit]:
                        if len(bfs_flags) < split_scale and neighbor not in bfs_flags:
                            bfs_flags.add(neighbor)
                            bfs_queue.append((neighbor, direction))

                if method in ['RandomFull', 'CascadeFull']:
                    if len(bfs_flags) < split_scale:
                        for neighbor, direction in next_levels:
                            if neighbor not in bfs_flags:
                                bfs_flags.add(neighbor)
                                bfs_queue.append((neighbor, direction))

        induced_subgraph = logicx.dag.subgraph(bfs_flags)

        subgraph = networkx.DiGraph()
        subgraph.add_nodes_from(induced_subgraph.nodes(data=False))
        subgraph.add_edges_from(induced_subgraph.edges(data=False))
        for node_index in subgraph.nodes():
            subgraph.nodes[node_index]['node_uuid'] = induced_subgraph.nodes[node_index]['node_uuid']

        split = LogicX(src=logicx.src, dag=subgraph)
        return split

    @classmethod
    def mark_node_levels(cls, dag: networkx.DiGraph) -> None:
        """
        Mark level for each node in the DAG using reverse topological order.
        Level represents the longest path from the node to any leaf node (negative value).
        Leaf nodes (out-degree = 0) have level = 0.

        Args:
            dag: The directed acyclic graph to mark levels for (modified in-place)
        """

        node_ods: dict[str, int] = {node_index: dag.out_degree(node_index) for node_index in dag.nodes}
        bfs_queue = collections.deque()
        for node_index in dag.nodes:
            if node_ods[node_index] == 0:
                bfs_queue.append(node_index)
        while len(bfs_queue) != 0:
            prev_level_size = len(bfs_queue)
            for _ in range(prev_level_size):
                node_index = bfs_queue.popleft()
                dag.nodes[node_index]['level'] = min([dag.nodes[successor].get('level', 0) - 1 for successor in dag.successors(node_index)] + [0])
                for predecessor in dag.predecessors(node_index):
                    node_ods[predecessor] -= 1
                    if node_ods[predecessor] == 0:
                        bfs_queue.append(predecessor)

    @classmethod
    def save_dataset(cls, uuid_occurence: dict[str, int], dag_with_hashes: list[tuple[str, LogicX]], save_dirpath: pathlib.Path, ignored: set[str]):
        node_types = [node_type for node_type, node_ocr in uuid_occurence.items() if node_type not in ignored]
        item_names = [item_name for item_name, item_dag in dag_with_hashes]
        meta = dict(
            node_types = node_types,
            item_names = item_names,
        )

        pack_filepath = save_dirpath.joinpath('pack.pkl')
        # create_dir(items_dirpath)
        meta_filepath = save_dirpath.joinpath('meta.json')

        logger.info(f'Saving META ... ')
        save_json(meta, meta_filepath, indent=2)
        logger.info(f'Saved.')

        logger.info(f'Packing Items ... ')
        package = dict()
        with tqdm.tqdm(total=len(dag_with_hashes), desc='Packing') as progress_bar:
            for dag_hash, dag in dag_with_hashes:
                package[dag_hash] = LogicX.saves_dag(dag)
                progress_bar.update(1)
        logger.info(f'Packed.')

        logger.info(f'Saving Package ... ')
        save_pickle(package, pack_filepath)
        # with tqdm.tqdm(total=len(split_with_hashes), desc='Saving') as progress_bar:
        #     for split_hash, split in split_with_hashes:
        #         item_filepath = items_dirpath.joinpath(f'{split_hash}')
        #         split.save(item_filepath)
        #         progress_bar.update(1)
        logger.info(f'Saved.')
