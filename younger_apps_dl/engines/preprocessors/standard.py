#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-03-14 21:43:17
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-01-21 23:22:19
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import tqdm
import numpy
import random
import pathlib
import networkx
import collections

from typing import Any, Literal
from pydantic import BaseModel, Field

from younger.commons.io import save_json, save_pickle

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
                                                                                               '\'Window\' randomly selects a graph and identifies nodes at a specific level; then performs a `split_scale`-step backward traversal, incorporating all traversed nodes and edges into the subgraph.'
                                                                                               '\'MixBasic\' uniformly samples one of the following methods for each subgraph: \'Random\', \'RandomFull\', \'Cascade\', or \'CascadeFull\'. '
                                                                                               '\'MixSuper\' extends MixBasic by additionally including \'Window\' in the set of candidate methods, sampling uniformly among all five.')

    split_scale: list[int] | None = Field(None, description='List of node counts to include in subgraph splits expanded from central nodes. Each value specifies a different subgraph split scale to generate.')
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


@register_engine('preprocessor', 'standard')
class StandardPreprocessor(BaseEngine[StandardPreprocessorOptions]):
    OPTIONS = StandardPreprocessorOptions

    def run(self) -> None:
        random.seed(self.options.seed)
        numpy.random.seed(self.options.seed)

        logger.info(f'Reading LogicX Files from {self.options.load_dirpath.absolute()} ... ')
        logger.info(f'Random Seed = {self.options.seed}')
        logger.info(f'Subgraph Extraction: {"Enabled" if self.options.subgraph else "Disabled"}')

        # Step 1: Load and Filter LogicX files
        logicx_filepaths = sorted([logicx_filepath for logicx_filepath in self.options.load_dirpath.iterdir()])
        logger.info(f'Scanning and Loading LogicX files ...')
        logicxs: list[LogicX] = list() # [logicx1, logicx2, ...]
        logicx_hashes: list[str] = list() # [logicx1_hash, logicx2_hash, ...]
        all_uuid_positions: dict[str, dict[int, set[str]]] = dict() # {uuid: {logicx_index: set[node_index]}}
        all_nid2nod: dict[str, dict[str, int]] = dict() # {logicx_index: {node_index: order}}
        all_nod2nids: dict[str, dict[int, list[str]]] = dict() # {logicx_index: {order: list[node_index]}}

        with tqdm.tqdm(total=len(logicx_filepaths)) as progress_bar:
            logicx_index = 0
            for logicx_filepath in logicx_filepaths:
                progress_bar.update(1)
                logicx = LogicX()
                logicx.load(logicx_filepath)

                dag_size = len(logicx.dag)
                if self.options.min_dag_size is None or self.options.min_dag_size <= dag_size:
                    min_dag_size_meet = True
                else:
                    min_dag_size_meet = False

                if self.options.max_dag_size is None or dag_size <= self.options.max_dag_size:
                    max_dag_size_meet = True
                else:
                    max_dag_size_meet = False

                if not (min_dag_size_meet and max_dag_size_meet):
                    continue

                logicxs.append(logicx)
                logicx_hashes.append(logicx_filepath.name)

                # Generate UUID Positions and Node Orders
                # {uuid: {logicx_index: set[node_index]}}
                # uuid is the operator type
                # logicx_index is the index of logicx in logicxs
                # node_index is the index of node in logicx.dag
                # e.g., {'uuid1': {0: {'n1', 'n2'}, 2: {'n3'}}, 'uuid2': {1: {'n4'}}}
                # Thus, uuid1 appears in logicx 0 at nodes n1 and n2, and in logicx 2 at node n3; uuid2 appears in logicx 1 at node n4.
                for node_index in logicx.dag.nodes:
                    uuid = logicx.dag.nodes[node_index]['node_uuid']
                    uuid_positions = all_uuid_positions.get(uuid, dict())
                    node_indices = uuid_positions.get(logicx_index, set())
                    node_indices.add(node_index)
                    uuid_positions[logicx_index] = node_indices
                    all_uuid_positions[uuid] = uuid_positions

                # From Topological Order, Generate Node Index to Node Order & Node Order to Node Indices
                # Node Index: The Original Node Index in DAG
                # Node Order: The Longest Path from Root to Node
                all_nid2nod[logicx_index] = dict()
                all_nod2nids[logicx_index] = dict()
                for node_index in networkx.topological_sort(logicx.dag):
                    predecessors = logicx.dag.predecessors(node_index)
                    all_nid2nod[logicx_index][node_index] = max([all_nid2nod[logicx_index][predecessor] + 1 for predecessor in predecessors] + [0])
                    all_nod2nids[logicx_index].setdefault(all_nid2nod[logicx_index][node_index], list()).append(node_index)

                logicx_index += 1

        logger.info(f'Loaded {len(logicxs)} DAGs matching size criteria.')

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
            expected_overall_split_count = len(uuid_occurrence)*len(self.options.split_scale)*min(self.options.split_count, self.options.split_tries)
            expected_training_dataset_size = min(expected_overall_split_count, self.options.training_dataset_size)
            expected_validation_dataset_size = min(max(0, expected_overall_split_count-self.options.training_dataset_size), self.options.validation_dataset_size)
            expected_test_dataset_size = min(max(0, expected_overall_split_count-self.options.training_dataset_size-self.options.validation_dataset_size), self.options.test_dataset_size)
            logger.info(f'Expected Overal # of Splits = {expected_overall_split_count}')
            logger.info(f'Expected # of Splits - Training/Validation/Test = {expected_training_dataset_size} / {expected_validation_dataset_size} / {expected_test_dataset_size}')

            # For Each Split Size, For Each Operator, Generate Specific Number of Subgraph Splits
            logger.info(f'Splitting ...')
            splits: dict[int, dict[str, LogicX]] = {split_scale: dict() for split_scale in self.options.split_scale} # {split_scale: {split_hash: split}}
            split_hashes: dict[int, dict[str, list[str]]] = {split_scale: dict() for split_scale in self.options.split_scale} # {split_scale: {uuid: list[split_hash]}}
            # For Each Split Size:
            for split_scale in self.options.split_scale:
                # For Each Operator:
                logger.info(f' -> Now Retrieving Subgraph Splits with Size {split_scale}...')
                with tqdm.tqdm(total=len(all_uuid_positions)) as progress_bar:
                    for uuid, uuid_positions in all_uuid_positions.items():
                        current_tries = 0
                        current_split_count = 0
                        candidate_logicx_indices: set[int] = set(uuid_positions.keys())
                        # Generate Subgraph Split Repeatedly
                        while len(candidate_logicx_indices) != 0 and current_split_count < self.options.split_count:
                            if not (current_tries < self.options.split_tries):
                                break
                            selected_logicx_index: int = int(numpy.random.choice(list(candidate_logicx_indices)))
                            selected_node_index: str = str(numpy.random.choice(list(uuid_positions[selected_logicx_index])))

                            method = self.options.method
                            if self.options.method == 'MixBasic':
                                method = random.choice(['Random', 'Cascade', 'RandomFull', 'CascadeFull'])
                            if self.options.method == 'MixSuper':
                                method = random.choice(['Random', 'Cascade', 'RandomFull', 'CascadeFull', 'Window'])

                            if method == 'Window':
                                selected_node_order: int = all_nid2nod[selected_logicx_index][selected_node_index]
                                if selected_node_order < split_scale - 1:
                                    continue
                                selected_node_indices: list[int] = all_nod2nids[selected_logicx_index][selected_node_order]
                                split = self.__class__.retrieve_split(logicxs[selected_logicx_index], selected_node_indices, split_scale, self.options.split_limit, self.options.method)
                                split_size = split_scale
                            else:
                                split = self.__class__.retrieve_split(logicxs[selected_logicx_index], [selected_node_index], split_scale, self.options.split_limit, self.options.method)
                                split_size = len(split.dag)

                            if self.options.level:
                                self.__class__.mark_node_levels(split.dag)
                                split.dag.graph['level'] = True
                            else:
                                split.dag.graph['level'] = False

                            current_tries += 1
                            if split_size not in self.options.split_scale:
                                continue
                            split_hash = LogicX.hash(split)
                            if split_hash in splits[split_size]:
                                continue
                            split.dag.graph['origin'] = logicx_hashes[selected_logicx_index]
                            splits[split_size][split_hash] = split
                            current_split_count += 1
                            # Add To Split
                            split_hashes[split_size].setdefault(uuid, list()).append(split_hash)

                        # Mark This LogicX as Exhausted if No More Splits Can Be Generated
                        if current_split_count < self.options.split_count:
                            flag = f'No!'
                        else:
                            flag = f'Ye!'
                        progress_bar.set_description(f'Current: {uuid} Enough: ({flag})')
                        progress_bar.update(1)

            items_with_hashes = [
                (split_hash, splits[split_scale][split_hash])
                for split_scale, split_hashes_at_split_scale in split_hashes.items()
                for uuid, uuid_split_hashes_at_split_scale in split_hashes_at_split_scale.items()
                for split_hash in uuid_split_hashes_at_split_scale
            ]
        else:
            # Full DAG Mode: Use Full DAGs Directly Without Subgraph Extraction
            logger.info(f'Marking levels for full DAGs ...')
            with tqdm.tqdm(total=len(logicxs)) as progress_bar:
                for logicx in logicxs:
                    if self.options.level:
                        self.__class__.mark_node_levels(logicx.dag)
                        logicx.dag.graph['level'] = True
                    else:
                        logicx.dag.graph['level'] = False
                    progress_bar.update(1)
            items_with_hashes = [(logicx_hash, logicx) for logicx_hash, logicx in zip(logicx_hashes, logicxs)]

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
    def retrieve_split(cls, logicx: LogicX, center_node_indices: list[str], split_scale: int, split_limit: int, method: Literal['Random', 'Cascade', 'RandomFull', 'CascadeFull', 'Window']) -> LogicX:
        # Direction: Literal[0, 1, -1] Center Node: 0; Successor: 1; Predecessors: -1;
        bfs_flags = set(center_node_indices)
        bfs_queue = collections.deque([[center_node_index, 0] for center_node_index in center_node_indices])
        if method == 'Window':
            current_level = 0
            while len(bfs_queue) != 0 and current_level > -split_scale:
                prev_level_size = len(bfs_queue)
                for _ in range(prev_level_size):
                    node_index = bfs_queue.popleft()
                    neighbors = numpy.random.shuffle([predecessor for predecessor in logicx.dag.predecessors(node_index)])
                    for neighbor in neighbors:
                        if len(bfs_flags) < split_limit and neighbor not in bfs_flags:
                            bfs_flags.add(neighbor)
                            bfs_queue.append(neighbor)
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
                            bfs_queue.append([neighbor, direction])

                if method in ['RandomFull', 'CascadeFull']:
                    if len(bfs_flags) < split_scale:
                        for neighbor, direction in next_levels:
                            if neighbor not in bfs_flags:
                                bfs_flags.add(neighbor)
                                bfs_queue.append([neighbor, direction])

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
    def save_dataset(cls, uuid_occurence: dict[str, int], split_with_hashes: list[tuple[str, LogicX]], save_dirpath: pathlib.Path, ignored: set[str]):
        node_types = [node_type for node_type, node_occr in uuid_occurence.items() if node_type not in ignored]
        item_names = [item_name for item_name, item_lgcx in split_with_hashes]
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
        with tqdm.tqdm(total=len(split_with_hashes), desc='Packing') as progress_bar:
            for split_hash, split in split_with_hashes:
                package[split_hash] = LogicX.saves_dag(split)
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
