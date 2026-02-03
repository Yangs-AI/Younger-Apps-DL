#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-02-24 12:10:50
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-02-03 14:50:57
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import os
import re
import torch
import pathlib

from typing import Literal


class Checkpoint(object):
    def __init__(self, epoch: int, step: int, itr: int, model_state_dict: dict, optimizer_state_dict: dict, scheduler_state_dict: dict, metrics: dict[str, float]):
        self._epoch = epoch
        self._step = step
        self._itr = itr
        self._model_state_dict = model_state_dict 
        self._optimizer_state_dict = optimizer_state_dict
        self._scheduler_state_dict = scheduler_state_dict
        self._metrics = metrics

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def step(self) -> int:
        return self._step

    @property
    def itr(self) -> int:
        return self._itr

    @property
    def model_state_dict(self) -> dict:
        return self._model_state_dict

    @property
    def optimizer_state_dict(self) -> dict:
        return self._optimizer_state_dict

    @property
    def scheduler_state_dict(self) -> dict:
        return self._scheduler_state_dict

    @property
    def metrics(self) -> dict[str, float]:
        return self._metrics

    @classmethod
    def load_dict(cls, d: dict) -> 'Checkpoint':
        c = Checkpoint(
            d['epoch'],
            d['step'],
            d['itr'],
            d['model_state_dict'],
            d['optimizer_state_dict'],
            d['scheduler_state_dict'],
            d['metrics'],
        )
        return c

    @classmethod
    def save_dict(cls, c: 'Checkpoint') -> dict:
        c = dict(
            epoch = c._epoch,
            step = c._step,
            itr = c._itr,
            model_state_dict = c._model_state_dict,
            optimizer_state_dict = c._optimizer_state_dict,
            scheduler_state_dict = c._scheduler_state_dict,
            metrics = c._metrics,
        )
        return c


def retrieve_checkpoint_filepaths(dirpath: pathlib.Path, basename: str = 'checkpoint') -> dict[int, pathlib.Path]:
    checkpoint_filename_pattern = re.compile(rf'{basename}_Itr_(\d+)\.cp')
    checkpoint_filepaths = dict()
    for path in dirpath.iterdir():
        if path.is_file():
            result = checkpoint_filename_pattern.fullmatch(path.name)
            if result is not None:
                itr = int(result.group(1))
                checkpoint_filepaths[itr] = path
            else:
                continue
        else:
            continue

    return checkpoint_filepaths


def load_checkpoint(load_path: pathlib.Path, basename: str = 'checkpoint') -> Checkpoint | None:
    checkpoint = None
    if load_path.is_file():
        checkpoint = Checkpoint.load_dict(torch.load(load_path, map_location=torch.device('cpu')))

    if load_path.is_dir():
        assert len(basename) != 0, f'Invalid checkpoint name.'
        checkpoint_filepaths = retrieve_checkpoint_filepaths(load_path, basename)

        if len(checkpoint_filepaths) == 0:
            checkpoint = None
        else:
            itr = max(checkpoint_filepaths.keys())
            checkpoint_path = checkpoint_filepaths[itr]
            if checkpoint_path.is_file():
                checkpoint = Checkpoint.load_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
                assert itr == checkpoint.itr, 'An error occurred when loading checkpoint.'
            else:
                checkpoint = None

    return checkpoint


def save_checkpoint(checkpoint: Checkpoint, save_path: pathlib.Path, basename: str = 'checkpoint', keep_number: int = 1) -> None:
    if save_path.is_dir():
        assert len(basename) != 0, f'Invalid checkpoint name.'
        checkpoint_filename = f'{basename}_Itr_{checkpoint.itr}.cp'
        checkpoint_filepath = save_path.joinpath(checkpoint_filename)
        torch.save(Checkpoint.save_dict(checkpoint), checkpoint_filepath)

        latest_checkpoint_filepath = save_path.joinpath(f'{basename}_latest.cp')
        torch.save(Checkpoint.save_dict(checkpoint), latest_checkpoint_filepath)

        checkpoint_filepaths = retrieve_checkpoint_filepaths(save_path, basename)
        itrs = sorted(list(checkpoint_filepaths.keys()), reverse=True)
        for itr in itrs[keep_number:]:
            remove_checkpoint(checkpoint_filepaths[itr])
    else:
        checkpoint_filepath = save_path
        torch.save(Checkpoint.save_dict(checkpoint), checkpoint_filepath)


def remove_checkpoint(checkpoint_filepath: pathlib.Path) -> None:
    if os.path.isfile(checkpoint_filepath):
        os.remove(checkpoint_filepath)
    else:
        raise IOError(f'Invalid address: {checkpoint_filepath}')
