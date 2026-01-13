#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-05-17 03:29:00
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-01-02 10:05:58
# Copyright (c) 2024 - 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


from abc import ABC, abstractmethod
from typing import Literal, Type

from younger_apps_dl.commons.mixins.options import OptionsMixin, OPTIONS_TYPE


class BaseTask(OptionsMixin[OPTIONS_TYPE], ABC):
    def __init__(self, options: OPTIONS_TYPE):
        super().__init__(options)

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def preprocess(self):
        raise NotImplementedError

    def postprocess(self):
        raise NotImplementedError


TASK_REGISTRY: dict[ Literal['ir', 'core'], dict[ str, Type[BaseTask] ] ] = dict(
    ir = dict(),
    core = dict(),
)


def register_task(
    kind: Literal['ir', 'core'],
    name: str
):
    assert kind in {'ir', 'core'}
    assert name not in TASK_REGISTRY[kind]
    def wrapper(cls):
        assert issubclass(cls, BaseTask)
        TASK_REGISTRY[kind][name] = cls
        return cls
    return wrapper

from .ir import *
from .core import *
