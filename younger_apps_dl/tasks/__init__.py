#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-05-17 03:29:00
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-02-02 02:16:24
# Copyright (c) 2024 - 2026 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


from abc import ABC, abstractmethod
from typing import Literal, Type, ClassVar

from younger_apps_dl.commons.mixins.options import OptionsMixin, OPTIONS_TYPE


class BaseTask(OptionsMixin[OPTIONS_TYPE], ABC):
    STAGE_REQUIRED_OPTION: ClassVar[dict[str, list[str]]]

    def __init__(self, options: OPTIONS_TYPE):
        super().__init__(options)
        if not hasattr(self.__class__, 'STAGE_REQUIRED_OPTION'):
            raise NotImplementedError(
                f"{self.__class__.__name__} must define STAGE_REQUIRED_OPTION class attribute, which mapping from stage name to list of required option field names."
            )

    def _check_options_(self, stage: Literal['preprocess', 'train', 'evaluate', 'predict', 'postprocess']) -> None:
        """Check if required options are provided for the given stage."""
        for required_option_name in self.__class__.STAGE_REQUIRED_OPTION.get(stage, list()):
            if getattr(self.options, required_option_name, None) is None:
                raise ValueError(f"{required_option_name} option is required for {stage} stage")

    def preprocess(self):
        """Preprocess data. Automatically checks requirements before execution."""
        self._check_options_('preprocess')
        self._preprocess_()

    def train(self):
        """Train model. Automatically checks requirements before execution."""
        self._check_options_('train')
        self._train_()

    def evaluate(self):
        """Evaluate model. Automatically checks requirements before execution."""
        self._check_options_('evaluate')
        self._evaluate_()

    def predict(self):
        """Predict with model. Automatically checks requirements before execution."""
        self._check_options_('predict')
        self._predict_()

    def postprocess(self):
        """Postprocess results. Automatically checks requirements before execution."""
        self._check_options_('postprocess')
        self._postprocess_()

    @abstractmethod
    def _preprocess_(self):
        """Implement preprocessing logic in subclass."""
        raise NotImplementedError

    @abstractmethod
    def _train_(self):
        """Implement training logic in subclass."""
        raise NotImplementedError

    @abstractmethod
    def _evaluate_(self):
        """Implement evaluation logic in subclass."""
        raise NotImplementedError

    @abstractmethod
    def _predict_(self):
        """Implement prediction logic in subclass."""
        raise NotImplementedError

    @abstractmethod
    def _postprocess_(self):
        """Implement postprocessing logic in subclass."""
        raise NotImplementedError


TASK_REGISTRY: dict[ Literal['ir', 'core', 'optional'], dict[ str, Type[BaseTask] ] ] = dict(
    ir = dict(),
    core = dict(),
    optional = dict(),
)


def register_task(
    kind: Literal['ir', 'core', 'optional'],
    name: str
):
    assert kind in {'ir', 'core', 'optional'}
    assert name not in TASK_REGISTRY[kind]
    def wrapper(cls):
        assert issubclass(cls, BaseTask)
        TASK_REGISTRY[kind][name] = cls
        return cls
    return wrapper

from .ir import *
from .core import *
