#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-12-25 02:26:46
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-01-24 08:54:32
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


from pydantic import BaseModel, Field

from younger_apps_dl.tasks import BaseTask, register_task
from younger_apps_dl.engines import StandardPreprocessor, StandardPreprocessorOptions


class BasicPreprocessOptions(BaseModel):
    # Stage-specific options
    preprocessor: StandardPreprocessorOptions = Field(..., description='Preprocessor options, required for preprocessing stage.')


@register_task('ir', 'basic_preprocess')
class BasicPreprocess(BaseTask[BasicPreprocessOptions]):
    """
    This task performs basic preprocessing on directed acyclic graphs (DAGs) using a standard preprocessor.
    It is designed to handle the preprocessing stage of graph-based data for various applications.
    """
    OPTIONS = BasicPreprocessOptions

    @property
    def required_option_names_by_stage(self) -> dict[str, list[str]]:
        return {
            'preprocess': ['preprocessor'],
        }

    def _preprocess_(self):
        preprocessor = StandardPreprocessor(self.options.preprocessor)
        preprocessor.run()

    def _train_(self):
        return super()._train_()

    def _evaluate_(self):
        return super()._evaluate_()

    def _predict_(self):
        return super()._predict_()

    def _postprocess_(self):
        return super()._postprocess_()
