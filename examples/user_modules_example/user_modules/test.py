#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2026-02-03 16:03:30
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-02-03 16:35:16
# Copyright (c) 2026 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import pathlib

from .tasks import SimpleDemoTask, SimpleDemoTaskOptions

from younger_apps_dl.commons.logging import equip_logger, logger


if __name__ == '__main__':
    # Simple test for SimpleDemoTask
    equip_logger()
    logger.info("Test SimpleDemoTask...")

    # Create 
    options = SimpleDemoTaskOptions(
        input_dim=10,
        hidden_dim=20,
        output_dim=5,
    )

    task = SimpleDemoTask(options)
    # Test each stage
    logger.info("--- Test Preprocess ---")
    task.preprocess()
    logger.info("--- Done Preprocess ---")
    logger.info("--- Test Train ---")
    task.train()
    logger.info("--- Done Train ---")
    logger.info("--- Test Evaluate ---")
    task.evaluate()
    logger.info("--- Done Evaluate ---")
    logger.info("--- Test Predict ---")
    raw_options = options.predictor.raw
    raw_options.load_dirpath.mkdir(parents=True, exist_ok=True)
    sample_values = " ".join(str(v) for v in range(options.model.input_dim))
    sample_path = raw_options.load_dirpath / "sample.txt"
    sample_path.write_text(sample_values, encoding="utf-8")
    task.predict()
    logger.info("--- Done Predict ---")
    logger.info("--- Test Postprocess ---")
    task.postprocess()
    logger.info("--- Done Postprocess ---")
    logger.info("✓ All Test Passed!")
