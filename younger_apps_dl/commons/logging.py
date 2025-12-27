#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-02-21 13:19:17
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-12-27 02:47:44
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import pathlib

from younger.commons.logging import equip_package_logger, get_package_logger


def get_package_name():
    """Lazy import __thename__ to avoid circular import issues"""
    from younger_apps_dl import __thename__
    return __thename__


def equip_logger(logging_filepath: pathlib.Path | str | None = None):
    """Configure the logger for the current application."""
    equip_package_logger(get_package_name(), logging_filepath)


def __getattr__(name: str):
    """Return a lazy logger without configuring handlers to avoid side effects on import."""
    if name == 'logger':
        return get_package_logger(get_package_name(), auto_create=False)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
