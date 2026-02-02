#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2026-01-30 00:00:00
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-02-02 01:53:58
# Copyright (c) 2026 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import os
import sys
import pathlib
import importlib
import contextlib

from younger_apps_dl.commons.logging import logger


@contextlib.contextmanager
def manage_sys_path(modpath: pathlib.Path):
    modpath = str(modpath.resolve())
    syspaths = list(sys.path)

    sys.path[:] = [p for p in sys.path if p != modpath]
    sys.path.insert(0, modpath)

    try:
        yield
    finally:
        sys.path[:] = syspaths


def load_user_modules(optional_dirpath: pathlib.Path | None = None):
    """
    Load user-defined modules from a user-provided directory by importing a
    single, required entry module.

    Directory convention (required):
    1) optional_dirpath is a directory that is treated as a temporary import root.
    2) The directory MUST provide an entry module named "register.py" at root.
    3) The entry module must ONLY import all user modules that need
        registration (e.g., models, tasks, datasets, engines). Avoid heavy
        runtime logic in this file.

    Example layout:
    optional_dirpath/
    ├── register.py
    ├── models/
    │   └── *.py  (using @register_model)
    ├── tasks/
    │   └── *.py  (using @register_task)
    ├── datasets/
    │   └── *.py  (using @register_dataset)
    └── engines/
        └── *.py  (using @register_engine)

    Args:
        optional_dirpath: Path to the user modules directory. If not provided,
            it will fall back to the environment variable YADL_OPTIONAL_DIRPATH.

    Raises:
        FileNotFoundError: If optional_dirpath or register.py is missing
        ImportError: If the entry module cannot be imported
    """
    if optional_dirpath is None:
        env_optional_dirpath = os.environ.get('YADL_OPTIONAL_DIRPATH', '').strip()
        if len(env_optional_dirpath) == 0:
            logger.info('No user-defined modules directory provided or found in environment variable YADL_OPTIONAL_DIRPATH.')
            return None
        optional_dirpath = pathlib.Path(env_optional_dirpath)

    if optional_dirpath.exists():
        package_init_filepath = optional_dirpath.joinpath('__init__.py')
        is_package_dir = package_init_filepath.is_file()

        logger.info(f'Loading user modules from: {optional_dirpath}')

        if is_package_dir:
            package_name = optional_dirpath.name
            package_parent = optional_dirpath.parent
            logger.info(f'Package detected. Using {package_init_filepath} as entry.')
            with manage_sys_path(package_parent):
                try:
                    module = importlib.import_module(package_name)
                except Exception as exception:
                    raise ImportError(
                        f'Failed to import package "{package_name}" from {optional_dirpath}. '
                        f'Please ensure {package_init_filepath} exists and keep imports there.'
                    ) from exception
            logger.info(f'Successfully loaded package: {package_name}')
        else:
            raise FileNotFoundError(
                f'Missing package entry file: {package_init_filepath}. '
                f'Please provide __init__.py under {optional_dirpath}.'
            )
    else:
        logger.info('No user-defined modules directory provided or found in environment variable YADL_OPTIONAL_DIRPATH.')
