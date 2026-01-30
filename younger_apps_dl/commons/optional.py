#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2026-01-30 00:00:00
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-01-30 14:50:14
# Copyright (c) 2026 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import os
import sys
import tqdm
import pathlib

import importlib.util

from younger_apps_dl.commons.logging import logger


def load_user_modules(optional_dirpath: pathlib.Path):
    """
    Load user-defined modules from the specified directory.

    The structure of the directory does not matter, but each module file should register itself using the appropriate decorators when imported.

    The possible registrations include:
    - Models: Use `@register_model` from `younger_apps_dl.models`
    - Datasets: Use `@register_dataset` from `younger_apps_dl.datasets`
    - Engines: Use `@register_engine` from `younger_apps_dl.engines`
    - Tasks: Use `@register_task` from `younger_apps_dl.tasks`

    However, for better organization, it is recommended to follow this structure.

    optional_dirpath/
    ├── models/
    │   └── *.py  (using @register_model)
    ├── tasks/
    │   └── *.py  (using @register_task)
    ├── datasets/
    │   └── *.py  (using @register_dataset)
    └── engines/
        └── *.py  (using @register_engine)

    :param optional_dirpath: Path to the user modules directory
    
    Example:
        load_user_modules('/home/user/my_yadl_project')
        # This will load /home/user/my_yadl_project/models/*.py that register models,
        # /home/user/my_yadl_project/tasks/*.py that register tasks, etc.
    """
    logger.info(f'Loading user modules from: {optional_dirpath}')
    python_files_to_load = list(optional_dirpath.rglob('*.py'))
    loaded_count = 0
    ignore_count = 0
    with tqdm(total=len(python_files_to_load), desc='Scanning optional directory') as progress_bar:
        for python_file_to_load in python_files_to_load:
            progress_bar.update(1)
            if '__pycache__' in python_file_to_load.parts or python_file_to_load.name.startswith('_'):
                logger.debug(f'  Ignored {python_file_to_load} (private or cache file)')
                ignore_count += 1
                continue
            try:
                spec = importlib.util.spec_from_file_location(module_name, python_file_to_load)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    logger.info(f'  Loaded {component_type}: {py_file.name}')
                    loaded_count += 1
            except Exception as e:
                logger.error(f'  Failed to load {py_file}: {e}')
                import traceback
                logger.debug(traceback.format_exc())
    
            loaded_count += 1

    if loaded_count > 0:
        logger.info(f'Successfully loaded {loaded_count} user module(s)')
    else:
        logger.warning(f'No user modules found in {optional_dirpath}')

    # 递归查找所有 .py 文件
    for py_file in directory.rglob('*.py'):
        # 跳过 __init__.py 和 __pycache__
        
        # 构造模块名称
        relative_path = py_file.relative_to(directory)
        module_name = f'_user_{component_type}_{relative_path.with_suffix("").as_posix().replace("/", "_")}'
        
        try:
            # 动态加载模块
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                logger.info(f'  Loaded {component_type}: {py_file.name}')
                loaded_count += 1
        except Exception as e:
            logger.error(f'  Failed to load {py_file}: {e}')
            import traceback
            logger.debug(traceback.format_exc())
    
    return loaded_count


def find_user_modules():
    """
    从环境变量 YOUNGER_APPS_DL_OPTIONAL_DIRPATH 中发现并加载用户模块。
    
    环境变量可以包含多个路径，用冒号分隔（Linux/Mac）或分号分隔（Windows）：
    export YOUNGER_APPS_DL_OPTIONAL_DIRPATH=/path/to/modules1:/path/to/modules2
    """
    env_var = os.environ.get('YOUNGER_APPS_DL_OPTIONAL_DIRPATH', '')
    if not env_var:
        return
    
    # 根据操作系统选择分隔符
    separator = ';' if sys.platform == 'win32' else ':'
    paths = [p.strip() for p in env_var.split(separator) if p.strip()]
    
    for path in paths:
        load_user_modules(path)
