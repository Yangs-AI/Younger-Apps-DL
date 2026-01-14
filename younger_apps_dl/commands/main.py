#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-11-27 09:53:09
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-01-14 09:42:33
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import click
import pathlib
import tabulate

from typing import Literal

from younger.commons.io import load_toml

from younger_apps_dl.commons.help import generate_helping_for_pydantic_model
from younger_apps_dl.commons.logging import equip_logger


@click.group(name='younger-apps-dl')
def main():
    pass


@main.command(name='glance')
@click.option('--some-type', required=True, type=click.Choice(['models', 'datasets', 'engines', 'tasks'], case_sensitive=True), help='Indicates the type of task will be used.')
@click.option('--logging-filepath', required=False, type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='Path to the log file; if not provided, defaults to outputting to the terminal only.')
def glance(some_type: Literal['models', 'datasets', 'engines', 'tasks'], logging_filepath: pathlib.Path):
    """
    Displays all possible candidates under a specific `type`.

    :param some_type: Indicates the type of DL component.
    :type some_type: Literal[&#39;models&#39;, &#39;datasets&#39;, &#39;engines&#39;, &#39;tasks&#39;]
    :param logging_filepath: Path to the log file; if not provided, defaults to outputting to the terminal only.
    :type logging_filepath: pathlib.Path
    """
    equip_logger(logging_filepath=logging_filepath)

    table_name = some_type.capitalize()
    table_data = list()
    registry = dict()
    if some_type in ['models', 'datasets']:
        if some_type == 'models':
            from younger_apps_dl.models import MODEL_REGISTRY
            registry = MODEL_REGISTRY

        if some_type == 'datasets':
            from younger_apps_dl.datasets import DATASET_REGISTRY
            registry = DATASET_REGISTRY

        for name, cls in registry.items():
            table_data.append([name, cls.__name__])

        if len(table_data) == 0:
            print(f'{table_name}\'s Registry is Empty')
        else:
            print(f'{table_name}\'s Registry')
            print(tabulate.tabulate(table_data, headers=['Name', 'Class'], tablefmt='grid'))

    if some_type in ['engines', 'tasks']:
        if some_type == 'engines':
            from younger_apps_dl.engines import ENGINE_REGISTRY
            registry = ENGINE_REGISTRY

        if some_type == 'tasks':
            from younger_apps_dl.tasks import TASK_REGISTRY
            registry = TASK_REGISTRY

        for kind, name2cls in registry.items():
            for name, cls in name2cls.items():
                table_data.append([kind, name, cls.__name__])

        if len(table_data) == 0:
            print(f'{table_name}\'s Registry is Empty')
        else:
            print(f'{table_name}\'s Registry')
            print(tabulate.tabulate(table_data, headers=['Kind', 'Name', 'Class'], tablefmt='grid'))


@main.command(name='option')
@click.option('--task-kind', required=True, type=str, help='Indicates the type of task.')
@click.option('--task-name', required=True, type=str, help='Indicates the name of task.')
@click.option('--toml-path', required=True, type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='Path to the configuration file.')
@click.option('--logging-filepath', required=False, type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='Path to the log file; if not provided, defaults to outputting to the terminal only.')
def option(task_kind: str, task_name: str, toml_path: pathlib.Path, logging_filepath: pathlib.Path):
    """
    Gets the configuration template for a specific task.

    :param task_kind: Indicates the type of task.
    :type task_kind: str
    :param task_name: Indicates the name of task.
    :type task_name: str
    :raises exception: If the specified task is not found in the registry.
    :param toml_path: Path to the configuration file.
    :type toml_path: pathlib.Path
    :param logging_filepath: Path to the log file; if not provided, defaults to outputting to the terminal only.
    :type logging_filepath: pathlib.Path
    """
    equip_logger(logging_filepath=logging_filepath)

    from younger_apps_dl.tasks import TASK_REGISTRY
    try:
        Task = TASK_REGISTRY[task_kind][task_name]
    except Exception as exception:
        click.echo(f'No <{task_kind}, {task_name}> Task in Task Registry.')
        raise exception

    helping_lines = generate_helping_for_pydantic_model(Task.OPTIONS)
    helping = '\n'.join(helping_lines)
    toml_path.write_text(helping, encoding='utf-8')
    click.echo(f'Configuration template for "<{task_kind}> -> <{task_name}>" task has been written to the file - "{toml_path.absolute()}".')


@main.command(name='launch')
@click.option('--task-kind', required=True, type=str, help='Indicates the type of task.')
@click.option('--task-name', required=True, type=str, help='Indicates the name of task.')
@click.option('--task-step', required=True, type=click.Choice(['train', 'evaluate', 'predict', 'preprocess', 'postprocess'], case_sensitive=True), help='Indicates the step of task.')
@click.option('--toml-path', required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='Path to the configuration file.')
@click.option('--logging-filepath', required=False, type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='Path to the log file; if not provided, defaults to outputting to the terminal only.')
def launch(task_kind: str, task_name: str, task_step: str, toml_path: pathlib.Path, logging_filepath: pathlib.Path):
    """
    Launches a specific task with given configuration.

    :param task_kind: Indicates the type of task.
    :type task_kind: str
    :param task_name: Indicates the name of task.
    :type task_name: str
    :param task_step: Indicates the step of task.
    :type task_step: str
    :param toml_path: Path to the configuration file.
    :type toml_path: pathlib.Path
    :raises exception: If the specified task is not found in the registry.
    :param logging_filepath: Path to the log file; if not provided, defaults to outputting to the terminal only.
    :type logging_filepath: pathlib.Path
    """
    equip_logger(logging_filepath=logging_filepath)

    from younger_apps_dl.tasks import TASK_REGISTRY
    try:
        Task = TASK_REGISTRY[task_kind][task_name]
    except Exception as exception:
        click.echo(f'No <{task_kind}, {task_name}> Task in Task Registry.')
        raise exception

    task = Task(Task.OPTIONS(**load_toml(toml_path)))
    if task_step == 'train':
        task.train()

    if task_step == 'evaluate':
        task.evaluate()

    if task_step == 'predict':
        task.predict()

    if task_step == 'preprocess':
        task.preprocess()

    if task_step == 'postprocess':
        task.postprocess()


if __name__ == '__main__':
    main()
