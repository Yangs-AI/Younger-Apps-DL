#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-11-27 09:53:09
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-12-24 16:20:19
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import click
import pathlib
import tabulate

from typing import Literal

from younger.commons.io import load_toml, save_toml

from younger_apps_dl.commons.help import generate_helping_for_pydantic_model


@click.group(name='younger-apps-dl')
def main():
    pass


@main.command(name='glance')
@click.option('--some-type', required=True, type=click.Choice(['models', 'datasets', 'engines', 'tasks'], case_sensitive=True), help='Indicates the type of task will be used.')
def glance(some_type: Literal['models', 'datasets', 'engines', 'tasks']):
    """
    Displays all possible candidates under a specific `type`.

    :param some_type: _description_
    :type some_type: Literal[&#39;models&#39;, &#39;datasets&#39;, &#39;engines&#39;, &#39;tasks&#39;]
    """
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
def option(task_kind, task_name, toml_path):
    """
    _summary_

    :param task_kind: _description_
    :type task_kind: _type_
    :param task_name: _description_
    :type task_name: _type_
    :raises exception: _description_
    """

    from younger_apps_dl.tasks import TASK_REGISTRY
    try:
        Task = TASK_REGISTRY[task_kind][task_name]
    except Exception as exception:
        click.echo(f'No <{task_kind}, {task_name}> Task in Task Registry.')
        raise exception

    helping_lines = generate_helping_for_pydantic_model(Task.OPTIONS)
    helping = '\n'.join(helping_lines)
    save_toml(helping, toml_path)


@main.command(name='launch')
@click.option('--task-kind', required=True, type=str, help='Indicates the type of task.')
@click.option('--task-name', required=True, type=str, help='Indicates the name of task.')
@click.option('--task-step', required=True, type=click.Choice(['train', 'evaluate', 'predict', 'preprocess', 'postprocess'], case_sensitive=True), help='Indicates the step of task.')
@click.option('--toml-path', required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path), default=None, help='Path to the configuration file.')
def launch(task_kind, task_name, task_step, toml_path):
    """
    _summary_

    :param task_kind: _description_
    :type task_kind: _type_
    :param task_name: _description_
    :type task_name: _type_
    :param task_step: _description_
    :type task_step: _type_
    :param options_filepath: _description_
    :type options_filepath: _type_
    :raises exception: _description_
    """

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
