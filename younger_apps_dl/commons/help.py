#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-04-02 22:22:14
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-01-27 14:09:40
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import types

from pydantic import BaseModel
from pydantic_core import PydanticUndefined
from typing import get_origin, get_args, Any, Type


def remove_none(annotation: Any) -> Any:
    if get_origin(annotation) is types.UnionType:
        args = [arg for arg in get_args(annotation) if not is_none_type(arg)]
        if len(args) == 1:
            return args[0]
    return annotation


def is_base_model(data_type: type):
    return isinstance(data_type, type) and issubclass(data_type, BaseModel)


def is_str_type(data_type: type):
    return data_type == str


def is_int_type(data_type: type):
    return data_type == int


def is_float_type(data_type: type):
    return data_type == float


def is_bool_type(data_type: type):
    return data_type == bool


def is_any_type(data_type: type):
    return data_type == Any


def is_none_type(data_type: type):
    return data_type == type(None)


def get_placeholder(data_type: type | None):
    """Generate TOML-compatible placeholder for given type.

    All placeholders are wrapped in quotes to ensure valid TOML syntax.
    This allows the template to be loaded without syntax errors.
    Users should replace these with actual values.
    """
    if is_str_type(data_type):
        return '"<str>"'
    if is_int_type(data_type):
        return '"<int>"'  # Quoted to ensure valid TOML syntax
    if is_float_type(data_type):
        return '"<float>"'  # Quoted to ensure valid TOML syntax
    if is_bool_type(data_type):
        return '"<bool>"'  # Quoted to ensure valid TOML syntax

    if data_type is None:
        return '"<unknown>"'

    if get_origin(data_type) is not None:
        return '"<generic>"'

    return f'"<{getattr(data_type, "__name__", "special")}>"'


def generate_helping_for_pydantic_model(pydantic_model: Type[BaseModel], location: str = '', include_fields: list[str] | None = None) -> list[str]:
    """Generate TOML configuration template for a Pydantic model.

    Args:
        pydantic_model: The Pydantic model class to generate configuration for.
        location: The current location in the TOML hierarchy (for nested models).
        include_fields: If provided, only generate configuration for these field names.
                       If None, all fields will be included.

    Returns:
        List of TOML configuration lines.
    """
    toml_lines = list()
    nested_fields = list()
    global_fields = list()

    for name, annotation in pydantic_model.__annotations__.items():
        # Skip field if include_fields is specified and this field is not included
        if include_fields is not None and name not in include_fields:
            continue
        annotation = remove_none(annotation)
        origin = get_origin(annotation)

        default = pydantic_model.model_fields[name].default
        if isinstance(default, bool):
            default = f'{repr(default).lower()}'
        elif isinstance(default, str):
            default = f'"{default}"'
        else:
            default = f'{repr(default)}' if default is not None and default is not PydanticUndefined else ''
        description = pydantic_model.model_fields[name].description
        description = f' # {description}' if description is not None else ''

        if origin is list:
            element_type = get_args(annotation)[0] if len(get_args(annotation)) != 0 else Any
            element_type = remove_none(element_type)
            if is_base_model(element_type):
                nested_fields.append(('model_list', name, element_type))
            else:
                placeholder = get_placeholder(element_type)
                global_fields.append(f'{name} = {default if default else placeholder}{description}')
            continue

        if origin is dict:
            # Generaly, Dict is not a good data type for TOML and PyDantic.
            value_type = get_args(annotation)[1] if len(get_args(annotation)) != 0 else Any
            value_type = remove_none(value_type)
            if is_base_model(value_type):
                nested_fields.append(('model_dict', name, value_type))
            else:
                nested_fields.append(('inner_dict', name, value_type))
            continue

        if is_base_model(annotation):
            nested_fields.append(('model_self', name, annotation))
            continue

        placeholder = get_placeholder(annotation)
        global_fields.append(f'{name} = {default if default else placeholder}{description}')

    toml_lines.extend(global_fields)

    for index, (kind, name, field_type) in enumerate(nested_fields):
        # Get description for the field
        field_description = pydantic_model.model_fields[name].description
        description_comment = f'# {field_description}' if field_description is not None else f'# {name}'

        toml_lines.append('')  # Empty line for readability
        toml_lines.append(description_comment)

        if kind == 'model_self':
            section_name = f'{location + "." if location else ""}{name}'
            toml_lines.append(f'[{section_name}]')
            toml_lines.extend(generate_helping_for_pydantic_model(field_type, location=section_name))
        if kind == 'model_list':
            section_name = f'{location + "." if location else ""}{name}'
            toml_lines.append(f'[[{section_name}]]')
            toml_lines.extend(generate_helping_for_pydantic_model(field_type, location=section_name))
        if kind == 'model_dict':
            section_name_base = f'{location + "." if location else ""}{name}'
            example_key = '<key>'
            section_name = f'{section_name_base}.{example_key}'
            toml_lines.append(f'[{section_name}]')
            toml_lines.extend(generate_helping_for_pydantic_model(field_type, location=section_name))
        if kind == 'inner_dict':
            section_name = f'{location + "." if location else ""}{name}'
            toml_lines.append(f'[{section_name}]')
            value_type = get_origin(field_type)
            if value_type is list:
                element_type = get_args(value_type)[0] if len(get_args(value_type)) != 0 else Any
                element_type = remove_none(element_type)
                if is_base_model(element_type):
                    example_key = '<key>'
                    dict_list_section_name = f'{section_name}.{example_key}'
                    toml_lines.append(f'[[{dict_list_section_name}]]')
                    toml_lines.extend(generate_helping_for_pydantic_model(element_type, location=dict_list_section_name))
                else:
                    placeholder = get_placeholder(element_type)
                    toml_lines.append(f'<key> = [{placeholder}]')
            if value_type is dict:
                dict_dict_type = get_args(value_type)[1] if len(get_args(value_type)) > 1 else Any
                dict_dict_type = remove_none(dict_dict_type)
                placeholder = get_placeholder(dict_dict_type)
                toml_lines.append(f'<key> = {{ <subkey> = {dict_dict_type} }}')
            else:
                placeholder = get_placeholder(value_type)
                toml_lines.append(f'<key> = {placeholder}')
    return toml_lines
