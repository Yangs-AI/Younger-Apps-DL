#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-04-07 20:09:32
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-01-14 22:03:22
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


# The graph datasets in LogicX format are implemented here.
# They are designed to work seamlessly with the DAG-based models and generation tasks.
# YADL only supports the processing of LogicX graph datasets (not including raw data download).


from .dag import *
from .dag_with_labels import *
