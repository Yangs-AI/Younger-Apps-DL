#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2024-08-27 18:03:43
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2025-12-26 01:31:25
# Copyright (c) 2024 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


import importlib.metadata

from younger.commons.constants import YoungerHandle


__version__ = importlib.metadata.version("younger_apps_dl")


__thename__ = YoungerHandle.AppsName + '-' + 'DL'


# Import submodules after __thename__ is defined to avoid circular import
