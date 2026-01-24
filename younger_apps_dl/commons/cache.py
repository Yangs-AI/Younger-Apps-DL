#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2025-01-05 20:58:35
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-01-24 15:18:39
# Copyright (c) 2025 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################


from younger.commons.cache import CACHE_ROOT
from younger.commons.constants import YoungerHandle


YADL_CACHE_ROOT = CACHE_ROOT.joinpath(f'{YoungerHandle.AppsName}-DL')
