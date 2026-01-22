#!/usr/bin/env python3
# -*- encoding=utf8 -*-

########################################################################
# Created time: 2026-01-22 11:52:00
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-01-22 11:52:00
# Copyright (c) 2026 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.
########################################################################

"""
Multiprocessing Progress Management Utilities

This module provides utilities for managing progress bars in multiprocessing contexts
using message queues. This approach centralizes progress bar control in the main process,
preventing position conflicts and display issues that can occur when multiple subprocess
workers try to manage their own progress bars.

Key Features:
- Centralized progress bar management via message queues
- Support for multiple concurrent progress bars with position management
- Clean API for subprocess workers to report progress
- Thread-safe progress bar updates from multiple workers
- Reusable components for any multiprocessing workflow with progress reporting

Usage Pattern:
1. Worker functions use a progress_callback to send progress messages
2. Progress messages are sent via a shared Queue to the main process
3. A listener thread in the main process receives messages and updates tqdm progress bars
4. Each worker is assigned a position to avoid progress bar conflicts

Example:
    ```python
    from younger_apps_dl.commons.multiprocessing import ProgressMessage
    
    def worker_func(parameters, progress_callback):
        # Initialize progress bar
        if progress_callback:
            progress_callback(ProgressMessage.Type.INIT, total=100, desc="Processing")
        
        for i in range(100):
            # Do work...
            
            # Update progress
            if progress_callback:
                progress_callback(ProgressMessage.Type.UPDATE, n=1)
        
        # Close progress bar
        if progress_callback:
            progress_callback(ProgressMessage.Type.CLOSE)
        
        return result
    
    # In main process:
    # Create wrapper that handles progress queue communication
    def wrapper(args_tuple):
        parameters, progress_queue, worker_id = args_tuple
        
        def progress_callback(msg_type, **kwargs):
            progress_queue.put(ProgressMessage(msg_type, worker_id, **kwargs))
        
        return worker_func(parameters, progress_callback)
    
    # Use with multiprocessing
    # (See _run_with_progress_management in standard.py for full example)
    ```
"""


import tqdm
import multiprocessing
from typing import Any, Callable
from enum import Enum


class ProgressMessage:
    """Message types for progress communication between processes."""
    
    class Type(Enum):
        INIT = "init"       # Initialize a progress bar
        UPDATE = "update"   # Update progress
        SET_DESC = "set_desc"  # Set description
        CLOSE = "close"     # Close a progress bar
    
    def __init__(self, msg_type: Type, worker_id: int, **kwargs):
        self.type = msg_type
        self.worker_id = worker_id
        self.data = kwargs
