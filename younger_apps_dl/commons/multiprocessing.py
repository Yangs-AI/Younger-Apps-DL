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


class MultiprocessProgressManager:
    """
    Manager for controlling progress bars in multiprocessing contexts.
    
    Uses message queues to centralize progress bar management in the main process,
    preventing position conflicts and display issues.
    
    Example:
        ```python
        def worker_func(task_data, progress_queue, worker_id):
            # Initialize progress bar
            progress_queue.put(ProgressMessage(
                ProgressMessage.Type.INIT,
                worker_id,
                total=100,
                desc=f"Worker {worker_id}"
            ))
            
            for i in range(100):
                # Do work...
                
                # Update progress
                progress_queue.put(ProgressMessage(
                    ProgressMessage.Type.UPDATE,
                    worker_id,
                    n=1
                ))
            
            # Close progress bar
            progress_queue.put(ProgressMessage(
                ProgressMessage.Type.CLOSE,
                worker_id
            ))
            
            return result
        
        # Use the manager
        manager = MultiprocessProgressManager(num_workers=4)
        results = manager.run_with_progress(
            worker_func,
            tasks,
            num_workers=4
        )
        ```
    """
    
    def __init__(self, num_workers: int):
        """
        Initialize the progress manager.
        
        Args:
            num_workers: Number of worker processes
        """
        self.num_workers = num_workers
        self.progress_queue = None
        self.progress_bars = {}
    
    def _progress_listener(self):
        """Listen for progress messages and update progress bars accordingly."""
        while True:
            msg = self.progress_queue.get()
            
            if msg is None:  # Poison pill to stop the listener
                break
            
            if msg.type == ProgressMessage.Type.INIT:
                # Create a new progress bar at the specified position
                self.progress_bars[msg.worker_id] = tqdm.tqdm(
                    position=msg.worker_id,
                    leave=False,
                    **msg.data
                )
            
            elif msg.type == ProgressMessage.Type.UPDATE:
                # Update existing progress bar
                if msg.worker_id in self.progress_bars:
                    self.progress_bars[msg.worker_id].update(msg.data.get('n', 1))
            
            elif msg.type == ProgressMessage.Type.SET_DESC:
                # Set description for existing progress bar
                if msg.worker_id in self.progress_bars:
                    self.progress_bars[msg.worker_id].set_description(msg.data.get('desc', ''))
            
            elif msg.type == ProgressMessage.Type.CLOSE:
                # Close and remove progress bar
                if msg.worker_id in self.progress_bars:
                    self.progress_bars[msg.worker_id].close()
                    del self.progress_bars[msg.worker_id]
    
    def _cleanup_progress_bars(self):
        """Clean up any remaining progress bars."""
        for progress_bar in self.progress_bars.values():
            progress_bar.close()
        self.progress_bars.clear()
    
    def run_with_progress(
        self,
        worker_func: Callable,
        tasks: list[Any],
        pool_method: str = 'imap',
        **pool_kwargs
    ) -> list[Any]:
        """
        Run tasks with centralized progress bar management.
        
        Args:
            worker_func: Worker function that accepts (task_data, progress_queue, worker_id)
            tasks: List of tasks to process
            pool_method: Pool method to use ('imap', 'imap_unordered', 'map', etc.)
            **pool_kwargs: Additional keyword arguments for Pool
        
        Returns:
            List of results from worker function
        """
        # Create shared queue for progress messages
        manager = multiprocessing.Manager()
        self.progress_queue = manager.Queue()
        
        # Start progress listener in a separate thread
        import threading
        listener_thread = threading.Thread(target=self._progress_listener)
        listener_thread.start()
        
        try:
            # Prepare tasks with progress queue and worker IDs
            # For pool methods, we need to assign worker IDs based on task order
            tasks_with_queue = [
                (task, self.progress_queue, i % self.num_workers)
                for i, task in enumerate(tasks)
            ]
            
            # Run tasks in multiprocessing pool
            with multiprocessing.Pool(processes=self.num_workers) as pool:
                pool_func = getattr(pool, pool_method)
                if pool_method in ('imap', 'imap_unordered'):
                    results = list(pool_func(worker_func, tasks_with_queue, **pool_kwargs))
                else:
                    results = pool_func(worker_func, tasks_with_queue, **pool_kwargs)
        
        finally:
            # Send poison pill to stop listener
            self.progress_queue.put(None)
            listener_thread.join()
            
            # Clean up any remaining progress bars
            self._cleanup_progress_bars()
        
        return results


def create_progress_wrapper(func: Callable) -> Callable:
    """
    Decorator to wrap a worker function with progress reporting.
    
    The wrapped function should accept an additional 'progress_callback' parameter
    that can be called to report progress.
    
    Args:
        func: Original worker function
    
    Returns:
        Wrapped function that handles progress reporting via queue
    """
    def wrapper(args_tuple):
        task_data, progress_queue, worker_id = args_tuple
        
        def progress_callback(msg_type: ProgressMessage.Type, **kwargs):
            """Callback to send progress messages."""
            progress_queue.put(ProgressMessage(msg_type, worker_id, **kwargs))
        
        # Call original function with progress callback
        return func(task_data, progress_callback, worker_id)
    
    return wrapper
