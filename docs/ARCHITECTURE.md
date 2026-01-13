# Younger Apps DL Architecture

## Design Principles

### 1. Task-Centric Callback Pattern

#### Problem Statement
In a distributed training framework supporting both single-GPU and distributed (DDP) training scenarios, callbacks need to access shared resources (model, device, options, etc.). The challenge is designing an interface that is both flexible and maintainable.

#### The Constraint vs. Flexibility Trade-off
Without architectural constraints, callbacks would require excessive parameters:

```python
# Without constraints - parameter explosion
def train_fn(model, optimizer, device, options, dataloader, batch_size, ...):
    pass

def initialize_fn(model, options, device, ...):
    pass
```

This approach is neither maintainable nor scalable. Adding new features requires modifying 
callback signatures across all Tasks.

#### Solution: Task as Context Container
We enforce a clear design contract:

1. **Callbacks are methods of Task objects** (subclass of `BaseTask`)
2. **Shared state is accessed through `self`**, not parameters
3. **Engine only knows the callback interface**, not Task internals
4. **Separation of concerns**: Engine handles lifecycle, Task handles business logic

#### Implementation Pattern

```python
from younger_apps_dl.tasks import BaseTask
from typing import Any
import torch

class MyGraphEmbeddingTask(BaseTask):
    """Task implementation following the callback pattern."""

    def __init__(self, options):
        super().__init__(options)

    def _initialize_fn_(self, model: torch.nn.Module) -> None:
        """
        Called by engine after model setup (device placement + DDP wrapping).
        Synchronizes self.model with the framework-processed model.

        In single-GPU mode: model is on the specified device
        In distributed mode: model is wrapped with DistributedDataParallel

        Args:
            model: The model after all framework processing
        """
        self.model = model
        self.device_descriptor = next(self.model.parameters()).device

    def _train_fn_(self, minibatch: Any) ->  tuple[list[str], list[torch.Tensor], list[Callable[[float], str]]]:
        """
        Training step. Access shared state through self.

        Args:
            minibatch: The training batch

        Returns:
            Dictionary with training metrics
        """

        # self.model.train()
        # User do not need make train mode for model, engine handle it.
        minibatch = minibatch.to(self.device_descriptor)

        # Use self.model and self.options
        output = self.model(minibatch.x, minibatch.edge_index)
        loss = self.compute_loss(output, minibatch)

        train_metric_names = ['loss']
        train_metric_values = [loss]
        train_metric_formats = [lambda x: f'{x:.4f}']
        return train_metric_names, train_metric_values, train_metric_formats

    def _valid_fn_(self, dataloader: Any) -> dict[str, float]:
        """
        Validation step. Access shared state through self.

        Args:
            dataloader: The validation dataloader

        Returns:
            Dictionary with validation metrics
        """
        # self.model.eval()
        # User do not need make train mode for model, engine handle it.
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self._device_descriptor_)
                output = self.model(batch.x, batch.edge_index)
                loss = self.compute_loss(output, batch)
                total_loss += loss.item()

        return {"avg_loss": total_loss / len(dataloader)}
```

#### Engine's Responsibility

The Engine (`StandardTrainer`) handles the framework concerns:

1. **Device placement**: `model.to(device)`
2. **Distributed wrapping**: `DistributedDataParallel(model, ...)`
3. **Lifecycle management**: Calls `initialize_fn(model)` after all setup
4. **Data loading**: Manages dataloaders
5. **Training loop**: Calls `train_fn` and `valid_fn`

```python
class StandardTrainer(BaseEngine):
    def solo_train(self, model, train_fn, initialize_fn, ...):
        # Framework setup
        model.to(device)
        initialize_fn(model)  # Task syncs self.model

        # Training loop
        for epoch in range(epochs):
            for minibatch in dataloader:
                train_fn(minibatch)  # Task uses self.model

    def pile_train(self, model, train_fn, initialize_fn, ...):
        # Framework setup
        model = DDP(model, ...)
        initialize_fn(model)  # Task syncs self.model (now DDP-wrapped)

        # Training loop
        for epoch in range(epochs):
            for minibatch in dataloader:
                train_fn(minibatch)  # Task uses DDP-wrapped self.model
```

#### Why This Design Works

1. **Predictability**: Clear contract for implementing Tasks
   - Developers know exactly where to put code (methods of Task)
   - Developers know what to access (self.model, self.options, self.device)

2. **Maintainability**: Consistent structure across all Tasks
   - Adding new features doesn't break existing Tasks
   - Code patterns are uniform and recognizable

3. **Extensibility**: Easy to add new Tasks following the pattern
   - New developers can copy existing Task as template
   - Clear examples of how to implement callbacks

4. **Decoupling**: Clean separation between Engine and Task
   - Engine doesn't need to know Task implementation details
   - Engine only works with well-defined interfaces
   - Tasks can evolve independently

5. **Multi-process Safety**: Handles PyTorch's multiprocessing correctly
   - In distributed training, model objects are pickled and copied to worker processes
   - DDP wrapping happens after model.to(device) on each worker
   - `initialize_fn` provides the synchronization point for task state
   - `self.model` reference is always synchronized to the current process's model

#### Key Insight: Constraints Enable Abstraction

"Good frameworks embrace reasonable constraints to provide meaningful abstractions."
Without constraints, there's no abstraction-only parameter explosion and confusion.

This follows the principle:
> "There should be one-- and preferably only one --obvious way to do it." (The Zen of Python)

By establishing one clear pattern for Tasks, we:
- Reduce cognitive load on developers
- Improve code consistency
- Enable better tooling support
- Make debugging easier

---

## Framework Architecture Overview

### Component Hierarchy

```
Engine (StandardTrainer, StandardEvaluator, ...)
  ↓ (trains/evaluates/...)
Task (subclass of BaseTask)
  ↓ (provides)
Model + Data + Options + ...
```

### Data Flow in Training

**Single-GPU Mode:**
```
Task created with options
    ↓
trainer.run(model, ...)
    ↓
StandardTrainer.solo_train()
    ↓
model.to(device)
    ↓
initialize_fn(model) → Task.self.model = model
    ↓
for epoch, minibatch in training_loop:
    train_fn(minibatch) → uses self.model
    ↓
model.eval()
    ↓
valid_fn(dataloader) → uses self.model
```

**Distributed Mode (DDP):**
```
Task created with options
    ↓
trainer.run(model, ..., world_size=2)
    ↓
torch.multiprocessing.spawn() → 2 processes
    ↓
Each process gets a copy of Task (pickled)
    ↓
StandardTrainer.pile_train()
    ↓
model.to(device)
    ↓
model = DDP(model, ...)
    ↓
initialize_fn(model) → Task.self.model = DDP-wrapped model
    ↓
for epoch, minibatch in training_loop:
    train_fn(minibatch) → uses DDP-wrapped self.model
    ↓
model.eval()
    ↓
valid_fn(dataloader) → uses DDP-wrapped self.model
```

### Model Lifecycle

1. **Creation**: User creates model instance
2. **Passing to Engine**: `trainer.run(model, ...)`
3. **Device Placement**: `model.to(device)` (single-GPU) or each process (distributed)
4. **DDP Wrapping**: `DDP(model)` (distributed only)
5. **Task Synchronization**: `initialize_fn(model)` syncs `self.model` with framework-processed model
6. **Usage**: Task methods access `self.model` (already on device, already DDP-wrapped if needed)

**Critical Point**: After `initialize_fn` returns, `self.model` and engine's `model` reference 
the same object. Any modifications to the model state affect both references.

---

## Common Patterns

### Device Access Pattern

```python
def _initialize_fn_(self, model: torch.nn.Module) -> torch.device:
    self.model = model
    self.device_descriptor = next(self.model.parameters()).device

def _train_fn_(self, minibatch):
    minibatch = minibatch.to(self.device_descriptor)
```

### Multi-Model Pattern

```python
class MultiModelTask(BaseTask):
    def __init__(self, options):
        super().__init__(options)
        self.encoder = None
        self.decoder = None

    # Developers should also create a custom Trainer engine that handles multiple models.
    def _initialize_fn_(self, models: list[torch.nn.Module]):
        # models[0]: encoder, models[1]: decoder
        pass

    def _train_fn_(self, minibatch):
        # Use both models
        pass
```

---

## Related Files

- [BaseTask](../younger_apps_dl/tasks/__init__.py): Base class for all tasks
- [StandardTrainer](../younger_apps_dl/engines/trainers/standard.py): Standard Trainer engine
- [StandardEvaluator](../younger_apps_dl/engines/evaluators/standard.py): Standard Evaluator engine
- Example: [GraphEmbedding Task](../younger_apps_dl/tasks/ir/generation/basic_dag_generation.py)
