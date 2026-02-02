# User Custom Modules Guide

## Overview

Younger Apps DL allows you to extend custom Models, Tasks, Datasets, and Engines by providing a directory. You only need to create a directory and provide an entry file `register.py` that imports the modules you want to register.

## Quick Start

### 1. Create the user module directory

Create the following structure in your project:

```
my_dl_modules/
├── register.py
├── models/
│   ├── __init__.py
│   ├── my_model.py
│   └── another_model.py
├── tasks/
│   ├── __init__.py
│   ├── my_task.py
│   └── custom_task.py
├── datasets/
│   ├── __init__.py
│   └── my_dataset.py
└── engines/
    ├── __init__.py
    └── my_trainer.py
```

**Notes**:
- `register.py` is required at the root of the directory.
- `register.py` should only import modules to trigger registration. Avoid heavy runtime logic.
- Directory names are not enforced (`models/`, `tasks/`, etc. are just conventions).

### 2. Write your custom components

**Example 1: Custom Model** (`my_dl_modules/models/my_model.py`):

```python
import torch
from younger_apps_dl.models import register_model

@register_model('my_custom_gnn')
class MyCustomGNN(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        return self.fc2(x)
```

**Example 2: Custom Task** (`my_dl_modules/tasks/my_task.py`):

One can define a custom task by subclassing `BaseTask` and using the `@register_task` decorator.
The kind for user-defined tasks should be set to `'optional'`.

```python
from pydantic import BaseModel, Field
from younger_apps_dl.tasks import BaseTask, register_task
from younger_apps_dl.commons.logging import logger

class MyTaskOptions(BaseModel):
    model_name: str = Field('my_custom_gnn', description='Model name')
    input_dim: int = Field(128, description='Input dimension')
    hidden_dim: int = Field(256, description='Hidden dimension')
    output_dim: int = Field(10, description='Output dimension')

@register_task('optional', 'classification')
class MyClassificationTask(BaseTask[MyTaskOptions]):
    OPTIONS = MyTaskOptions
    STAGE_REQUIRED_OPTION = {
        'preprocess': [],
        'train': [],
        'evaluate': [],
        'predict': [],
        'postprocess': [],
    }

    def _preprocess_(self):
        logger.info("Preprocessing data...")
        # Your preprocessing logic

    def _train_(self):
        logger.info("Start training...")
        # Your training logic

    def _evaluate_(self):
        logger.info("Start evaluation...")
        # Your evaluation logic

    def _predict_(self):
        logger.info("Start prediction...")
        # Your prediction logic

    def _postprocess_(self):
        logger.info("Post-processing...")
        # Your post-processing logic
```

**Example 3: Custom Dataset** (`my_dl_modules/datasets/my_dataset.py`):

```python
import torch
from younger_apps_dl.datasets import register_dataset

@register_dataset('my_custom_dataset')
class MyCustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str):
        super().__init__()
        self.data_path = data_path
        # Load data

    def __len__(self):
        return 1000  # Example

    def __getitem__(self, idx):
        # Return one data item
        return torch.randn(128), torch.randint(0, 10, (1,))
```

## Usage

### Method 1: CLI argument (recommended for one-off usage)

```bash
# List registered components (including user modules)
younger-apps-dl --optional-dirpath /path/to/my_dl_modules glance --some-type tasks

# Generate a config template
younger-apps-dl --optional-dirpath /path/to/my_dl_modules option \
    --task-kind optional \
    --task-name classification \
    --task-step train \
    --toml-path config.toml

# Run a task
younger-apps-dl --optional-dirpath /path/to/my_dl_modules launch \
    --task-kind optional \
    --task-name classification \
    --task-step train \
    --toml-path config.toml
```

### Method 2: Environment variable (recommended for persistent usage)

After setting the env var, you don't need to pass `--optional-dirpath` every time:

```bash
# Linux/Mac
export YADL_OPTIONAL_DIRPATH=/path/to/my_dl_modules

# Windows
set YADL_OPTIONAL_DIRPATH=C:\path\to\my_dl_modules

# Then use it directly
younger-apps-dl glance --some-type tasks
younger-apps-dl launch --task-kind optional --task-name classification --task-step train --toml-path config.toml
```

Currently only a single directory is supported.

### Method 3: Use a relative path in your project

If your user modules live alongside your project:

```
my_project/
├── configs/
│   └── train.toml
├── custom_modules/
│   ├── models/
│   │   └── my_model.py
│   └── tasks/
│       └── my_task.py
└── run.sh
```

In `run.sh`:

```bash
#!/bin/bash
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
younger-apps-dl --optional-dirpath "$PROJECT_DIR/custom_modules" launch \
    --task-kind optional \
    --task-name my_task \
    --task-step train \
    --toml-path "$PROJECT_DIR/configs/train.toml"
```

## End-to-End Example

### Step 1: Create directories and files

```bash
mkdir -p my_dl_modules/{models,tasks,datasets,engines}
touch my_dl_modules/register.py
```

### Step 2: Write custom components

Create `my_dl_modules/models/simple_nn.py`:

```python
import torch
from younger_apps_dl.models import register_model

@register_model('simple_nn')
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)
```

Create `my_dl_modules/tasks/demo_task.py`:

```python
from pydantic import BaseModel
from younger_apps_dl.tasks import BaseTask, register_task

class DemoOptions(BaseModel):
    message: str = "Hello from custom task!"

@register_task('optional', 'hello')
class DemoTask(BaseTask[DemoOptions]):
    OPTIONS = DemoOptions
    STAGE_REQUIRED_OPTION = {'preprocess': [], 'train': [], 'evaluate': [], 'predict': [], 'postprocess': []}

    def _preprocess_(self):
        print(f"Preprocess: {self.options.message}")

    def _train_(self):
        print(f"Train: {self.options.message}")

    def _evaluate_(self):
        print(f"Evaluate: {self.options.message}")

    def _predict_(self):
        print(f"Predict: {self.options.message}")

    def _postprocess_(self):
        print(f"Postprocess: {self.options.message}")
```

### Step 3: Write register.py

In `my_dl_modules/register.py`, import the modules to be registered:

```python
from models import simple_nn
from tasks import demo_task
```

### Step 4: Verify loading

```bash
export YADL_OPTIONAL_DIRPATH=./my_dl_modules

# List models
younger-apps-dl glance --some-type models
# You should see 'simple_nn'

# List tasks
younger-apps-dl glance --some-type tasks
# You should see 'hello' under kind 'optional'
```

### Step 5: Generate config and run

```bash
# Generate a config file
younger-apps-dl option \
    --task-kind optional \
    --task-name hello \
    --task-step train \
    --toml-path demo_config.toml

# Run the task
younger-apps-dl launch \
    --task-kind optional \
    --task-name hello \
    --task-step train \
    --toml-path demo_config.toml
```

## Advanced Usage

### 1. Organize complex module structures

Nested directories are fine:

```
my_dl_modules/
├── models/
│   ├── vision/
│   │   ├── cnn.py
│   │   └── transformer.py
│   └── nlp/
│       └── bert.py
└── tasks/
    ├── computer_vision/
    │   ├── classification.py
    │   └── detection.py
    └── nlp/
        └── text_generation.py
```

Only modules imported by `register.py` will be loaded (no recursive scan).

### 2. Use a custom Model inside a custom Task

```python
@register_task('optional', 'my_task')
class MyTask(BaseTask[MyTaskOptions]):
    def _train_(self):
        # Retrieve a custom model from the registry directly
        from younger_apps_dl.models import MODEL_REGISTRY

        model_class = MODEL_REGISTRY['my_custom_gnn']
        self.model = model_class(
            input_dim=128,
            hidden_dim=256,
            output_dim=10
        )
        # ... training logic
```

### 3. Share code across files

Create a utility module (without registration decorators):

```
my_dl_modules/
├── models/
│   ├── my_model.py
│   └── utils.py  # helper functions, no @register_*
└── tasks/
    └── my_task.py
```

In `my_model.py`:

```python
from .utils import some_helper_function

@register_model('my_model')
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        some_helper_function()
```

## FAQ

### Q1: Why aren't my modules loaded?

Check:
1. `register.py` exists and imports the modules you want to register
2. The Python files have no syntax errors
3. You used the correct `@register_*` decorators
4. The path is correct (test with an absolute path)

View detailed logs:
```bash
younger-apps-dl --optional-dirpath /path/to/modules glance --some-type tasks --logging-filepath debug.log
cat debug.log
```

### Q2: Can I override built-in components?

Not recommended. If you register a component with the same name, the later one wins. User modules are loaded after built-ins, so override is possible, but use unique names if possible.

### Q3: What about external dependencies?

Just import them as usual:

```python
import numpy as np
import pandas as pd
from transformers import BertModel

@register_model('my_bert')
class MyBert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
```

Make sure those dependencies are installed.

### Q4: Can I specify the user module path in TOML config?

Not supported yet. Use the CLI argument or the environment variable.

### Q5: How do I debug custom modules?

1. Add print statements or use `logger`
2. Use the Python debugger (pdb)
3. Write logs to a file

```python
from younger_apps_dl.commons.logging import logger

@register_model('debug_model')
class DebugModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        logger.info("DebugModel initialized!")
        # Or
        print("DebugModel initialized!")
```

## Best Practices

1. **Use clear names**: Give your components descriptive names
2. **Organize code**: Group related models and tasks together
3. **Document**: Add docstrings to your classes
4. **Test**: Test your classes before registration
5. **Version control**: Put your custom modules under git
6. **Logging**: Use `younger_apps_dl.commons.logging.logger` for important logs

## Summary

With user module directories, you can:
- ✅ Extend without modifying Younger Apps DL source
- ✅ Avoid creating a Python package or pyproject.toml
- ✅ Organize custom code flexibly
- ✅ Reuse components across projects
- ✅ Prototype quickly

Create a directory, write your modules, add `register.py`, and point the CLI to it — your custom components will work like built-ins.
