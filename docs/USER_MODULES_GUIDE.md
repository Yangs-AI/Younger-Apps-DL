# 用户自定义模块扩展指南

## 概述

Younger Apps DL 允许用户通过指定目录的方式扩展自定义的 Models、Tasks、Datasets 和 Engines。你只需要创建一个目录，组织好文件结构，然后告诉 younger-apps-dl 去哪里找这些自定义模块即可。

## 快速开始

### 1. 创建用户模块目录结构

在你的项目中创建如下结构：

```
my_dl_modules/
├── models/
│   ├── my_model.py
│   └── another_model.py
├── tasks/
│   ├── my_task.py
│   └── custom_task.py
├── datasets/
│   └── my_dataset.py
└── engines/
    └── my_trainer.py
```

**注意**: 
- 目录名必须是 `models`、`tasks`、`datasets`、`engines` 之一
- 可以包含子目录，系统会递归扫描
- Python 文件名随意，不需要 `__init__.py`

### 2. 编写自定义组件

**示例1: 自定义 Model** (`my_dl_modules/models/my_model.py`):

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

**示例2: 自定义 Task** (`my_dl_modules/tasks/my_task.py`):

```python
from pydantic import BaseModel, Field
from younger_apps_dl.tasks import BaseTask, register_task
from younger_apps_dl.commons.logging import logger

class MyTaskOptions(BaseModel):
    model_name: str = Field('my_custom_gnn', description='模型名称')
    input_dim: int = Field(128, description='输入维度')
    hidden_dim: int = Field(256, description='隐藏层维度')
    output_dim: int = Field(10, description='输出维度')

@register_task('custom', 'classification')
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
        logger.info("预处理数据...")
        # 你的预处理逻辑
    
    def _train_(self):
        logger.info("开始训练...")
        # 你的训练逻辑
    
    def _evaluate_(self):
        logger.info("开始评估...")
        # 你的评估逻辑
    
    def _predict_(self):
        logger.info("开始预测...")
        # 你的预测逻辑
    
    def _postprocess_(self):
        logger.info("后处理...")
        # 你的后处理逻辑
```

**示例3: 自定义 Dataset** (`my_dl_modules/datasets/my_dataset.py`):

```python
import torch
from younger_apps_dl.datasets import register_dataset

@register_dataset('my_custom_dataset')
class MyCustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str):
        super().__init__()
        self.data_path = data_path
        # 加载数据
    
    def __len__(self):
        return 1000  # 示例
    
    def __getitem__(self, idx):
        # 返回数据项
        return torch.randn(128), torch.randint(0, 10, (1,))
```

## 使用方式

### 方法1: 命令行参数（推荐用于单次使用）

```bash
# 查看已注册的组件（包括用户自定义的）
younger-apps-dl --user-modules /path/to/my_dl_modules glance --some-type tasks

# 生成配置文件
younger-apps-dl --user-modules /path/to/my_dl_modules option \
    --task-kind custom \
    --task-name classification \
    --task-step train \
    --toml-path config.toml

# 运行任务
younger-apps-dl --user-modules /path/to/my_dl_modules launch \
    --task-kind custom \
    --task-name classification \
    --task-step train \
    --toml-path config.toml
```

### 方法2: 环境变量（推荐用于持久使用）

设置环境变量后，无需每次都指定 `--user-modules`：

```bash
# Linux/Mac
export YOUNGER_APPS_DL_USER_MODULES=/path/to/my_dl_modules

# Windows
set YOUNGER_APPS_DL_USER_MODULES=C:\path\to\my_dl_modules

# 然后直接使用
younger-apps-dl glance --some-type tasks
younger-apps-dl launch --task-kind custom --task-name classification --task-step train --toml-path config.toml
```

**支持多个目录**（用冒号分隔，Windows 用分号）：

```bash
# Linux/Mac
export YOUNGER_APPS_DL_USER_MODULES=/path/to/modules1:/path/to/modules2

# Windows
set YOUNGER_APPS_DL_USER_MODULES=C:\path\to\modules1;C:\path\to\modules2
```

### 方法3: 在项目中使用相对路径

如果你的用户模块和项目在一起：

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

在 `run.sh` 中：

```bash
#!/bin/bash
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
younger-apps-dl --user-modules "$PROJECT_DIR/custom_modules" launch \
    --task-kind custom \
    --task-name my_task \
    --task-step train \
    --toml-path "$PROJECT_DIR/configs/train.toml"
```

## 完整工作流示例

### 步骤1: 创建目录和文件

```bash
mkdir -p my_dl_modules/{models,tasks,datasets,engines}
```

### 步骤2: 编写自定义组件

创建 `my_dl_modules/models/simple_nn.py`:

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

创建 `my_dl_modules/tasks/demo_task.py`:

```python
from pydantic import BaseModel
from younger_apps_dl.tasks import BaseTask, register_task

class DemoOptions(BaseModel):
    message: str = "Hello from custom task!"

@register_task('demo', 'hello')
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

### 步骤3: 验证加载

```bash
export YOUNGER_APPS_DL_USER_MODULES=./my_dl_modules

# 查看models
younger-apps-dl glance --some-type models
# 应该看到 'simple_nn' 出现在列表中

# 查看tasks
younger-apps-dl glance --some-type tasks
# 应该看到 'demo' kind 下的 'hello' task
```

### 步骤4: 生成配置并运行

```bash
# 生成配置文件
younger-apps-dl option \
    --task-kind demo \
    --task-name hello \
    --task-step train \
    --toml-path demo_config.toml

# 运行任务
younger-apps-dl launch \
    --task-kind demo \
    --task-name hello \
    --task-step train \
    --toml-path demo_config.toml
```

## 高级用法

### 1. 组织复杂的模块结构

支持子目录和任意深度的嵌套：

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

所有 `.py` 文件都会被自动发现和加载。

### 2. 在自定义Task中使用自定义Model

```python
@register_task('custom', 'my_task')
class MyTask(BaseTask[MyTaskOptions]):
    def _train_(self):
        # 直接从 MODEL_REGISTRY 获取自定义模型
        from younger_apps_dl.models import MODEL_REGISTRY
        
        model_class = MODEL_REGISTRY['my_custom_gnn']
        self.model = model_class(
            input_dim=128,
            hidden_dim=256,
            output_dim=10
        )
        # ... 训练逻辑
```

### 3. 跨文件共享代码

创建工具模块（不包含注册装饰器）：

```
my_dl_modules/
├── models/
│   ├── my_model.py
│   └── utils.py  # 工具函数，没有 @register_*
└── tasks/
    └── my_task.py
```

在 `my_model.py` 中：

```python
import sys
from pathlib import Path

# 添加父目录到路径以便导入
sys.path.insert(0, str(Path(__file__).parent))

from utils import some_helper_function

@register_model('my_model')
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        some_helper_function()
```

## 常见问题

### Q1: 为什么我的模块没有被加载？

检查：
1. 目录结构是否正确（必须有 `models/tasks/datasets/engines` 这些名字）
2. Python 文件是否有语法错误
3. 是否正确使用了 `@register_*` 装饰器
4. 路径是否正确（使用绝对路径测试）

查看详细日志：
```bash
younger-apps-dl --user-modules /path/to/modules glance --some-type tasks --logging-filepath debug.log
cat debug.log
```

### Q2: 可以覆盖内置组件吗？

不推荐，但如果注册了相同名称的组件，后加载的会覆盖先加载的。用户模块在内置模块之后加载，所以理论上可以覆盖。但建议使用不同的名称。

### Q3: 依赖其他包怎么办？

正常导入即可：

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

确保这些依赖已安装在你的环境中。

### Q4: 可以在配置文件中指定用户模块路径吗？

目前暂不支持在 TOML 配置文件中指定。请使用命令行参数或环境变量。

### Q5: 如何调试自定义模块？

1. 添加打印语句或使用 `logger`
2. 使用 Python 调试器（pdb）
3. 查看日志文件

```python
from younger_apps_dl.commons.logging import logger

@register_model('debug_model')
class DebugModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        logger.info("DebugModel initialized!")
        # 或者
        print("DebugModel initialized!")
```

## 最佳实践

1. **使用清晰的命名**：为你的组件使用描述性的名称
2. **组织代码**：将相关的模型、任务放在一起
3. **文档化**：为你的类添加文档字符串
4. **测试**：在注册前先单独测试你的类
5. **版本控制**：将自定义模块放入 git 版本控制
6. **日志记录**：使用 `younger_apps_dl.commons.logging.logger` 记录重要信息

## 总结

通过用户模块目录功能，你可以：
- ✅ 无需修改 younger_apps_dl 源码
- ✅ 无需创建 Python 包和 pyproject.toml
- ✅ 灵活组织你的自定义代码
- ✅ 轻松在不同项目间复用组件
- ✅ 快速原型开发和实验

只需创建目录、编写代码、指定路径，就可以像使用内置组件一样使用你的自定义组件！
