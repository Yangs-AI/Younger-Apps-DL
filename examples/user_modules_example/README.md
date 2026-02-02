# User Modules Example

This is a complete user module example showing how to extend Younger Apps DL.

## Directory Structure

```
user_modules_example/
├── README.md (this file)
└── run.sh
└── user_modules/
    ├── __init__.py
    ├── models/
    │   └── simple_mlp.py
    └── tasks/
        └── demo_task.py
```

## Usage

### Method 1: Environment variable

```bash
# Set environment variable
export YADL_OPTIONAL_DIRPATH=/home/jason/Development/Younger/younger/apps/dl/examples/user_modules_example/user_modules

# 查看注册的组件
younger-apps-dl glance --some-type models
younger-apps-dl glance --some-type tasks

# 生成配置文件
younger-apps-dl option \
    --task-kind optional \
    --task-name simple_demo \
    --task-step train \
    --toml-path demo_config.toml

# 运行任务
younger-apps-dl launch \
    --task-kind optional \
    --task-name simple_demo \
    --task-step train \
    --toml-path demo_config.toml
```

### Method 2: CLI argument

```bash
# Specify user modules directory
younger-apps-dl --optional-dirpath ./user_modules glance --some-type tasks
```

### Method 3: Use the script

```bash
chmod +x run.sh
./run.sh
```

## Components

### SimpleMLP Model
- Registry name: `simple_mlp_example`
- A simple MLP
- Configurable input/hidden/output dimensions

### SimpleDemoTask Task
- Task kind: `optional`
- Task name: `simple_demo`
- Demonstrates a simple training task
- Demonstrates using a custom model

## Extend This Example

You can:
1. Add more models under `user_modules/models/`
2. Add more tasks under `user_modules/tasks/`
3. Create a `user_modules/datasets/` directory for custom datasets
4. Create a `user_modules/engines/` directory for custom engines

Remember: only modules imported by `__init__.py` will be loaded.
