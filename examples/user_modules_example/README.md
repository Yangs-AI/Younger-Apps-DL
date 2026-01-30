# 用户模块示例

这是一个完整的用户自定义模块示例，展示如何扩展 Younger Apps DL。

## 目录结构

```
user_modules_example/
├── README.md (本文件)
├── models/
│   └── simple_mlp.py
├── tasks/
│   └── demo_task.py
└── run_example.sh
```

## 使用方法

### 方法1: 使用环境变量

```bash
# 设置环境变量
export YOUNGER_APPS_DL_USER_MODULES=/home/jason/Development/Younger/younger/apps/dl/examples/user_modules_example

# 查看注册的组件
younger-apps-dl glance --some-type models
younger-apps-dl glance --some-type tasks

# 生成配置文件
younger-apps-dl option \
    --task-kind example \
    --task-name simple_demo \
    --task-step train \
    --toml-path demo_config.toml

# 运行任务
younger-apps-dl launch \
    --task-kind example \
    --task-name simple_demo \
    --task-step train \
    --toml-path demo_config.toml
```

### 方法2: 使用命令行参数

```bash
# 直接指定用户模块目录
younger-apps-dl --user-modules . glance --some-type tasks
```

### 方法3: 使用脚本

```bash
chmod +x run_example.sh
./run_example.sh
```

## 组件说明

### SimpleMLP 模型
- 注册名: `simple_mlp_example`
- 一个简单的多层感知机
- 可配置输入、隐藏层和输出维度

### SimpleDemoTask 任务
- Task kind: `example`
- Task name: `simple_demo`
- 演示了如何创建一个简单的训练任务
- 展示了如何使用自定义模型

## 扩展此示例

你可以：
1. 在 `models/` 下添加更多模型
2. 在 `tasks/` 下添加更多任务
3. 创建 `datasets/` 目录添加自定义数据集
4. 创建 `engines/` 目录添加自定义训练引擎

所有新添加的 Python 文件都会被自动发现和加载！
