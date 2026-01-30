#!/bin/bash

# 用户模块示例运行脚本

# 获取当前脚本所在目录（即用户模块目录）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==========================================="
echo "Younger Apps DL - 用户模块示例"
echo "==========================================="
echo ""
echo "用户模块目录: $SCRIPT_DIR"
echo ""

# 方法1: 使用环境变量
export YOUNGER_APPS_DL_USER_MODULES="$SCRIPT_DIR"

echo "1. 查看已注册的模型..."
echo "-------------------------------------------"
younger-apps-dl glance --some-type models | grep -A 5 "simple_mlp_example" || echo "  (查看所有模型请运行: younger-apps-dl glance --some-type models)"
echo ""

echo "2. 查看已注册的任务..."
echo "-------------------------------------------"
younger-apps-dl glance --some-type tasks | grep -A 5 "example" || echo "  (查看所有任务请运行: younger-apps-dl glance --some-type tasks)"
echo ""

echo "3. 生成配置文件..."
echo "-------------------------------------------"
CONFIG_FILE="/tmp/demo_config_$(date +%s).toml"
younger-apps-dl option \
    --task-kind example \
    --task-name simple_demo \
    --task-step train \
    --toml-path "$CONFIG_FILE"
echo "  配置文件已生成: $CONFIG_FILE"
echo ""

echo "4. 运行训练任务..."
echo "-------------------------------------------"
younger-apps-dl launch \
    --task-kind example \
    --task-name simple_demo \
    --task-step train \
    --toml-path "$CONFIG_FILE"
echo ""

echo "==========================================="
echo "示例运行完成！"
echo "==========================================="
echo ""
echo "你可以尝试："
echo "  - 修改 $SCRIPT_DIR/models/simple_mlp.py"
echo "  - 修改 $SCRIPT_DIR/tasks/demo_task.py"
echo "  - 添加新的模型或任务"
echo ""
echo "然后重新运行此脚本查看效果！"
