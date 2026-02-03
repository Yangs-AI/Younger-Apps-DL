#!/bin/sh

# Created time: 2026-02-01 20:31:04
# Author: Jason Young (杨郑鑫).
# E-Mail: AI.Jason.Young@outlook.com
# Last Modified by: Jason Young (杨郑鑫)
# Last Modified time: 2026-02-03 16:55:01
# Copyright (c) 2026 Yangs.AI
# 
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.

# Example script to run the user modules example

# Get the directory of the current script (i.e., the user modules directory)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==========================================="
echo "Younger Apps DL - User Modules Example"
echo "==========================================="
echo ""
echo "User modules directory: $SCRIPT_DIR/user_modules"
echo ""

# Method 1: Environment variable
export YADL_OPTIONAL_DIRPATH="$SCRIPT_DIR/user_modules"

echo "1. Loading registered models..."
echo "-------------------------------------------"
younger-apps-dl glance --some-type models
echo ""

echo "2. Loading registered tasks..."
echo "-------------------------------------------"
younger-apps-dl glance --some-type tasks
echo ""

echo "3. Generating configuration file..."
echo "-------------------------------------------"
CONFIG_FILE="/tmp/demo_config_$(date +%s).toml"
younger-apps-dl option \
    --task-kind optional \
    --task-name simple_demo \
    --task-step train \
    --toml-path "$CONFIG_FILE"
echo "Configuration file generated: $CONFIG_FILE"
echo ""

echo "4. Running training task..."
echo "-------------------------------------------"
younger-apps-dl launch \
    --task-kind optional \
    --task-name simple_demo \
    --task-step train \
    --toml-path "$CONFIG_FILE"
echo ""

echo "==========================================="
echo "Example finished!"
echo "==========================================="
echo ""
echo "You can try:"
echo "  - Modify $SCRIPT_DIR/models/simple_mlp.py"
echo "  - Modify $SCRIPT_DIR/tasks/demo_task.py"
echo "  - Add new models or tasks"
echo "  - Update __init__.py to import new modules"
echo ""
echo "Then rerun this script to see the changes!"

echo "You can also run command to test full pipeline:"
echo "   python -m user_modules.test"
