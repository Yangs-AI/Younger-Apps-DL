#!/usr/bin/env python3
# -*- encoding=utf8 -*-

"""
测试用户自定义模块加载功能
"""

import sys
import tempfile
from pathlib import Path


def create_test_user_modules(base_dir: Path):
    """创建测试用的用户模块"""
    
    # 创建目录结构
    models_dir = base_dir / 'models'
    tasks_dir = base_dir / 'tasks'
    models_dir.mkdir(parents=True)
    tasks_dir.mkdir(parents=True)
    
    # 创建自定义模型
    model_code = '''
import torch
from younger_apps_dl.models import register_model

@register_model('test_user_model')
class TestUserModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)
'''
    (models_dir / 'test_model.py').write_text(model_code)
    
    # 创建自定义任务
    task_code = '''
from pydantic import BaseModel
from younger_apps_dl.tasks import BaseTask, register_task

class TestTaskOptions(BaseModel):
    param1: int = 42

@register_task('test_user', 'test_task')
class TestUserTask(BaseTask[TestTaskOptions]):
    OPTIONS = TestTaskOptions
    STAGE_REQUIRED_OPTION = {
        'preprocess': [],
        'train': [],
        'evaluate': [],
        'predict': [],
        'postprocess': [],
    }
    
    def _preprocess_(self):
        print("Test task preprocessing")
    
    def _train_(self):
        print("Test task training")
    
    def _evaluate_(self):
        print("Test task evaluating")
    
    def _predict_(self):
        print("Test task predicting")
    
    def _postprocess_(self):
        print("Test task postprocessing")
'''
    (tasks_dir / 'test_task.py').write_text(task_code)
    
    print(f"✓ 创建测试模块于: {base_dir}")
    print(f"  - {models_dir / 'test_model.py'}")
    print(f"  - {tasks_dir / 'test_task.py'}")


def test_user_modules():
    """测试用户模块加载"""
    print("=== 测试用户模块加载系统 ===\n")
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir) / 'test_user_modules'
        test_dir.mkdir()
        
        # 创建测试文件
        create_test_user_modules(test_dir)
        print()
        
        # 测试加载
        print("1. 测试加载用户模块...")
        try:
            from younger_apps_dl.commons.optional import load_user_modules
            load_user_modules(test_dir)
            print("   ✓ 加载函数执行成功\n")
        except Exception as e:
            print(f"   ✗ 加载失败: {e}\n")
            import traceback
            traceback.print_exc()
            return False
        
        # 验证注册
        print("2. 验证组件是否已注册...")
        try:
            from younger_apps_dl.models import MODEL_REGISTRY
            from younger_apps_dl.tasks import TASK_REGISTRY
            
            # 检查模型
            if 'test_user_model' in MODEL_REGISTRY:
                print("   ✓ 自定义模型 'test_user_model' 已注册")
                print(f"     类: {MODEL_REGISTRY['test_user_model']}")
            else:
                print("   ✗ 自定义模型未找到")
                print(f"     当前注册的模型: {list(MODEL_REGISTRY.keys())}")
                return False
            
            # 检查任务
            if 'test_user' in TASK_REGISTRY and 'test_task' in TASK_REGISTRY['test_user']:
                print("   ✓ 自定义任务 'test_user/test_task' 已注册")
                print(f"     类: {TASK_REGISTRY['test_user']['test_task']}")
            else:
                print("   ✗ 自定义任务未找到")
                print(f"     当前注册的任务kinds: {list(TASK_REGISTRY.keys())}")
                return False
            
            print()
        except Exception as e:
            print(f"   ✗ 验证失败: {e}\n")
            import traceback
            traceback.print_exc()
            return False
        
        # 测试实例化
        print("3. 测试实例化自定义组件...")
        try:
            import torch
            
            # 实例化模型
            model_class = MODEL_REGISTRY['test_user_model']
            model = model_class()
            print(f"   ✓ 模型实例化成功: {type(model)}")
            
            # 测试前向传播
            x = torch.randn(5, 10)
            output = model(x)
            print(f"   ✓ 模型前向传播成功: input {x.shape} -> output {output.shape}")
            
            # 实例化任务
            task_class = TASK_REGISTRY['test_user']['test_task']
            task = task_class(task_class.OPTIONS())
            print(f"   ✓ 任务实例化成功: {type(task)}")
            
            print()
        except Exception as e:
            print(f"   ✗ 实例化失败: {e}\n")
            import traceback
            traceback.print_exc()
            return False
    
    print("=== 所有测试通过! ===")
    return True


if __name__ == '__main__':
    success = test_user_modules()
    sys.exit(0 if success else 1)
