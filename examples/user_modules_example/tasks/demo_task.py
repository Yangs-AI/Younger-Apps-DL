#!/usr/bin/env python3
# -*- encoding=utf8 -*-

"""
简单的演示任务示例
"""

import torch
from pydantic import BaseModel, Field

# 从 younger_apps_dl 导入基类和注册装饰器
from younger_apps_dl.tasks import BaseTask, register_task
from younger_apps_dl.commons.logging import logger


class SimpleDemoTaskOptions(BaseModel):
    """SimpleDemoTask 的配置选项"""
    
    # 模型配置
    model_name: str = Field('simple_mlp_example', description='要使用的模型名称')
    input_dim: int = Field(128, description='模型输入维度')
    hidden_dim: int = Field(256, description='模型隐藏层维度')
    output_dim: int = Field(10, description='模型输出维度')
    
    # 训练配置
    learning_rate: float = Field(0.001, description='学习率')
    batch_size: int = Field(32, description='批次大小')
    num_epochs: int = Field(10, description='训练轮数')
    
    # 其他配置
    message: str = Field('Hello from SimpleDemoTask!', description='演示消息')


@register_task('example', 'simple_demo')
class SimpleDemoTask(BaseTask[SimpleDemoTaskOptions]):
    """
    一个简单的演示任务
    
    展示了如何:
    1. 定义任务配置
    2. 使用自定义模型
    3. 实现各个阶段的逻辑
    """
    
    OPTIONS = SimpleDemoTaskOptions
    
    STAGE_REQUIRED_OPTION = {
        'preprocess': [],
        'train': [],
        'evaluate': [],
        'predict': [],
        'postprocess': [],
    }
    
    def __init__(self, options: SimpleDemoTaskOptions):
        super().__init__(options)
        self.model = None
        self.optimizer = None
    
    def _preprocess_(self):
        """预处理阶段"""
        logger.info("=" * 60)
        logger.info("开始预处理...")
        logger.info(f"消息: {self.options.message}")
        logger.info("=" * 60)
        
        # 在这里实现你的数据预处理逻辑
        # 例如: 加载数据、清洗数据、特征工程等
        
        logger.info("预处理完成!")
    
    def _train_(self):
        """训练阶段"""
        logger.info("=" * 60)
        logger.info("开始训练...")
        logger.info(f"配置: LR={self.options.learning_rate}, "
                   f"Batch={self.options.batch_size}, "
                   f"Epochs={self.options.num_epochs}")
        logger.info("=" * 60)
        
        # 1. 构建模型
        self._build_model()
        
        # 2. 创建优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.options.learning_rate
        )
        logger.info(f"优化器: {self.optimizer}")
        
        # 3. 训练循环（这里是演示，实际应该使用真实数据）
        logger.info("开始训练循环...")
        for epoch in range(self.options.num_epochs):
            # 生成假数据用于演示
            dummy_input = torch.randn(self.options.batch_size, self.options.input_dim)
            dummy_target = torch.randint(0, self.options.output_dim, (self.options.batch_size,))
            
            # 前向传播
            output = self.model(dummy_input)
            
            # 计算损失
            loss = torch.nn.functional.cross_entropy(output, dummy_target)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 2 == 0:
                logger.info(f"  Epoch {epoch+1}/{self.options.num_epochs}, Loss: {loss.item():.4f}")
        
        logger.info("训练完成!")
    
    def _evaluate_(self):
        """评估阶段"""
        logger.info("=" * 60)
        logger.info("开始评估...")
        logger.info("=" * 60)
        
        if self.model is None:
            logger.warning("模型未构建，先构建模型")
            self._build_model()
        
        # 在这里实现你的评估逻辑
        # 例如: 加载测试数据、计算指标等
        
        self.model.eval()
        with torch.no_grad():
            # 生成假数据用于演示
            dummy_input = torch.randn(10, self.options.input_dim)
            output = self.model(dummy_input)
            logger.info(f"评估输出形状: {output.shape}")
        
        logger.info("评估完成!")
    
    def _predict_(self):
        """预测阶段"""
        logger.info("=" * 60)
        logger.info("开始预测...")
        logger.info("=" * 60)
        
        if self.model is None:
            logger.warning("模型未构建，先构建模型")
            self._build_model()
        
        # 在这里实现你的预测逻辑
        
        self.model.eval()
        with torch.no_grad():
            # 生成假数据用于演示
            dummy_input = torch.randn(5, self.options.input_dim)
            predictions = self.model(dummy_input)
            logger.info(f"预测结果形状: {predictions.shape}")
            logger.info(f"预测值 (前3个): {predictions[:3].argmax(dim=1).tolist()}")
        
        logger.info("预测完成!")
    
    def _postprocess_(self):
        """后处理阶段"""
        logger.info("=" * 60)
        logger.info("开始后处理...")
        logger.info("=" * 60)
        
        # 在这里实现你的后处理逻辑
        # 例如: 结果格式化、保存输出等
        
        logger.info("后处理完成!")
    
    def _build_model(self):
        """构建模型的辅助方法"""
        from younger_apps_dl.models import MODEL_REGISTRY
        
        logger.info(f"构建模型: {self.options.model_name}")
        
        if self.options.model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"模型 '{self.options.model_name}' 未注册。"
                f"可用模型: {list(MODEL_REGISTRY.keys())}"
            )
        
        model_class = MODEL_REGISTRY[self.options.model_name]
        self.model = model_class(
            input_dim=self.options.input_dim,
            hidden_dim=self.options.hidden_dim,
            output_dim=self.options.output_dim,
        )
        
        logger.info(f"模型构建完成: {type(self.model).__name__}")
        logger.info(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")


if __name__ == '__main__':
    # 测试任务
    print("测试 SimpleDemoTask...")
    
    # 创建任务实例
    options = SimpleDemoTaskOptions(
        input_dim=10,
        hidden_dim=20,
        output_dim=5,
        num_epochs=3
    )
    
    task = SimpleDemoTask(options)
    
    # 测试各个阶段
    print("\n--- 测试预处理 ---")
    task.preprocess()
    
    print("\n--- 测试训练 ---")
    task.train()
    
    print("\n--- 测试评估 ---")
    task.evaluate()
    
    print("\n--- 测试预测 ---")
    task.predict()
    
    print("\n--- 测试后处理 ---")
    task.postprocess()
    
    print("\n✓ 所有测试通过!")
