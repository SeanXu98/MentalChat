#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目配置文件
集中管理所有训练、模型、数据处理相关的配置参数
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Any

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class ModelConfig:
    """模型配置"""
    # 基座模型
    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    # AutoDL 数据盘路径（避免占用系统盘空间）
    base_model_path: str = "/root/autodl-tmp/MentalChat/models/base/Qwen_Qwen2.5-7B-Instruct"

    # 是否使用本地模型
    use_local_model: bool = True

    # 模型精度
    torch_dtype: str = "float16"

    # 信任远程代码
    trust_remote_code: bool = True


@dataclass
class QLoRAConfig:
    """QLoRA 配置"""
    # 量化配置
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class LoRAConfig:
    """LoRA 配置"""
    # LoRA 参数
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """训练配置"""
    # 输出目录（AutoDL数据盘路径）
    output_dir: str = "/root/autodl-tmp/MentalChat/outputs/checkpoints"

    # 训练参数
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4

    # 学习率
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"

    # 精度
    fp16: bool = True
    bf16: bool = False

    # 日志和保存
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3

    # 评估
    evaluation_strategy: str = "steps"
    load_best_model_at_end: bool = True

    # 优化器
    optim: str = "paged_adamw_8bit"

    # 其他
    max_grad_norm: float = 1.0
    seed: int = 42
    report_to: str = "tensorboard"


@dataclass
class DataConfig:
    """数据配置"""
    # 数据路径
    raw_data_path: str = str(PROJECT_ROOT / "data" / "raw" / "data.csv")
    processed_data_dir: str = str(PROJECT_ROOT / "data" / "processed")

    # 训练数据 (JSONL 格式，ChatML 风格)
    train_file: str = "train.jsonl"
    valid_file: str = "valid.jsonl"
    test_file: str = "test.jsonl"

    # 数据划分比例
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    test_ratio: float = 0.1

    # 数据处理
    max_input_length: int = 512
    max_output_length: int = 512
    max_total_length: int = 2048

    # 数据格式
    data_format: str = "chatml"  # ChatML 格式

    # 系统提示词 (System Prompt)
    system_prompt: str = (
        "你是一名专业的心理咨询客服，请根据来访者的问题，给出专业、共情的回复。\n\n"
        "回复要求：\n"
        "1. 首先表达对来访者情绪的理解和接纳\n"
        "2. 分析来访者可能面临的问题\n"
        "3. 提供具体、可行的建议\n"
        "4. 以开放式问题或鼓励性话语引导来访者继续表达\n\n"
        "注意：保持温和、专业的语气，避免过于绝对的建议，尊重来访者的感受。"
    )


@dataclass
class AugmentConfig:
    """数据增强配置"""
    # 是否启用增强
    enabled: bool = False

    # 增强比例 (0-1)
    augment_ratio: float = 0.3

    # 增强策略
    strategies: List[str] = field(default_factory=lambda: ["paraphrase", "enhance"])

    # Qwen API 配置
    api_key: Optional[str] = None  # 从环境变量 DASHSCOPE_API_KEY 读取
    api_model: str = "qwen-plus"  # qwen-turbo, qwen-plus, qwen-max

    # 增强场景类型（用于场景扩展）
    scenario_types: List[str] = field(default_factory=lambda: ["青少年", "职场", "家庭", "学业"])


@dataclass
class InferenceConfig:
    """推理配置"""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True


@dataclass
class Config:
    """总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    qlora: QLoRAConfig = field(default_factory=QLoRAConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


# 默认配置实例
config = Config()


def get_config():
    """获取配置"""
    return config


def print_config():
    """打印配置"""
    import json
    from dataclasses import asdict

    print("="*60)
    print("项目配置")
    print("="*60)

    cfg_dict = {
        "model": asdict(config.model),
        "qlora": asdict(config.qlora),
        "lora": asdict(config.lora),
        "training": asdict(config.training),
        "data": asdict(config.data),
        "augment": asdict(config.augment),
        "inference": asdict(config.inference),
    }

    for section, params in cfg_dict.items():
        print(f"\n[{section.upper()}]")
        for key, value in params.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    print_config()
