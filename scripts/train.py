#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练脚本
============

本脚本使用 QLoRA 技术对 Qwen2.5-7B-Instruct 进行轻量化微调，使其适配心理咨询客服场景。

主要功能：
    - 加载 4-bit 量化的基座模型
    - 配置 LoRA 适配器
    - 加载 ChatML 格式的训练数据
    - 执行模型训练
    - 保存训练后的 LoRA 权重

运行方法：
    # 使用默认配置训练
    python scripts/train.py

    # 自定义参数
    python scripts/train.py --epochs 5 --learning-rate 1e-4 --batch-size 4

    # 从检查点恢复训练
    python scripts/train.py --resume-from output/checkpoints/checkpoint-1000

技术说明：
    - QLoRA: 结合 4-bit 量化和 LoRA，大幅降低显存需求
    - NF4: NormalFloat4 量化类型，适合正态分布的权重
    - 梯度检查点: 以时间换空间，进一步降低显存占用

作者：MentalChat 项目组
"""

# ==================== 标准库导入 ====================
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import asdict

# ==================== 项目内部导入 ====================
# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import config


# ==================== 数据集类 ====================

class ChatMLDataset:
    """
    ChatML 格式数据集类

    该类用于加载和处理 ChatML 格式的训练数据。

    ChatML 格式示例：
        {
            "messages": [
                {"role": "system", "content": "你是一位专业的心理咨询师..."},
                {"role": "user", "content": "我最近感觉很焦虑"},
                {"role": "assistant", "content": "我理解你的感受..."}
            ]
        }

    属性：
        tokenizer: 分词器实例
        max_length (int): 最大序列长度
        data (List): 加载的数据列表

    使用示例：
        >>> dataset = ChatMLDataset("train.jsonl", tokenizer, max_length=2048)
        >>> print(len(dataset))
        1000
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        verbose: bool = True
    ):
        """
        初始化数据集

        Args:
            data_path: 数据文件路径（JSONL 格式）
            tokenizer: 分词器实例
            max_length: 最大序列长度，超过将被截断
            verbose: 是否输出详细信息
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # ========== 加载数据 ==========
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

        if verbose:
            print(f"  加载数据: {len(self.data)} 条 from {data_path}")

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            包含 input_ids 和 labels 的字典

        处理流程：
        1. 获取 messages 字段
        2. 应用聊天模板格式化文本
        3. 进行 tokenize
        4. 设置 labels（对于因果语言模型，labels = input_ids）
        """
        item = self.data[idx]
        messages = item.get("messages", [])

        # ========== 应用聊天模板 ==========
        # 使用 tokenizer 的 apply_chat_template 方法格式化输入
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,           # 返回文本而非 token IDs
            add_generation_prompt=False  # 不添加生成提示
        )

        # ========== Tokenize ==========
        tokenized = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,          # 超长截断
            padding=False,            # 不填充（动态填充更高效）
            return_tensors=None       # 返回列表而非张量
        )

        # ========== 设置 labels ==========
        # 对于因果语言模型（Causal LM），labels 就是 input_ids
        # Trainer 会自动处理 mask
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized


# ==================== 模型加载函数 ====================

def load_model_and_tokenizer(model_path: str = None, use_4bit: bool = True):
    """
    加载模型和分词器

    加载 Qwen2.5 基座模型，并可选地进行 4-bit 量化。

    Args:
        model_path: 模型路径（本地路径或 HuggingFace 模型名）
        use_4bit: 是否使用 4-bit 量化（推荐开启以节省显存）

    Returns:
        Tuple[model, tokenizer]: 加载的模型和分词器

    加载流程：
        1. 确定模型路径
        2. 配置量化参数
        3. 加载分词器
        4. 加载模型
        5. 准备 k-bit 训练
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # ========== 确定模型路径 ==========
    model_path = model_path or config.model.base_model_path
    if not os.path.exists(model_path):
        model_path = config.model.base_model_name
        print(f"本地模型不存在，使用远程模型: {model_path}")

    print(f"加载模型: {model_path}")

    # ========== 配置 4-bit 量化 ==========
    # QLoRA 使用 NF4 量化 + 双量化来最大化压缩效果
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config.qlora.load_in_4bit,                    # 启用 4-bit 量化
            bnb_4bit_quant_type=config.qlora.bnb_4bit_quant_type,      # NF4 量化类型
            bnb_4bit_compute_dtype=getattr(torch, config.qlora.bnb_4bit_compute_dtype),  # 计算精度
            bnb_4bit_use_double_quant=config.qlora.bnb_4bit_use_double_quant,  # 双量化
        )
    else:
        bnb_config = None

    # ========== 加载分词器 ==========
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=config.model.trust_remote_code,  # 信任远程代码（Qwen 需要）
        use_fast=False  # 使用慢速分词器以确保兼容性
    )

    # 确保 pad_token 存在（Qwen 默认没有 pad_token）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  设置 pad_token = eos_token")

    # ========== 加载模型 ==========
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,                      # 量化配置
        device_map="auto",                                   # 自动分配设备
        trust_remote_code=config.model.trust_remote_code,
        torch_dtype=getattr(torch, config.model.torch_dtype),
    )

    # ========== 准备 k-bit 训练 ==========
    # 这一步对于 QLoRA 训练是必须的
    # 它会启用梯度检查点并设置正确的 dtype
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def setup_lora(model):
    """
    配置 LoRA 适配器

    为模型添加 LoRA 适配器，只训练低秩分解矩阵。

    Args:
        model: 基座模型

    Returns:
        配置了 LoRA 的模型

    LoRA 原理：
        - 冻结预训练权重
        - 在每层添加可训练的低秩分解矩阵
        - 大幅减少可训练参数量（通常 <1%）

    配置参数说明：
        - r: LoRA 秩，控制低秩矩阵的维度
        - lora_alpha: 缩放因子，实际缩放为 lora_alpha / r
        - target_modules: 应用 LoRA 的模块
        - lora_dropout: dropout 概率
    """
    from peft import LoraConfig, get_peft_model

    # 创建 LoRA 配置
    lora_config = LoraConfig(
        r=config.lora.r,                                    # LoRA 秩
        lora_alpha=config.lora.lora_alpha,                  # 缩放因子
        target_modules=config.lora.target_modules,          # 目标模块
        lora_dropout=config.lora.lora_dropout,              # dropout
        bias=config.lora.bias,                              # bias 处理方式
        task_type=config.lora.task_type,                    # 任务类型
    )

    # 应用 LoRA 配置
    model = get_peft_model(model, lora_config)

    # 打印可训练参数信息
    model.print_trainable_parameters()

    return model


# ==================== 训练函数 ====================

def train(
    model_path: str = None,
    data_dir: str = None,
    output_dir: str = None,
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    resume_from: str = None,
    use_4bit: bool = True,
    verbose: bool = True
):
    """
    执行模型训练

    完整的训练流程，包括：
        1. 加载模型和分词器
        2. 配置 LoRA 适配器
        3. 加载训练数据
        4. 配置训练参数
        5. 执行训练
        6. 保存模型

    Args:
        model_path: 基座模型路径
        data_dir: 训练数据目录
        output_dir: 输出目录
        epochs: 训练轮次
        batch_size: 批次大小
        learning_rate: 学习率
        resume_from: 从检查点恢复训练的路径
        use_4bit: 是否使用 4-bit 量化
        verbose: 是否输出详细信息

    Returns:
        Trainer: 训练完成后的 Trainer 对象
    """
    import torch
    from transformers import TrainingArguments, Trainer
    from datasets import Dataset

    # ========== 使用配置默认值 ==========
    model_path = model_path or config.model.base_model_path
    data_dir = data_dir or config.data.processed_data_dir
    output_dir = output_dir or config.training.output_dir
    epochs = epochs or config.training.num_train_epochs
    batch_size = batch_size or config.training.per_device_train_batch_size
    learning_rate = learning_rate or config.training.learning_rate

    # ========== 打印训练配置 ==========
    print("=" * 60)
    print("MentalChat 模型训练")
    print("=" * 60)
    print(f"模型路径: {model_path}")
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    print(f"训练轮次: {epochs}")
    print(f"批次大小: {batch_size}")
    print(f"学习率: {learning_rate}")
    print(f"梯度累积: {config.training.gradient_accumulation_steps}")
    print(f"梯度检查点: {config.training.gradient_checkpointing}")
    print()

    # ========== 1. 加载模型和分词器 ==========
    print("1. 加载模型和分词器...")
    model, tokenizer = load_model_and_tokenizer(model_path, use_4bit=use_4bit)

    # ========== 2. 配置 LoRA ==========
    print("\n2. 配置 LoRA...")
    model = setup_lora(model)

    # ========== 3. 加载数据 ==========
    print("\n3. 加载训练数据...")
    train_dataset = ChatMLDataset(
        os.path.join(data_dir, config.data.train_file),
        tokenizer,
        max_length=config.data.max_total_length,
        verbose=verbose
    )

    # 加载验证集（如果存在）
    eval_dataset = None
    eval_path = os.path.join(data_dir, config.data.valid_file)
    if os.path.exists(eval_path):
        eval_dataset = ChatMLDataset(
            eval_path,
            tokenizer,
            max_length=config.data.max_total_length,
            verbose=verbose
        )
        print(f"  加载验证集: {len(eval_dataset)} 条")

    # ========== 4. 配置训练参数 ==========
    print("\n4. 配置训练参数...")

    training_args = TrainingArguments(
        # 基本配置
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,

        # 学习率配置
        learning_rate=learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,

        # 精度配置
        fp16=config.training.fp16,
        bf16=config.training.bf16,

        # 显存优化配置（重要！）
        gradient_checkpointing=config.training.gradient_checkpointing,  # 梯度检查点
        optim=config.training.optim,                                     # 优化器（paged_adamw 适合 QLoRA）

        # 日志配置
        logging_steps=config.training.logging_steps,
        logging_dir=config.training.logs_dir,
        report_to=config.training.report_to,

        # 保存配置
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,

        # 评估配置
        eval_steps=config.training.eval_steps if eval_dataset else None,
        evaluation_strategy=config.training.evaluation_strategy if eval_dataset else "no",
        load_best_model_at_end=config.training.load_best_model_at_end if eval_dataset else False,

        # 其他配置
        max_grad_norm=config.training.max_grad_norm,  # 梯度裁剪
        seed=config.training.seed,                     # 随机种子
    )

    # ========== 5. 创建 Trainer ==========
    print("\n5. 创建 Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # ========== 6. 开始训练 ==========
    print("\n6. 开始训练...")
    print("-" * 60)
    trainer.train(resume_from_checkpoint=resume_from)

    # ========== 7. 保存最终模型 ==========
    print("\n7. 保存模型...")
    final_output_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print(f"   ✓ 模型已保存到: {final_output_dir}")

    # ========== 8. 完成 ==========
    print("\n" + "=" * 60)
    print("✓ 训练完成！")
    print("=" * 60)
    print(f"\n输出目录: {final_output_dir}")
    print(f"\n使用方法:")
    print(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"  from peft import PeftModel")
    print(f"  ")
    print(f"  # 加载基座模型")
    print(f"  base_model = AutoModelForCausalLM.from_pretrained('{config.model.base_model_name}')")
    print(f"  ")
    print(f"  # 加载 LoRA 权重")
    print(f"  model = PeftModel.from_pretrained(base_model, '{final_output_dir}')")

    return trainer


# ==================== 主函数 ====================

def main():
    """
    主函数 - 解析命令行参数并启动训练

    支持的命令行参数：
        --model-path: 基座模型路径
        --data-dir: 训练数据目录
        --output-dir: 输出目录
        --epochs: 训练轮次
        --batch-size: 批次大小
        --learning-rate: 学习率
        --resume-from: 从检查点恢复
        --no-4bit: 禁用 4-bit 量化
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="MentalChat 模型训练脚本 - 使用 QLoRA 技术微调心理咨询模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    # 使用默认配置训练
    python scripts/train.py

    # 自定义参数
    python scripts/train.py --epochs 5 --learning-rate 1e-4 --batch-size 4

    # 从检查点恢复训练
    python scripts/train.py --resume-from output/checkpoints/checkpoint-1000

    # 使用本地模型
    python scripts/train.py --model-path /path/to/model
        """
    )

    # 定义命令行参数
    parser.add_argument(
        "--model-path",
        type=str,
        default=config.model.base_model_path,
        help="基座模型路径（本地路径或 HuggingFace 模型名）"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=config.data.processed_data_dir,
        help="训练数据目录"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.training.output_dir,
        help="输出目录"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.training.num_train_epochs,
        help="训练轮次"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.training.per_device_train_batch_size,
        help="批次大小"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=config.training.learning_rate,
        help="学习率"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="从检查点恢复训练的路径"
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="禁用 4-bit 量化（不推荐，需要更多显存）"
    )

    # 解析参数
    args = parser.parse_args()

    # 执行训练
    train(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        resume_from=args.resume_from,
        use_4bit=not args.no_4bit
    )

    return 0


# ==================== 程序入口 ====================

if __name__ == "__main__":
    sys.exit(main())
