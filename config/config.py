#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目配置文件
============

本模块集中管理所有训练、模型、数据处理相关的配置参数。
这是项目的唯一配置源，所有脚本都从这里读取配置。

设计理念：
    - 使用 dataclass 定义配置类，类型安全且易于序列化
    - 分层设计：模型配置、LoRA配置、训练配置、数据配置等
    - 支持通过环境变量覆盖部分配置

使用方法：
    from config import config, get_config, print_config

    # 访问配置
    print(config.model.base_model_path)
    print(config.training.learning_rate)
    print(config.data.system_prompt)

    # 打印所有配置
    print_config()

配置分类：
    - ModelConfig: 基座模型配置
    - QLoRAConfig: 4-bit 量化配置
    - LoRAConfig: LoRA 适配器配置
    - TrainingConfig: 训练参数配置
    - DataConfig: 数据处理配置
    - AugmentConfig: 数据增强配置
    - InferenceConfig: 推理参数配置

作者：MentalChat 项目组
"""

# ==================== 标准库导入 ====================
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# ==================== 路径配置 ====================
# 项目根目录（配置文件所在目录的上一级）
PROJECT_ROOT = Path(__file__).parent.parent


# ==================== 模型配置类 ====================

@dataclass
class ModelConfig:
    """
    基座模型配置类

    管理与基座模型相关的所有配置参数。

    属性：
        base_model_name (str): HuggingFace 模型名称，用于远程下载
        base_model_path (str): 本地模型路径，优先使用
        use_local_model (bool): 是否优先使用本地模型
        torch_dtype (str): 模型权重的数据类型
        trust_remote_code (bool): 是否信任远程代码（Qwen 模型需要）

    说明：
        - AutoDL 环境建议将模型放在 /root/autodl-tmp/ 目录下（数据盘）
        - torch_dtype 建议 float16，兼顾精度和显存
    """
    # ========== 基座模型配置 ==========
    # HuggingFace 模型名称（用于远程下载）
    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct"

    # 本地模型路径（AutoDL 数据盘路径，避免占用系统盘空间）
    base_model_path: str = "/root/autodl-tmp/MentalChat/models/base/Qwen_Qwen2.5-7B-Instruct"

    # 是否优先使用本地模型
    use_local_model: bool = True

    # 模型权重的数据类型
    # 可选: float32, float16, bfloat16
    torch_dtype: str = "float16"

    # 是否信任远程代码（Qwen 模型需要设置为 True）
    trust_remote_code: bool = True


# ==================== QLoRA 配置类 ====================

@dataclass
class QLoRAConfig:
    """
    QLoRA 量化配置类

    管理 4-bit 量化的相关配置参数。

    属性：
        load_in_4bit (bool): 是否启用 4-bit 量化
        bnb_4bit_quant_type (str): 量化类型
        bnb_4bit_compute_dtype (str): 计算时的数据类型
        bnb_4bit_use_double_quant (bool): 是否启用双量化

    量化原理：
        - NF4 (NormalFloat4): 专为正态分布权重设计的 4-bit 量化类型
        - 双量化: 对量化常数再次量化，进一步减少显存占用
        - 计算类型: 实际计算时使用更高精度（float16）以保证训练稳定性

    显存估算：
        - 7B 模型 FP16: ~14GB
        - 7B 模型 4-bit: ~4GB
    """
    # ========== 量化配置 ==========
    # 是否启用 4-bit 量化（QLoRA 的核心）
    load_in_4bit: bool = True

    # 量化类型，可选: "nf4"（推荐）, "fp4"
    # NF4 适合正态分布的权重，效果更好
    bnb_4bit_quant_type: str = "nf4"

    # 计算时的数据类型
    # 虽然 weight 是 4-bit，但计算时使用 float16 保证精度
    bnb_4bit_compute_dtype: str = "float16"

    # 是否启用双量化（Double Quantization）
    # 对量化常数再次量化，可额外节省约 0.5GB 显存
    bnb_4bit_use_double_quant: bool = True


# ==================== LoRA 配置类 ====================

@dataclass
class LoRAConfig:
    """
    LoRA 适配器配置类

    管理 LoRA（Low-Rank Adaptation）的相关配置参数。

    属性：
        r (int): LoRA 秩（rank），控制低秩矩阵的维度
        lora_alpha (int): 缩放因子，实际缩放为 lora_alpha / r
        target_modules (List[str]): 应用 LoRA 的模块列表
        lora_dropout (float): LoRA 层的 dropout 概率
        bias (str): bias 参数的处理方式
        task_type (str): 任务类型

    LoRA 原理：
        - 在预训练模型的权重矩阵旁添加低秩分解矩阵
        - 冻结原始权重，只训练低秩矩阵
        - 可训练参数量通常 <1%，大幅降低训练成本

    参数选择建议：
        - r=16, alpha=32: 通用选择，平衡效果和参数量
        - r=32, alpha=64: 更强的表达能力，但参数量增加
        - target_modules: 建议至少包含 q_proj, v_proj
    """
    # ========== LoRA 核心参数 ==========
    # LoRA 秩（rank），控制低秩矩阵的维度
    # 值越大，可训练参数越多，表达能力越强
    r: int = 16

    # 缩放因子
    # 实际缩放为 lora_alpha / r，通常设置为 r 的 2 倍
    lora_alpha: int = 32

    # 应用 LoRA 的目标模块
    # q_proj, v_proj 是最常见的选择
    # 也可以添加 k_proj, o_proj, gate_proj, up_proj, down_proj
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # LoRA 层的 dropout 概率
    # 0.05 是一个常用的值，防止过拟合
    lora_dropout: float = 0.05

    # bias 参数的处理方式
    # 可选: "none"（不训练）, "all"（全部训练）, "lora_only"（只训练 LoRA 的 bias）
    bias: str = "none"

    # 任务类型
    # CAUSAL_LM: 因果语言模型（GPT 风格）
    task_type: str = "CAUSAL_LM"


# ==================== 训练配置类 ====================

@dataclass
class TrainingConfig:
    """
    训练配置类

    管理模型训练过程中的所有参数。

    属性：
        output_dir (str): 模型输出目录
        logs_dir (str): 日志输出目录
        num_train_epochs (int): 训练轮次
        per_device_train_batch_size (int): 每设备的训练批次大小
        gradient_accumulation_steps (int): 梯度累积步数
        learning_rate (float): 学习率
        等等...

    显存优化策略：
        1. 减小 batch_size
        2. 增大 gradient_accumulation_steps（等效于增大 batch）
        3. 启用 gradient_checkpointing（以时间换空间）
        4. 使用 paged_adamw_8bit 优化器
    """
    # ========== 输出目录配置 ==========
    # 模型检查点输出目录
    output_dir: str = str(PROJECT_ROOT / "output" / "checkpoints")
    # 日志输出目录（TensorBoard 等）
    logs_dir: str = str(PROJECT_ROOT / "output" / "logs")

    # ========== 训练参数 ==========
    # 训练轮次（完整遍历数据集的次数）
    num_train_epochs: int = 3

    # 每设备的训练批次大小
    # 显存不足时可减小此值
    per_device_train_batch_size: int = 4

    # 每设备的评估批次大小
    per_device_eval_batch_size: int = 4

    # 梯度累积步数
    # 实际 batch_size = per_device_batch_size * gradient_accumulation_steps
    gradient_accumulation_steps: int = 4

    # ========== 显存优化配置 ==========
    # 梯度检查点（以计算时间换显存空间）
    # 启用后可显著减少显存占用，但会增加约 20% 的训练时间
    gradient_checkpointing: bool = True

    # ========== 学习率配置 ==========
    # 学习率
    # QLoRA 推荐使用较高的学习率，如 2e-4
    learning_rate: float = 2e-4

    # 权重衰减（L2 正则化）
    weight_decay: float = 0.01

    # 预热比例
    # 前 10% 的训练步数会逐渐增加学习率
    warmup_ratio: float = 0.1

    # 学习率调度器类型
    # cosine: 余弦退火（推荐）
    # linear: 线性衰减
    # constant: 恒定
    lr_scheduler_type: str = "cosine"

    # ========== 精度配置 ==========
    # 是否使用 FP16 混合精度训练
    fp16: bool = True

    # 是否使用 BF16 混合精度训练（需要 Ampere 架构以上 GPU）
    bf16: bool = False

    # ========== 日志和保存配置 ==========
    # 日志记录频率（步数）
    logging_steps: int = 10

    # 模型保存频率（步数）
    save_steps: int = 500

    # 评估频率（步数）
    eval_steps: int = 500

    # 最多保存的检查点数量
    save_total_limit: int = 3

    # ========== 评估配置 ==========
    # 评估策略
    # "steps": 按步数评估
    # "epoch": 每轮结束评估
    # "no": 不评估
    evaluation_strategy: str = "steps"

    # 是否在训练结束时加载最佳模型
    load_best_model_at_end: bool = True

    # ========== 优化器配置 ==========
    # 优化器类型
    # paged_adamw_8bit: QLoRA 推荐，支持显存分页，避免 OOM
    # adamw_torch: 标准 AdamW
    optim: str = "paged_adamw_8bit"

    # ========== 其他配置 ==========
    # 梯度裁剪阈值（防止梯度爆炸）
    max_grad_norm: float = 1.0

    # 随机种子（保证可复现）
    seed: int = 42

    # 日志上报目标
    # 可选: "tensorboard", "wandb", "none"
    report_to: str = "tensorboard"


# ==================== 数据配置类 ====================

@dataclass
class DataConfig:
    """
    数据配置类

    管理数据处理和格式相关的所有配置。

    属性：
        raw_data_path (str): 原始数据文件路径
        processed_data_dir (str): 处理后数据目录
        train_file (str): 训练集文件名
        valid_file (str): 验证集文件名
        test_file (str): 测试集文件名
        train_ratio (float): 训练集比例
        valid_ratio (float): 验证集比例
        test_ratio (float): 测试集比例
        max_input_length (int): 最大输入长度
        max_output_length (int): 最大输出长度
        max_total_length (int): 最大总长度
        data_format (str): 数据格式
        system_prompt (str): 系统提示词

    数据格式说明：
        使用 ChatML 格式，这是 Qwen 模型推荐的对齐格式。
    """
    # ========== 数据路径配置 ==========
    # 原始数据文件路径（CSV 格式）
    raw_data_path: str = str(PROJECT_ROOT / "data" / "raw" / "data.csv")

    # 处理后数据目录（JSONL 格式）
    processed_data_dir: str = str(PROJECT_ROOT / "data" / "processed")

    # ========== 数据文件名 ==========
    # 训练集文件（JSONL 格式，ChatML 风格）
    train_file: str = "train.jsonl"
    # 验证集文件
    valid_file: str = "valid.jsonl"
    # 测试集文件
    test_file: str = "test.jsonl"

    # ========== 数据划分比例 ==========
    # 训练集比例
    train_ratio: float = 0.8
    # 验证集比例
    valid_ratio: float = 0.1
    # 测试集比例
    test_ratio: float = 0.1

    # ========== 数据长度配置 ==========
    # 最大输入长度（用户问题）
    max_input_length: int = 512
    # 最大输出长度（模型回复）
    max_output_length: int = 512
    # 最大总长度（input + output）
    max_total_length: int = 2048

    # ========== 数据格式配置 ==========
    # 数据格式（ChatML 是 Qwen 推荐的格式）
    data_format: str = "chatml"

    # ========== 系统提示词 ==========
    # 这是模型的"人设"，定义了模型的角色和行为方式
    system_prompt: str = (
        "你是一名专业的心理咨询客服，请根据来访者的问题，给出专业、共情的回复。\n\n"
        "回复要求：\n"
        "1. 首先表达对来访者情绪的理解和接纳\n"
        "2. 分析来访者可能面临的问题\n"
        "3. 提供具体、可行的建议\n"
        "4. 以开放式问题或鼓励性话语引导来访者继续表达\n\n"
        "注意：保持温和、专业的语气，避免过于绝对的建议，尊重来访者的感受。"
    )


# ==================== 数据增强配置类 ====================

@dataclass
class AugmentConfig:
    """
    数据增强配置类

    管理 AI 辅助数据增强的相关配置。

    属性：
        enabled (bool): 是否启用数据增强
        augment_ratio (float): 增强比例
        strategies (List[str]): 增强策略列表
        api_key (Optional[str]): DashScope API 密钥
        api_model (str): Qwen 模型名称
        scenario_types (List[str]): 场景类型列表

    增强策略说明：
        - paraphrase: 同义改写用户输入
        - enhance: 优化回复内容
        - scenario: 场景扩展（将对话改编到不同场景）
        - clean: 清理回复格式
    """
    # ========== 增强开关 ==========
    # 是否启用 AI 辅助数据增强
    enabled: bool = False

    # 增强比例（0-1）
    # 0.3 表示对 30% 的数据进行增强
    augment_ratio: float = 0.3

    # ========== 增强策略 ==========
    # 可选: paraphrase, enhance, scenario, clean
    strategies: List[str] = field(default_factory=lambda: ["paraphrase", "enhance"])

    # ========== API 配置 ==========
    # DashScope API 密钥（从环境变量 DASHSCOPE_API_KEY 读取）
    api_key: Optional[str] = None

    # Qwen 模型名称
    # qwen-turbo: 快速，便宜
    # qwen-plus: 平衡
    # qwen-max: 效果最好
    api_model: str = "qwen-plus"

    # ========== 场景配置 ==========
    # 场景扩展时使用的场景类型
    scenario_types: List[str] = field(default_factory=lambda: ["青少年", "职场", "家庭", "学业"])


# ==================== 推理配置类 ====================

@dataclass
class InferenceConfig:
    """
    推理配置类

    管理模型推理（生成）过程中的参数。

    属性：
        max_new_tokens (int): 最大新生成 token 数
        temperature (float): 温度参数
        top_p (float): Nucleus Sampling 参数
        top_k (int): Top-K Sampling 参数
        repetition_penalty (float): 重复惩罚系数
        do_sample (bool): 是否采样

    生成参数说明：
        - temperature: 控制随机性，值越大输出越随机
        - top_p: 保留累积概率达到 p 的最小 token 集合
        - top_k: 只从概率最高的 k 个 token 中采样
        - repetition_penalty: 惩罚重复内容
    """
    # ========== 生成参数 ==========
    # 最大新生成 token 数
    max_new_tokens: int = 512

    # 温度参数（0-2）
    # 值越大输出越随机，值越小输出越确定
    # 心理咨询场景建议 0.7，保持一定的随机性同时保证质量
    temperature: float = 0.7

    # Nucleus Sampling 参数（0-1）
    # 保留累积概率达到此值的 token
    top_p: float = 0.9

    # Top-K Sampling 参数
    # 只从概率最高的 k 个 token 中采样
    top_k: int = 50

    # 重复惩罚系数（>1 惩罚重复，=1 不惩罚）
    repetition_penalty: float = 1.1

    # 是否采样
    # True: 使用采样生成（更随机）
    # False: 使用贪婪解码（更确定）
    do_sample: bool = True


# ==================== 总配置类 ====================

@dataclass
class Config:
    """
    总配置类

    整合所有子配置类，作为项目的统一配置入口。

    属性：
        model (ModelConfig): 模型配置
        qlora (QLoRAConfig): QLoRA 量化配置
        lora (LoRAConfig): LoRA 适配器配置
        training (TrainingConfig): 训练配置
        data (DataConfig): 数据配置
        augment (AugmentConfig): 数据增强配置
        inference (InferenceConfig): 推理配置

    使用示例：
        >>> from config import config
        >>> print(config.model.base_model_name)
        Qwen/Qwen2.5-7B-Instruct
        >>> print(config.training.learning_rate)
        0.0002
    """
    # 各子配置
    model: ModelConfig = field(default_factory=ModelConfig)
    qlora: QLoRAConfig = field(default_factory=QLoRAConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


# ==================== 全局配置实例 ====================

# 默认配置实例（单例模式）
config = Config()


# ==================== 辅助函数 ====================

def get_config() -> Config:
    """
    获取配置实例

    Returns:
        Config: 全局配置实例

    使用示例：
        >>> cfg = get_config()
        >>> print(cfg.model.base_model_name)
    """
    return config


def print_config():
    """
    打印配置信息

    以格式化的方式打印所有配置参数，便于调试和查看。

    使用示例：
        >>> from config import print_config
        >>> print_config()
    """
    from dataclasses import asdict
    import json

    print("=" * 60)
    print("MentalChat 项目配置")
    print("=" * 60)

    # 构建配置字典
    cfg_dict = {
        "model": asdict(config.model),
        "qlora": asdict(config.qlora),
        "lora": asdict(config.lora),
        "training": asdict(config.training),
        "data": asdict(config.data),
        "augment": asdict(config.augment),
        "inference": asdict(config.inference),
    }

    # 分组打印
    for section, params in cfg_dict.items():
        print(f"\n[{section.upper()}]")
        for key, value in params.items():
            print(f"  {key}: {value}")


# ==================== 测试代码 ====================

if __name__ == "__main__":
    """
    配置模块测试入口

    运行方式：
        python config/config.py

    输出所有配置信息，便于检查配置是否正确。
    """
    print_config()
