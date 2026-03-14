#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估脚本
============

本脚本用于评估微调后模型在测试集上的性能。

主要功能：
    - 加载基座模型和 LoRA 权重
    - 在测试集上生成预测
    - 计算 ROUGE 评估指标
    - 保存评估结果和示例

评估指标说明：
    - ROUGE-1: 单词级别的重叠度
    - ROUGE-2: 双词级别的重叠度
    - ROUGE-L: 最长公共子序列

运行方法：
    # 评估微调后的模型
    python scripts/evaluate.py --lora-path output/checkpoints/final

    # 评估基座模型（不加载 LoRA）
    python scripts/evaluate.py

    # 限制评估样本数
    python scripts/evaluate.py --max-samples 100

作者：MentalChat 项目组
"""

# ==================== 标准库导入 ====================
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict

# ==================== 第三方库导入 ====================
from tqdm import tqdm

# ==================== 项目内部导入 ====================
# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import config


# ==================== 数据加载函数 ====================

def load_test_data(test_path: str) -> List[Dict]:
    """
    加载测试数据

    从 JSONL 格式的测试文件中加载数据。

    Args:
        test_path: 测试数据文件路径

    Returns:
        测试数据列表，每条数据包含 messages 字段
    """
    data = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


# ==================== 模型加载函数 ====================

def load_model(lora_path: str = None, use_4bit: bool = True):
    """
    加载模型和分词器

    加载基座模型，可选地加载 LoRA 权重。

    Args:
        lora_path: LoRA 权重路径（可选）
        use_4bit: 是否使用 4-bit 量化（推荐开启以节省显存）

    Returns:
        Tuple[model, tokenizer]: 加载的模型和分词器

    加载流程：
        1. 确定基座模型路径
        2. 配置量化参数
        3. 加载分词器和基座模型
        4. 如果有 LoRA 权重则加载
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    # ========== 确定基座模型路径 ==========
    model_path = config.model.base_model_path
    if not os.path.exists(model_path):
        model_path = config.model.base_model_name

    print(f"加载基座模型: {model_path}")

    # ========== 配置 4-bit 量化 ==========
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,                    # 启用 4-bit 量化
            bnb_4bit_quant_type="nf4",            # NF4 量化类型
            bnb_4bit_compute_dtype=torch.float16, # 计算精度
            bnb_4bit_use_double_quant=True,       # 双量化
        )
    else:
        bnb_config = None

    # ========== 加载分词器 ==========
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )

    # ========== 加载基座模型 ==========
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # ========== 加载 LoRA 权重（如果有）==========
    if lora_path and os.path.exists(lora_path):
        print(f"加载 LoRA 权重: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
    else:
        print("未加载 LoRA 权重，使用基座模型")

    # 设置为评估模式
    model.eval()

    return model, tokenizer


# ==================== 回复生成函数 ====================

def generate_response(model, tokenizer, messages: List[Dict], max_new_tokens: int = 512) -> str:
    """
    生成模型回复

    根据输入的消息列表生成模型的回复。

    Args:
        model: 语言模型
        tokenizer: 分词器
        messages: 消息列表，格式为 [{"role": "user", "content": "..."}, ...]
        max_new_tokens: 最大新生成 token 数

    Returns:
        生成的回复文本

    生成流程：
        1. 应用聊天模板格式化输入
        2. Tokenize
        3. 模型推理
        4. 解码并提取助手回复
    """
    import torch

    # ========== 应用聊天模板 ==========
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # ========== Tokenize ==========
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # ========== 模型推理 ==========
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=config.inference.temperature,
            top_p=config.inference.top_p,
            top_k=config.inference.top_k,
            repetition_penalty=config.inference.repetition_penalty,
            do_sample=config.inference.do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # ========== 解码输出 ==========
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ========== 提取助手回复部分 ==========
    # Qwen2 使用 <|im_start|>assistant 标记助手回复
    if "<|im_start|>assistant" in generated_text:
        response = generated_text.split("<|im_start|>assistant")[-1].strip()
    elif "assistant" in generated_text:
        # 回退方案
        response = generated_text.split("assistant")[-1].strip()
    else:
        # 最终回退：取输入之后的部分
        response = generated_text[len(text):].strip()

    # 清理特殊 token
    response = response.replace("<|im_end|>", "").strip()

    return response


# ==================== 指标计算函数 ====================

def calculate_metrics(predictions: List[str], references: List[str]) -> Dict:
    """
    计算评估指标

    使用 ROUGE 指标评估预测结果的质量。

    Args:
        predictions: 模型预测的回复列表
        references: 真实的参考回复列表

    Returns:
        包含各项评估指标的字典

    指标说明：
        - ROUGE-1: 单词（unigram）级别的重叠度
        - ROUGE-2: 双词（bigram）级别的重叠度
        - ROUGE-L: 基于最长公共子序列的重叠度
    """
    from rouge_score import rouge_scorer
    import numpy as np

    # 创建 ROUGE 评分器
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # 存储各项分数
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    # 计算每对预测和参考的分数
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    # 返回平均分数
    return {
        "rouge1": float(np.mean(rouge1_scores)),
        "rouge2": float(np.mean(rouge2_scores)),
        "rougeL": float(np.mean(rougeL_scores)),
        "num_samples": len(predictions)
    }


# ==================== 主评估函数 ====================

def evaluate(
    lora_path: str = None,
    test_path: str = None,
    output_path: str = None,
    max_samples: int = None,
    use_4bit: bool = True,
    verbose: bool = True
):
    """
    执行模型评估

    完整的评估流程，包括：
        1. 加载模型
        2. 加载测试数据
        3. 生成预测
        4. 计算评估指标
        5. 保存结果

    Args:
        lora_path: LoRA 权重路径（可选）
        test_path: 测试数据路径
        output_path: 结果输出路径
        max_samples: 最大评估样本数（用于快速测试）
        use_4bit: 是否使用 4-bit 量化
        verbose: 是否输出详细信息

    Returns:
        评估指标字典
    """
    # ========== 设置默认路径 ==========
    test_path = test_path or os.path.join(config.data.processed_data_dir, config.data.test_file)
    output_path = output_path or os.path.join(config.training.output_dir, "evaluation_results.json")

    # ========== 打印配置信息 ==========
    print("=" * 60)
    print("MentalChat 模型评估")
    print("=" * 60)
    print(f"LoRA 路径: {lora_path or '无 (使用基座模型)'}")
    print(f"测试数据: {test_path}")
    print()

    # ========== 1. 加载模型 ==========
    print("1. 加载模型...")
    model, tokenizer = load_model(lora_path, use_4bit=use_4bit)

    # ========== 2. 加载测试数据 ==========
    print("\n2. 加载测试数据...")
    test_data = load_test_data(test_path)
    if max_samples:
        test_data = test_data[:max_samples]
        print(f"   限制样本数: {max_samples}")
    print(f"   加载 {len(test_data)} 条测试数据")

    # ========== 3. 生成预测 ==========
    print("\n3. 生成预测...")
    predictions = []
    references = []

    for item in tqdm(test_data, desc="评估中"):
        messages = item.get("messages", [])

        # 获取用户输入（排除助手回复）
        input_messages = [m for m in messages if m["role"] != "assistant"]

        # 获取参考回复（最后一个助手消息）
        # 注意：对于多轮对话，这里只取最后一个助手回复
        reference = next((m["content"] for m in reversed(messages) if m["role"] == "assistant"), "")

        # 生成预测
        prediction = generate_response(model, tokenizer, input_messages)

        predictions.append(prediction)
        references.append(reference)

    # ========== 4. 计算指标 ==========
    print("\n4. 计算评估指标...")
    metrics = calculate_metrics(predictions, references)

    # 打印结果
    print(f"\n[评估结果]")
    print(f"  ROUGE-1: {metrics['rouge1']:.4f}")
    print(f"  ROUGE-2: {metrics['rouge2']:.4f}")
    print(f"  ROUGE-L: {metrics['rougeL']:.4f}")
    print(f"  样本数量: {metrics['num_samples']}")

    # ========== 5. 保存结果 ==========
    results = {
        "lora_path": lora_path,
        "test_path": test_path,
        "metrics": metrics,
        "samples": []
    }

    # 保存一些示例（最多 5 个）
    for i in range(min(5, len(test_data))):
        # 获取用户输入
        user_messages = [m for m in test_data[i]["messages"] if m["role"] == "user"]
        user_input = user_messages[-1]["content"] if user_messages else ""

        results["samples"].append({
            "input": user_input,
            "reference": references[i],
            "prediction": predictions[i]
        })

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存结果文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 结果已保存到: {output_path}")

    return metrics


# ==================== 主函数 ====================

def main():
    """
    主函数 - 解析命令行参数并执行评估

    支持的命令行参数：
        --lora-path: LoRA 权重路径
        --test-path: 测试数据路径
        --output-path: 结果输出路径
        --max-samples: 最大评估样本数
        --no-4bit: 禁用 4-bit 量化
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="MentalChat 模型评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    # 评估微调后的模型
    python scripts/evaluate.py --lora-path output/checkpoints/final

    # 评估基座模型
    python scripts/evaluate.py

    # 限制评估样本数（用于快速测试）
    python scripts/evaluate.py --max-samples 100
        """
    )

    # 定义命令行参数
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="LoRA 权重路径（可选，不指定则使用基座模型）"
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=None,
        help="测试数据路径"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="结果输出路径"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大评估样本数（用于快速测试）"
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="禁用 4-bit 量化（不推荐，需要更多显存）"
    )

    # 解析参数
    args = parser.parse_args()

    # 执行评估
    evaluate(
        lora_path=args.lora_path,
        test_path=args.test_path,
        output_path=args.output_path,
        max_samples=args.max_samples,
        use_4bit=not args.no_4bit
    )

    return 0


# ==================== 程序入口 ====================

if __name__ == "__main__":
    sys.exit(main())
