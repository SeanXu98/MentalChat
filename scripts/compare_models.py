#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型对比评估脚本
================

本脚本用于对比微调前后的模型效果，帮助评估微调的价值。

主要功能：
    - 同时评估基座模型和微调后模型
    - 生成直观的对比报告
    - 展示具体的回复样例对比
    - 计算多维度评估指标

评估维度：
    - ROUGE 指标：文本相似度
    - 回复长度：平均字数
    - 专业术语：心理咨询关键词覆盖率
    - 回复质量：人工评估参考

运行方法：
    # 基本用法
    python scripts/compare_models.py --lora-path output/checkpoints/final

    # 限制评估样本数（快速测试）
    python scripts/compare_models.py --lora-path output/checkpoints/final --max-samples 50

    # 保存详细对比结果
    python scripts/compare_models.py --lora-path output/checkpoints/final --save-details

作者：MentalChat 项目组
"""

# ==================== 标准库导入 ====================
import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ==================== 第三方库导入 ====================
from tqdm import tqdm

# ==================== 项目内部导入 ====================
# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import config


# ==================== 心理咨询关键词定义 ====================

# 专业术语关键词（微调后模型应该更常使用）
PROFESSIONAL_KEYWORDS = [
    # 情感认同
    "理解", "共情", "感受", "体会到", "感同身受",
    # 引导性提问
    "能说说", "愿意分享", "你觉得", "你认为", "有没有想过",
    # 情绪识别
    "焦虑", "抑郁", "压力", "情绪", "心情", "困扰",
    # 专业建议
    "建议", "尝试", "方法", "技巧", "练习", "放松",
    # 支持性语言
    "你不是一个人", "很正常", "没关系", "勇敢", "第一步",
    # 总结反馈
    "总结一下", "听起来", "你的意思是", "让我理解一下"
]

# 不推荐的表达（基座模型可能更多使用）
AVOID_KEYWORDS = [
    "作为一个AI", "作为人工智能", "我没有感情", "我无法理解",
    "建议您咨询", "请联系专业", "我不能", "我不建议",
    "很抱歉", "对不起", "我不太清楚"
]


# ==================== 数据加载函数 ====================

def load_test_data(test_path: str) -> List[Dict]:
    """
    加载测试数据

    从 JSONL 格式的测试文件中加载数据。

    Args:
        test_path: 测试数据文件路径

    Returns:
        测试数据列表
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

    Args:
        lora_path: LoRA 权重路径（可选，不指定则加载基座模型）
        use_4bit: 是否使用 4-bit 量化

    Returns:
        Tuple[model, tokenizer]: 加载的模型和分词器
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    # 确定基座模型路径
    model_path = config.model.base_model_path
    if not os.path.exists(model_path):
        model_path = config.model.base_model_name

    # 配置量化
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )

    # 加载基座模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # 如果指定了 LoRA 权重，则加载
    if lora_path and os.path.exists(lora_path):
        print(f"加载 LoRA 权重: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model_type = "微调后模型"
    else:
        model_type = "基座模型"

    model.eval()
    return model, tokenizer, model_type


# ==================== 推理函数 ====================

def generate_response(
    model,
    tokenizer,
    messages: List[Dict],
    max_new_tokens: int = 512
) -> str:
    """
    生成模型回复

    Args:
        model: 模型实例
        tokenizer: 分词器实例
        messages: 对话消息列表
        max_new_tokens: 最大生成 token 数

    Returns:
        生成的回复文本
    """
    import torch

    # 应用 ChatML 模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 编码输入
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # 生成回复
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 解码输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取助手回复部分
    if "<|im_start|>assistant" in generated_text:
        response = generated_text.split("<|im_start|>assistant")[-1].strip()
    elif "assistant" in generated_text:
        response = generated_text.split("assistant")[-1].strip()
    else:
        response = generated_text[len(text):].strip()

    # 清理特殊标记
    response = response.replace("<|im_end|>", "").strip()

    return response


# ==================== 指标计算函数 ====================

def calculate_rouge(predictions: List[str], references: List[str]) -> Dict:
    """
    计算 ROUGE 指标

    Args:
        predictions: 预测回复列表
        references: 参考回复列表

    Returns:
        ROUGE 指标字典
    """
    try:
        from rouge_score import rouge_scorer
        import numpy as np

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []

        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

        return {
            "rouge1": float(np.mean(rouge1_scores)),
            "rouge2": float(np.mean(rouge2_scores)),
            "rougeL": float(np.mean(rougeL_scores)),
        }
    except ImportError:
        print("警告: 未安装 rouge_score 库，跳过 ROUGE 计算")
        return {"rouge1": 0, "rouge2": 0, "rougeL": 0}


def calculate_keyword_coverage(responses: List[str], keywords: List[str]) -> Dict:
    """
    计算关键词覆盖率

    Args:
        responses: 回复列表
        keywords: 关键词列表

    Returns:
        覆盖率统计字典
    """
    total_responses = len(responses)
    coverage_count = 0
    total_matches = 0

    for response in responses:
        matches = sum(1 for kw in keywords if kw in response)
        if matches > 0:
            coverage_count += 1
            total_matches += matches

    return {
        "coverage_rate": coverage_count / total_responses if total_responses > 0 else 0,
        "avg_matches": total_matches / total_responses if total_responses > 0 else 0,
        "total_matches": total_matches
    }


def calculate_response_stats(responses: List[str]) -> Dict:
    """
    计算回复统计信息

    Args:
        responses: 回复列表

    Returns:
        统计信息字典
    """
    lengths = [len(r) for r in responses]

    import numpy as np

    return {
        "avg_length": float(np.mean(lengths)),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
        "std_length": float(np.std(lengths))
    }


# ==================== 对比评估函数 ====================

def compare_models(
    lora_path: str,
    test_path: str = None,
    max_samples: int = None,
    save_details: bool = False,
    output_path: str = None,
    use_4bit: bool = True
):
    """
    对比基座模型和微调后模型

    Args:
        lora_path: LoRA 权重路径
        test_path: 测试数据路径
        max_samples: 最大评估样本数
        save_details: 是否保存详细对比结果
        output_path: 结果输出路径
        use_4bit: 是否使用 4-bit 量化
    """
    # 设置默认路径
    test_path = test_path or os.path.join(config.data.processed_data_dir, config.data.test_file)
    output_path = output_path or os.path.join(config.training.output_dir, "model_comparison.json")

    print("=" * 70)
    print("模型对比评估")
    print("=" * 70)
    print(f"测试数据: {test_path}")
    print(f"LoRA 权重: {lora_path}")
    print(f"最大样本数: {max_samples or '全部'}")
    print()

    # 加载测试数据
    print("加载测试数据...")
    test_data = load_test_data(test_path)
    if max_samples:
        test_data = test_data[:max_samples]
    print(f"共 {len(test_data)} 条测试样本")
    print()

    # 存储结果
    base_predictions = []
    finetuned_predictions = []
    references = []
    inputs = []

    # ========== 评估基座模型 ==========
    print("=" * 70)
    print("【1/2】评估基座模型（微调前）")
    print("=" * 70)

    model, tokenizer, model_type = load_model(lora_path=None, use_4bit=use_4bit)
    print(f"模型类型: {model_type}")
    print()

    for i, sample in enumerate(tqdm(test_data, desc="生成基座模型回复")):
        messages = sample["messages"]

        # 获取用户输入
        user_messages = [m for m in messages if m["role"] == "user"]
        user_input = user_messages[-1]["content"] if user_messages else ""
        inputs.append(user_input)

        # 获取参考回复
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        reference = assistant_messages[-1]["content"] if assistant_messages else ""
        references.append(reference)

        # 生成基座模型回复
        input_messages = [m for m in messages if m["role"] != "assistant"]
        response = generate_response(model, tokenizer, input_messages)
        base_predictions.append(response)

    # 释放基座模型
    del model
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except:
        pass

    print()

    # ========== 评估微调后模型 ==========
    print("=" * 70)
    print("【2/2】评估微调后模型")
    print("=" * 70)

    model, tokenizer, model_type = load_model(lora_path=lora_path, use_4bit=use_4bit)
    print(f"模型类型: {model_type}")
    print()

    for i, sample in enumerate(tqdm(test_data, desc="生成微调模型回复")):
        messages = sample["messages"]
        input_messages = [m for m in messages if m["role"] != "assistant"]
        response = generate_response(model, tokenizer, input_messages)
        finetuned_predictions.append(response)

    print()

    # ========== 计算评估指标 ==========
    print("=" * 70)
    print("计算评估指标...")
    print("=" * 70)

    # ROUGE 指标
    base_rouge = calculate_rouge(base_predictions, references)
    finetuned_rouge = calculate_rouge(finetuned_predictions, references)

    # 专业关键词覆盖
    base_professional = calculate_keyword_coverage(base_predictions, PROFESSIONAL_KEYWORDS)
    finetuned_professional = calculate_keyword_coverage(finetuned_predictions, PROFESSIONAL_KEYWORDS)

    # 不推荐关键词覆盖
    base_avoid = calculate_keyword_coverage(base_predictions, AVOID_KEYWORDS)
    finetuned_avoid = calculate_keyword_coverage(finetuned_predictions, AVOID_KEYWORDS)

    # 回复统计
    base_stats = calculate_response_stats(base_predictions)
    finetuned_stats = calculate_response_stats(finetuned_predictions)

    # ========== 打印对比报告 ==========
    print()
    print("=" * 70)
    print("对比评估报告")
    print("=" * 70)
    print()

    # ROUGE 对比
    print("【1】ROUGE 指标（与参考回复的相似度）")
    print("-" * 50)
    print(f"{'指标':<15} {'基座模型':<15} {'微调后模型':<15} {'提升':<15}")
    print("-" * 50)
    for metric in ["rouge1", "rouge2", "rougeL"]:
        base_val = base_rouge[metric]
        ft_val = finetuned_rouge[metric]
        change = (ft_val - base_val) / base_val * 100 if base_val > 0 else 0
        change_str = f"+{change:.1f}%" if change > 0 else f"{change:.1f}%"
        print(f"{metric.upper():<15} {base_val:.4f}          {ft_val:.4f}          {change_str:<15}")
    print()

    # 专业关键词对比
    print("【2】专业术语覆盖率（越高越好）")
    print("-" * 50)
    print(f"{'指标':<20} {'基座模型':<15} {'微调后模型':<15} {'提升':<15}")
    print("-" * 50)

    base_cov = base_professional["coverage_rate"] * 100
    ft_cov = finetuned_professional["coverage_rate"] * 100
    change = ft_cov - base_cov
    change_str = f"+{change:.1f}%" if change > 0 else f"{change:.1f}%"
    print(f"{'覆盖率':<20} {base_cov:.1f}%           {ft_cov:.1f}%           {change_str:<15}")

    base_avg = base_professional["avg_matches"]
    ft_avg = finetuned_professional["avg_matches"]
    change = (ft_avg - base_avg) / base_avg * 100 if base_avg > 0 else 0
    change_str = f"+{change:.1f}%" if change > 0 else f"{change:.1f}%"
    print(f"{'平均匹配数':<20} {base_avg:.2f}            {ft_avg:.2f}            {change_str:<15}")
    print()

    # 不推荐关键词对比
    print("【3】不推荐表达覆盖率（越低越好）")
    print("-" * 50)
    base_avoid_rate = base_avoid["coverage_rate"] * 100
    ft_avoid_rate = finetuned_avoid["coverage_rate"] * 100
    change = ft_avoid_rate - base_avoid_rate
    change_str = f"{change:.1f}%" if change < 0 else f"+{change:.1f}%"
    print(f"{'覆盖率':<20} {base_avoid_rate:.1f}%           {ft_avoid_rate:.1f}%           {change_str:<15}")
    print()

    # 回复长度对比
    print("【4】回复长度统计")
    print("-" * 50)
    print(f"{'指标':<15} {'基座模型':<15} {'微调后模型':<15}")
    print("-" * 50)
    print(f"{'平均长度':<15} {base_stats['avg_length']:.1f}           {finetuned_stats['avg_length']:.1f}")
    print(f"{'最小长度':<15} {base_stats['min_length']:<15} {finetuned_stats['min_length']:<15}")
    print(f"{'最大长度':<15} {base_stats['max_length']:<15} {finetuned_stats['max_length']:<15}")
    print()

    # ========== 显示对比样例 ==========
    print("=" * 70)
    print("回复对比样例")
    print("=" * 70)

    # 选择几个代表性样例
    num_examples = min(3, len(test_data))
    for i in range(num_examples):
        print()
        print(f"【样例 {i+1}】")
        print("-" * 50)
        print(f"用户输入: {inputs[i][:100]}..." if len(inputs[i]) > 100 else f"用户输入: {inputs[i]}")
        print()
        print(f"参考回复: {references[i][:150]}..." if len(references[i]) > 150 else f"参考回复: {references[i]}")
        print()
        print(f"基座模型: {base_predictions[i][:150]}..." if len(base_predictions[i]) > 150 else f"基座模型: {base_predictions[i]}")
        print()
        print(f"微调模型: {finetuned_predictions[i][:150]}..." if len(finetuned_predictions[i]) > 150 else f"微调模型: {finetuned_predictions[i]}")
        print("-" * 50)

    # ========== 保存结果 ==========
    if save_details:
        results = {
            "config": {
                "lora_path": lora_path,
                "test_path": test_path,
                "num_samples": len(test_data)
            },
            "metrics": {
                "base_model": {
                    "rouge": base_rouge,
                    "professional_keywords": base_professional,
                    "avoid_keywords": base_avoid,
                    "response_stats": base_stats
                },
                "finetuned_model": {
                    "rouge": finetuned_rouge,
                    "professional_keywords": finetuned_professional,
                    "avoid_keywords": finetuned_avoid,
                    "response_stats": finetuned_stats
                }
            },
            "samples": []
        }

        # 保存详细样例
        for i in range(len(test_data)):
            results["samples"].append({
                "input": inputs[i],
                "reference": references[i],
                "base_prediction": base_predictions[i],
                "finetuned_prediction": finetuned_predictions[i]
            })

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print()
        print(f"✓ 详细结果已保存到: {output_path}")

    print()
    print("=" * 70)
    print("评估完成")
    print("=" * 70)


# ==================== 主函数 ====================

def main():
    """
    主函数 - 解析命令行参数并执行对比评估
    """
    parser = argparse.ArgumentParser(
        description="MentalChat 模型对比评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    # 基本用法
    python scripts/compare_models.py --lora-path output/checkpoints/final

    # 快速测试（限制样本数）
    python scripts/compare_models.py --lora-path output/checkpoints/final --max-samples 50

    # 保存详细对比结果
    python scripts/compare_models.py --lora-path output/checkpoints/final --save-details
        """
    )

    parser.add_argument(
        "--lora-path",
        type=str,
        required=True,
        help="LoRA 权重路径（必需）"
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=None,
        help="测试数据路径（默认使用配置中的测试集）"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大评估样本数（用于快速测试）"
    )
    parser.add_argument(
        "--save-details",
        action="store_true",
        help="保存详细的对比结果到 JSON 文件"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="结果输出路径"
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="禁用 4-bit 量化（需要更多显存）"
    )

    args = parser.parse_args()

    # 检查 LoRA 路径
    if not os.path.exists(args.lora_path):
        print(f"错误: LoRA 权重路径不存在: {args.lora_path}")
        return 1

    # 执行对比评估
    compare_models(
        lora_path=args.lora_path,
        test_path=args.test_path,
        max_samples=args.max_samples,
        save_details=args.save_details,
        output_path=args.output_path,
        use_4bit=not args.no_4bit
    )

    return 0


# ==================== 程序入口 ====================

if __name__ == "__main__":
    sys.exit(main())
