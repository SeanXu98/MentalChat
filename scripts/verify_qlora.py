#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QLoRA 配置验证脚本
验证 4-bit 量化配置是否正常工作
"""

import os
import sys
import gc
import argparse
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def clear_memory():
    """清理显存"""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except:
        pass


def check_gpu_memory():
    """检查GPU显存使用情况"""
    import torch
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return {
            "allocated": allocated,
            "reserved": reserved,
            "total": total,
            "free": total - reserved
        }
    return None


def print_memory_status(stage: str):
    """打印显存状态"""
    mem = check_gpu_memory()
    if mem:
        print(f"  [{stage}] 显存: 已分配 {mem['allocated']:.2f}GB / "
              f"已保留 {mem['reserved']:.2f}GB / 总计 {mem['total']:.2f}GB")


def verify_qlora_config(model_path: str):
    """
    验证 QLoRA 配置

    Args:
        model_path: 模型路径
    """
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig
    )
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

    print("="*60)
    print("QLoRA 配置验证")
    print("="*60)
    print(f"模型路径: {model_path}")
    print()

    # 1. 配置 4-bit 量化
    print("1. 配置 4-bit 量化参数...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print("  ✓ 量化配置完成")
    print("    - 量化类型: NF4")
    print("    - 计算精度: float16")
    print("    - 双量化: 启用")

    # 2. 加载量化模型
    print("\n2. 加载 4-bit 量化模型...")
    clear_memory()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # 设置 pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("  设置 pad_token = eos_token")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        print("  ✓ 模型加载成功")
        print_memory_status("模型加载后")

    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        return False

    # 3. 准备模型进行 k-bit 训练
    print("\n3. 准备模型进行 k-bit 训练...")
    try:
        model = prepare_model_for_kbit_training(model)
        print("  ✓ 模型准备完成")
    except Exception as e:
        print(f"  ✗ 模型准备失败: {e}")
        return False

    # 4. 配置 LoRA
    print("\n4. 配置 LoRA 参数...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    print("  ✓ LoRA 配置完成")
    print(f"    - 秩 (r): {lora_config.r}")
    print(f"    - alpha: {lora_config.lora_alpha}")
    print(f"    - 目标模块: {lora_config.target_modules}")
    print(f"    - dropout: {lora_config.lora_dropout}")

    # 5. 应用 LoRA
    print("\n5. 应用 LoRA 到模型...")
    try:
        model = get_peft_model(model, lora_config)
        print("  ✓ LoRA 应用成功")
    except Exception as e:
        print(f"  ✗ LoRA 应用失败: {e}")
        return False

    # 6. 打印可训练参数
    print("\n6. 可训练参数统计...")
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    trainable_ratio = 100 * trainable_params / all_params
    print(f"  可训练参数: {trainable_params:,} ({trainable_ratio:.4f}%)")
    print(f"  总参数: {all_params:,}")
    print_memory_status("LoRA应用后")

    # 7. 测试推理
    print("\n7. 测试推理...")
    try:
        test_prompt = "你是一名专业的心理咨询客服。来访者说：我最近心情很低落。请给出专业回复。"

        # 使用 Qwen2 的聊天模板
        messages = [
            {"role": "system", "content": "你是一名专业的心理咨询客服，请根据来访者的问题，给出专业、共情的回复。"},
            {"role": "user", "content": "我最近心情很低落。"}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取模型回复部分
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()

        print(f"  测试输入: {messages[-1]['content']}")
        print(f"  模型回复: {response[:200]}...")
        print("  ✓ 推理测试成功")

    except Exception as e:
        print(f"  ✗ 推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 8. 总结
    print("\n" + "="*60)
    print("✓ QLoRA 配置验证通过！")
    print("="*60)
    print("\n配置摘要:")
    print(f"  - 基座模型: Qwen2.5-7B-Instruct")
    print(f"  - 量化方案: 4-bit NF4")
    print(f"  - LoRA 秩: 16")
    print(f"  - 可训练参数比例: {trainable_ratio:.4f}%")

    mem = check_gpu_memory()
    if mem:
        print(f"  - 显存占用: {mem['allocated']:.2f}GB / {mem['total']:.2f}GB")
        if mem['allocated'] < 15:
            print("  ✓ 显存占用正常，适合单卡 RTX 4090 训练")
        else:
            print("  ⚠ 显存占用较高，可能需要调整 batch_size")

    print("\n下一步: 运行数据处理脚本准备训练数据")

    # 清理
    clear_memory()

    return True


def main():
    parser = argparse.ArgumentParser(description="验证 QLoRA 配置")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/root/autodl-tmp/MentalChat/models/base/Qwen_Qwen2.5-7B-Instruct",
        help="模型路径（默认使用AutoDL数据盘路径）"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="如果本地模型不存在，从 HuggingFace 下载的模型名称"
    )

    args = parser.parse_args()

    # 检查模型路径是否存在
    if not os.path.exists(args.model_path):
        print(f"本地模型路径不存在: {args.model_path}")
        print(f"尝试从 HuggingFace 下载: {args.model_name}")
        args.model_path = args.model_name

    success = verify_qlora_config(args.model_path)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
