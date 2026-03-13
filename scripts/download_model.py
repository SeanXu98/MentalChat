#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型下载脚本
下载 Qwen2.5-7B-Instruct 基座模型到本地
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def download_model(model_name: str, save_dir: str, use_mirror: bool = False):
    """
    下载模型

    Args:
        model_name: 模型名称
        save_dir: 保存目录
        use_mirror: 是否使用镜像
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    print(f"开始下载模型: {model_name}")
    print(f"保存目录: {save_dir}")

    # 设置镜像（国内网络环境）
    if use_mirror:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print("使用 Hugging Face 镜像: https://hf-mirror.com")

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "="*60)
    print("下载 Tokenizer...")
    print("="*60)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=save_dir
        )
        print(f"✓ Tokenizer 下载完成")
        print(f"  词表大小: {len(tokenizer)}")
        print(f"  特殊 tokens: pad_token={tokenizer.pad_token}, eos_token={tokenizer.eos_token}")
    except Exception as e:
        print(f"✗ Tokenizer 下载失败: {e}")
        return False

    print("\n" + "="*60)
    print("下载模型权重（约 14GB，请耐心等待）...")
    print("="*60)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=save_dir
        )
        print(f"✓ 模型下载完成")
        print(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
        print(f"  设备: {next(model.parameters()).device}")
    except Exception as e:
        print(f"✗ 模型下载失败: {e}")
        return False

    # 保存到指定目录
    local_save_path = os.path.join(save_dir, model_name.replace("/", "_"))
    print(f"\n保存模型到本地: {local_save_path}")

    tokenizer.save_pretrained(local_save_path)
    model.save_pretrained(local_save_path)

    print("\n" + "="*60)
    print("✓ 模型下载并保存成功！")
    print("="*60)
    print(f"模型路径: {local_save_path}")

    return True


def verify_model(model_path: str):
    """
    验证模型是否可以正常加载
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    print(f"\n验证模型: {model_path}")

    try:
        # 加载 tokenizer
        print("  加载 Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("  ✓ Tokenizer 加载成功")

        # 加载模型（使用 CPU 以节省显存）
        print("  加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True
        )
        print("  ✓ 模型加载成功")

        # 简单推理测试
        print("  进行推理测试...")
        test_input = "你好"
        inputs = tokenizer(test_input, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  测试输入: {test_input}")
        print(f"  模型输出: {response}")
        print("  ✓ 推理测试成功")

        return True

    except Exception as e:
        print(f"  ✗ 验证失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="下载微调基座模型")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="模型名称 (默认: Qwen/Qwen2.5-7B-Instruct)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/root/autodl-tmp/MentalChat/models/base",
        help="模型保存目录（默认使用AutoDL数据盘，避免占用系统盘空间）"
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="使用 Hugging Face 镜像（国内网络环境）"
    )
    parser.add_argument(
        "--verify-only",
        type=str,
        default=None,
        help="仅验证已下载的模型（指定模型路径）"
    )

    args = parser.parse_args()

    print("="*60)
    print("心理咨询客服对话模型微调 - 模型下载")
    print("="*60)

    if args.verify_only:
        # 仅验证模式
        success = verify_model(args.verify_only)
    else:
        # 下载模式
        success = download_model(
            model_name=args.model,
            save_dir=args.save_dir,
            use_mirror=args.mirror
        )

        if success:
            # 验证下载的模型
            local_path = os.path.join(args.save_dir, args.model.replace("/", "_"))
            success = verify_model(local_path)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
