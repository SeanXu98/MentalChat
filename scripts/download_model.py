#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型下载脚本
============

本脚本用于下载 Qwen2.5-7B-Instruct 基座模型到本地。

主要功能：
    - 从 Hugging Face 下载模型权重和分词器
    - 支持使用国内镜像加速下载
    - 下载后自动验证模型可用性

运行方法：
    # 基本用法
    python scripts/download_model.py

    # 使用国内镜像（推荐）
    python scripts/download_model.py --mirror

    # 指定保存目录
    python scripts/download_model.py --save-dir /path/to/save

    # 仅验证已下载的模型
    python scripts/download_model.py --verify-only /path/to/model

注意事项：
    - 模型大小约 14GB，请确保有足够的磁盘空间
    - 国内网络环境建议使用 --mirror 参数
    - AutoDL 用户建议保存到 /root/autodl-tmp/ 目录

作者：MentalChat 项目组
"""

# ==================== 标准库导入 ====================
import os
import sys
import argparse
from pathlib import Path

# ==================== 项目内部导入 ====================
# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ==================== 模型下载函数 ====================

def download_model(model_name: str, save_dir: str, use_mirror: bool = False):
    """
    下载模型和分词器

    从 Hugging Face 下载指定模型并保存到本地。

    Args:
        model_name: Hugging Face 模型名称（如 Qwen/Qwen2.5-7B-Instruct）
        save_dir: 本地保存目录
        use_mirror: 是否使用国内镜像（推荐国内用户开启）

    Returns:
        bool: 下载是否成功

    下载流程：
        1. 配置镜像（如果启用）
        2. 下载分词器（Tokenizer）
        3. 下载模型权重
        4. 保存到本地目录
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    print(f"开始下载模型: {model_name}")
    print(f"保存目录: {save_dir}")

    # ========== 配置镜像（国内网络环境）==========
    if use_mirror:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print("使用 Hugging Face 镜像: https://hf-mirror.com")

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # ========== 下载分词器 ==========
    print("\n" + "=" * 60)
    print("下载 Tokenizer...")
    print("=" * 60)

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

    # ========== 下载模型权重 ==========
    print("\n" + "=" * 60)
    print("下载模型权重（约 14GB，请耐心等待）...")
    print("=" * 60)

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

    # ========== 保存到本地目录 ==========
    local_save_path = os.path.join(save_dir, model_name.replace("/", "_"))
    print(f"\n保存模型到本地: {local_save_path}")

    tokenizer.save_pretrained(local_save_path)
    model.save_pretrained(local_save_path)

    print("\n" + "=" * 60)
    print("✓ 模型下载并保存成功！")
    print("=" * 60)
    print(f"模型路径: {local_save_path}")

    return True


# ==================== 模型验证函数 ====================

def verify_model(model_path: str):
    """
    验证模型是否可以正常加载

    验证流程：
        1. 加载分词器
        2. 加载模型
        3. 进行简单的推理测试

    Args:
        model_path: 本地模型路径

    Returns:
        bool: 验证是否通过
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    print(f"\n验证模型: {model_path}")

    try:
        # ========== 加载分词器 ==========
        print("  加载 Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("  ✓ Tokenizer 加载成功")

        # ========== 加载模型（使用 CPU 以节省显存）==========
        print("  加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True
        )
        print("  ✓ 模型加载成功")

        # ========== 简单推理测试 ==========
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


# ==================== 主函数 ====================

def main():
    """
    主函数 - 解析命令行参数并执行下载或验证

    支持的命令行参数：
        --model: 模型名称（默认 Qwen/Qwen2.5-7B-Instruct）
        --save-dir: 保存目录
        --mirror: 使用国内镜像
        --verify-only: 仅验证模式
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="下载微调基座模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    # 基本用法
    python scripts/download_model.py

    # 使用国内镜像（推荐）
    python scripts/download_model.py --mirror

    # 指定保存目录
    python scripts/download_model.py --save-dir /path/to/save

    # 仅验证已下载的模型
    python scripts/download_model.py --verify-only /path/to/model
        """
    )

    # 定义命令行参数
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
        help="模型保存目录（默认使用 AutoDL 数据盘，避免占用系统盘空间）"
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="使用 Hugging Face 镜像（国内网络环境推荐）"
    )
    parser.add_argument(
        "--verify-only",
        type=str,
        default=None,
        help="仅验证已下载的模型（指定模型路径）"
    )

    # 解析参数
    args = parser.parse_args()

    # 打印标题
    print("=" * 60)
    print("心理咨询客服对话模型微调 - 模型下载")
    print("=" * 60)

    # 执行下载或验证
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

        # 验证下载的模型
        if success:
            local_path = os.path.join(args.save_dir, args.model.replace("/", "_"))
            success = verify_model(local_path)

    return 0 if success else 1


# ==================== 程序入口 ====================

if __name__ == "__main__":
    sys.exit(main())
