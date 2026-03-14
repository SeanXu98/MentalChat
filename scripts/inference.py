#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型推理脚本
============

本模块用于加载微调后的心理咨询模型并进行对话推理。

主要功能：
    - 支持单轮对话和多轮对话
    - 支持流式输出
    - 支持批量推理
    - 支持交互式命令行对话

运行方法：
    # 交互式对话
    python scripts/inference.py --interactive

    # 指定 LoRA 权重
    python scripts/inference.py --lora-path output/checkpoints/final --interactive

    # 单次推理
    python scripts/inference.py --input "我最近感觉很焦虑"

作者：MentalChat 项目组
"""

# ==================== 标准库导入 ====================
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional

# ==================== 第三方库导入 ====================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ==================== 项目内部导入 ====================
# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import config


# ==================== 核心类定义 ====================

class MentalChatInference:
    """
    心理咨询模型推理器

    该类封装了模型加载、对话生成的全部逻辑。

    属性：
        lora_path (str): LoRA 权重路径
        device (str): 运行设备
        model: 加载的语言模型
        tokenizer: 加载的分词器

    使用示例：
        >>> inference = MentalChatInference(lora_path="output/checkpoints/final")
        >>> response = inference.chat("我最近感觉很焦虑")
        >>> print(response)
    """

    def __init__(
        self,
        lora_path: str = None,
        use_4bit: bool = True,
        device: str = "auto"
    ):
        """
        初始化推理器

        Args:
            lora_path: LoRA 权重路径，如果为 None 则只使用基座模型
            use_4bit: 是否使用 4-bit 量化（推荐开启以节省显存）
            device: 设备类型，可选 "auto"、"cuda"、"cpu"
        """
        # 保存配置参数
        self.lora_path = lora_path
        self.device = device

        # 模型相关属性（延迟加载）
        self.model = None
        self.tokenizer = None

        # 执行模型加载
        self._load_model(use_4bit)

    # ==================== 模型加载方法 ====================

    def _load_model(self, use_4bit: bool):
        """
        加载模型和分词器

        加载流程：
        1. 确定模型路径（本地优先，否则使用远程）
        2. 配置量化参数
        3. 加载分词器
        4. 加载基座模型
        5. 如果有 LoRA 权重则加载

        Args:
            use_4bit: 是否使用 4-bit 量化

        注意：
        - 使用 4-bit 量化可以将显存占用减少约 75%
        - NF4 量化类型适合正态分布的权重
        """
        # ========== 确定模型路径 ==========
        model_path = config.model.base_model_path
        if not os.path.exists(model_path):
            model_path = config.model.base_model_name
            print(f"本地模型不存在，使用远程模型: {model_path}")
        else:
            print(f"加载基座模型: {model_path}")

        # ========== 配置量化参数 ==========
        # QLoRA 使用 NF4 量化 + 双量化来最大化压缩效果
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,                    # 启用 4-bit 量化
                bnb_4bit_quant_type="nf4",            # 使用 NF4 量化类型（NormalFloat4）
                bnb_4bit_compute_dtype=torch.float16, # 计算时使用 float16 精度
                bnb_4bit_use_double_quant=True,       # 启用双量化进一步压缩
            )
        else:
            bnb_config = None

        # ========== 加载分词器 ==========
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,  # 信任远程代码（Qwen 模型需要）
            use_fast=False           # 使用慢速分词器以确保兼容性
        )

        # 设置 pad_token（Qwen 模型默认没有 pad_token）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("  设置 pad_token = eos_token")

        # ========== 加载基座模型 ==========
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,  # 量化配置
            device_map=self.device,          # 自动分配设备
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        # ========== 加载 LoRA 权重（如果有）==========
        if self.lora_path and os.path.exists(self.lora_path):
            print(f"加载 LoRA 权重: {self.lora_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.lora_path,
                device_map=self.device
            )
            print("✓ 模型加载完成（已加载 LoRA 权重）")
        else:
            print("✓ 基座模型加载完成（未加载 LoRA 权重）")

    # ==================== 对话方法 ====================

    def chat(self, message: str, history: List[Dict] = None) -> str:
        """
        执行单轮对话

        根据用户输入生成心理咨询师的回复，支持多轮对话上下文。

        Args:
            message: 用户输入的文本
            history: 历史对话记录，格式为 [{"content": "用户消息", "response": "助手回复"}, ...]

        Returns:
            模型生成的回复文本

        处理流程：
        1. 构建包含系统提示词和历史对话的消息列表
        2. 应用聊天模板格式化输入
        3. 进行模型推理
        4. 解码并清理输出
        """
        # ========== 构建消息列表 ==========
        # 系统提示词定义了模型的角色和行为
        messages = [{"role": "system", "content": config.data.system_prompt}]

        # 添加历史对话（如果有多轮对话）
        if history:
            for turn in history:
                messages.append({"role": "user", "content": turn["content"]})
                messages.append({"role": "assistant", "content": turn["response"]})

        # 添加当前用户输入
        messages.append({"role": "user", "content": message})

        # ========== 应用聊天模板 ==========
        # 使用 Qwen2 的聊天模板格式化输入
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,            # 返回文本而非 token IDs
            add_generation_prompt=True # 添加助手回复的起始标记
        )

        # ========== Tokenize ==========
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        # ========== 模型推理 ==========
        with torch.no_grad():  # 禁用梯度计算以节省显存
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.inference.max_new_tokens,      # 最大生成 token 数
                temperature=config.inference.temperature,             # 温度参数（控制随机性）
                top_p=config.inference.top_p,                         # nucleus sampling
                top_k=config.inference.top_k,                         # top-k sampling
                repetition_penalty=config.inference.repetition_penalty, # 重复惩罚
                do_sample=config.inference.do_sample,                 # 是否采样
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # ========== 解码输出 ==========
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取助手回复部分
        # Qwen2 使用 <|im_start|>assistant 标记助手回复的开始
        if "<|im_start|>assistant" in full_response:
            response = full_response.split("<|im_start|>assistant")[-1].strip()
        elif "<|im_end|>" in full_response:
            # 尝试通过结束标记提取
            response = full_response.split("<|im_end|>")[-2].strip()
        else:
            # 回退：返回完整响应
            response = full_response

        return response

    def chat_stream(self, message: str, history: List[Dict] = None):
        """
        流式对话

        以流式方式生成回复，逐个 token 输出，适用于需要实时显示的场景。

        Args:
            message: 用户输入的文本
            history: 历史对话记录

        Yields:
            生成的 token（字符串）

        使用示例：
            >>> for token in inference.chat_stream("你好"):
            ...     print(token, end="", flush=True)
        """
        from transformers import TextIteratorStreamer
        import threading

        # ========== 构建消息列表 ==========
        messages = [{"role": "system", "content": config.data.system_prompt}]
        if history:
            for turn in history:
                messages.append({"role": "user", "content": turn["content"]})
                messages.append({"role": "assistant", "content": turn["response"]})
        messages.append({"role": "user", "content": message})

        # ========== 应用聊天模板 ==========
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        # ========== 创建流式输出器 ==========
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,         # 跳过输入部分
            skip_special_tokens=True  # 跳过特殊 token
        )

        # ========== 配置生成参数 ==========
        generation_kwargs = {
            **inputs,
            "max_new_tokens": config.inference.max_new_tokens,
            "temperature": config.inference.temperature,
            "top_p": config.inference.top_p,
            "top_k": config.inference.top_k,
            "repetition_penalty": config.inference.repetition_penalty,
            "do_sample": config.inference.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "streamer": streamer,  # 绑定流式输出器
        }

        # ========== 在单独线程中运行生成 ==========
        # 流式输出需要在单独的线程中运行模型生成
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # 逐个 yield 生成的 token
        for new_token in streamer:
            if new_token:
                yield new_token

        # 等待生成线程结束
        thread.join()

    def batch_inference(self, inputs: List[str], batch_size: int = 8) -> List[str]:
        """
        批量推理

        对多个输入进行批量推理，适用于需要处理大量请求的场景。

        Args:
            inputs: 输入消息列表
            batch_size: 批次大小（默认 8）

        Returns:
            回复列表，与输入列表一一对应

        注意：
        - 批量推理目前是逐个处理，未来可以优化为真正的批量并行处理
        """
        responses = []

        # 按批次处理输入
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            for msg in batch:
                responses.append(self.chat(msg))

        return responses


# ==================== 交互模式函数 ====================

def interactive_mode(lora_path: str = None):
    """
    交互式命令行对话模式

    提供一个简单的命令行界面，允许用户与模型进行多轮对话。

    Args:
        lora_path: LoRA 权重路径（可选）

    支持的命令：
    - quit / exit: 退出程序
    - clear: 清空对话历史
    """
    # 打印欢迎信息
    print("=" * 60)
    print("MentalChat 心理咨询对话系统")
    print("=" * 60)
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'clear' 清空对话历史")
    print()

    # 初始化推理器
    inference = MentalChatInference(lora_path=lora_path)

    # 初始化对话历史
    history = []

    # 主循环
    while True:
        try:
            # 获取用户输入
            user_input = input("用户: ").strip()

            # 跳过空输入
            if not user_input:
                continue

            # 处理退出命令
            if user_input.lower() in ['quit', 'exit']:
                print("\n再见！祝您心理健康！")
                break

            # 处理清空历史命令
            if user_input.lower() == 'clear':
                history = []
                print("对话历史已清空\n")
                continue

            # 生成回复
            response = inference.chat(user_input, history)
            print(f"\n咨询师: {response}\n")

            # 更新对话历史
            history.append({"content": user_input, "response": response})

        except KeyboardInterrupt:
            # 处理 Ctrl+C
            print("\n再见！")
            break


# ==================== 主函数 ====================

def main():
    """
    主函数 - 解析命令行参数并执行相应操作

    支持的运行模式：
    - 交互式对话模式 (--interactive)
    - 单次推理模式 (--input)
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="MentalChat 模型推理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    # 交互式对话
    python scripts/inference.py --interactive

    # 指定 LoRA 权重
    python scripts/inference.py --lora-path output/checkpoints/final --interactive

    # 单次推理
    python scripts/inference.py --input "我最近感觉很焦虑"
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
        "--interactive",
        action="store_true",
        help="启动交互式对话模式"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="单次输入文本（用于单次推理）"
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        default=True,
        help="使用 4-bit 量化（默认开启）"
    )

    # 解析参数
    args = parser.parse_args()

    # 根据参数选择运行模式
    if args.interactive:
        # 交互式对话模式
        interactive_mode(args.lora_path)
    elif args.input:
        # 单次推理模式
        inference = MentalChatInference(
            lora_path=args.lora_path,
            use_4bit=args.use_4bit
        )
        response = inference.chat(args.input)
        print(f"回复: {response}")
    else:
        # 无参数时显示帮助信息
        parser.print_help()

    return 0


# ==================== 程序入口 ====================

if __name__ == "__main__":
    sys.exit(main())
