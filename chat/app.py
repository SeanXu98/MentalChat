#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio 对话界面
===============

本模块提供基于 Gradio 的 Web 对话界面，用于与训练后的心理咨询模型进行交互。

主要功能：
    - 加载基座模型和 LoRA 权重
    - 提供友好的 Web 对话界面
    - 支持多轮对话历史记录
    - 支持公开分享（通过 Gradio share）

运行方法：
    # 基本用法
    python chat/app.py

    # 指定 LoRA 权重
    python chat/app.py --lora-path output/checkpoints/final

    # 启用公开分享
    python chat/app.py --share

参数说明：
    --lora-path: LoRA 权重路径（可选）
    --share: 是否创建公开分享链接
    --port: 服务端口号（默认 7860）

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
import gradio as gr

# ==================== 项目内部导入 ====================
# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import config


# ==================== 核心类定义 ====================

class MentalChatInterface:
    """
    心理咨询对话界面类

    该类封装了模型加载、对话生成和 Gradio 界面创建的全部逻辑。

    属性：
        lora_path (str): LoRA 权重路径
        share (bool): 是否公开分享
        model: 加载的语言模型
        tokenizer: 加载的分词器
        system_prompt (str): 系统提示词
        history (List[Dict]): 对话历史记录

    使用示例：
        >>> interface = MentalChatInterface(lora_path="output/checkpoints/final")
        >>> interface.run()
    """

    def __init__(
        self,
        lora_path: str = None,
        share: bool = False
    ):
        """
        初始化对话界面

        Args:
            lora_path: LoRA 权重路径，如果为 None 则只使用基座模型
            share: 是否创建 Gradio 公开分享链接
        """
        # 保存配置
        self.lora_path = lora_path
        self.share = share

        # 模型相关属性（延迟加载）
        self.model = None
        self.tokenizer = None

        # 对话配置
        self.system_prompt = config.data.system_prompt

        # 对话历史
        self.history: List[Dict[str, str]] = []

        # 加载模型
        self._load_model()

    # ==================== 模型加载方法 ====================

    def _load_model(self):
        """
        加载模型和分词器

        加载流程：
        1. 确定模型路径（本地优先，否则使用远程）
        2. 配置 4-bit 量化参数
        3. 加载分词器
        4. 加载基座模型
        5. 如果有 LoRA 权重则加载

        注意：
        - 使用 4-bit 量化可以大幅减少显存占用
        - 模型加载到 GPU 上（device_map="auto"）
        """
        # 确定模型路径
        model_path = config.model.base_model_path
        if not os.path.exists(model_path):
            model_path = config.model.base_model_name
            print(f"本地模型不存在，使用远程模型: {model_path}")
        else:
            print(f"加载基座模型: {model_path}")

        # ========== 配置 4-bit 量化 ==========
        # QLoRA 使用 NF4 量化类型和双量化来减少显存占用
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,                    # 启用 4-bit 量化
            bnb_4bit_quant_type="nf4",            # 使用 NF4 量化类型
            bnb_4bit_compute_dtype=torch.float16,  # 计算时使用 float16
            bnb_4bit_use_double_quant=True,        # 启用双量化进一步压缩
        )

        # ========== 加载分词器 ==========
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False  # 使用慢速分词器以确保兼容性
        )

        # 设置 pad_token（如果不存在）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("  设置 pad_token = eos_token")

        # ========== 加载基座模型 ==========
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",                     # 自动分配设备
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        # ========== 加载 LoRA 权重（如果有）==========
        if self.lora_path and os.path.exists(self.lora_path):
            print(f"加载 LoRA 权重: {self.lora_path}")
            self.model = PeftModel.from_pretrained(
                base_model,
                self.lora_path,
                torch_dtype=torch.float16
            )
            print("✓ LoRA 权重加载完成")
        else:
            print("未找到 LoRA 权重，使用基座模型")
            self.model = base_model

        # 设置模型为评估模式
        self.model.eval()

        # 初始化对话历史
        self.history = []

        print("✓ 模型加载完成")

    # ==================== 对话方法 ====================

    def chat(self, user_input: str) -> str:
        """
        执行单轮对话

        根据用户输入生成心理咨询师的回复。

        Args:
            user_input: 用户输入的文本

        Returns:
            模型生成的回复文本

        处理流程：
        1. 构建包含历史对话的消息列表
        2. 应用聊天模板
        3. 进行模型推理
        4. 解码并清理输出
        5. 更新对话历史
        """
        # 确保模型已加载
        if self.model is None or self.tokenizer is None:
            self._load_model()

        # ========== 构建消息列表 ==========
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        # 添加历史对话
        for turn in self.history:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})

        # 添加当前用户输入
        messages.append({"role": "user", "content": user_input})

        # ========== 应用聊天模板 ==========
        # 使用 Qwen2 的聊天模板格式化输入
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # 添加助手回复的起始标记
        )

        # ========== Tokenize ==========
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        # ========== 模型推理 ==========
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.inference.max_new_tokens,
                temperature=config.inference.temperature,
                top_p=config.inference.top_p,
                top_k=config.inference.top_k,
                repetition_penalty=config.inference.repetition_penalty,
                do_sample=config.inference.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # ========== 解码输出 ==========
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取助手回复部分
        # Qwen2 使用 <|im_start|>assistant 标记助手回复
        if "<|im_start|>assistant" in full_response:
            response = full_response.split("<|im_start|>assistant")[-1].strip()
        elif "<|im_end|>" in full_response:
            # 尝试通过结束标记提取
            parts = full_response.split("<|im_end|>")
            response = parts[-2].strip() if len(parts) > 1 else full_response
        else:
            # 回退：取原始输入之后的部分
            response = full_response[len(text):].strip()

        # 清理特殊 token
        response = response.replace("<|im_end|>", "").replace("</s>", "").strip()

        # ========== 更新对话历史 ==========
        self.history.append({
            "user": user_input,
            "assistant": response
        })

        return response

    def clear_history(self):
        """
        清空对话历史

        在开始新对话时调用此方法。
        """
        self.history = []
        print("对话历史已清空")

    # ==================== Gradio 界面方法 ====================

    def create_interface(self) -> gr.Blocks:
        """
        创建 Gradio 对话界面

        创建一个包含以下组件的 Web 界面：
        - 标题和模型信息显示
        - 对话历史显示区域
        - 用户输入文本框
        - 发送和清除按钮

        Returns:
            gr.Blocks: Gradio 界面对象
        """
        # ========== 定义主题样式 ==========
        theme = gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        )

        # ========== 创建界面 ==========
        with gr.Blocks(
            theme=theme,
            title="MentalChat 心理咨询对话系统",
            css="""
                .chat-message {
                    padding: 10px;
                    border-radius: 10px;
                    margin-bottom: 10px;
                }
                .user-message {
                    background-color: #e3f2fd;
                }
                .assistant-message {
                    background-color: #f5f5f5;
                }
            """
        ) as demo:
            # ---------- 标题区域 ----------
            gr.Markdown(
                "# 🧠 MentalChat 心理咨询对话系统\n"
                "基于 Qwen2.5-7B-Instruct 微调的心理咨询助手"
            )

            # ---------- 模型信息 ----------
            with gr.Row():
                gr.Markdown(f"**基座模型**: `{config.model.base_model_name}`")
                gr.Markdown(f"**LoRA 权重**: `{self.lora_path or '未加载'}`")

            gr.Markdown("---")

            # ---------- 对话区域 ----------
            # 使用 Chatbot 组件显示对话历史
            chatbot = gr.Chatbot(
                label="对话",
                height=500,
                show_label=True,
                bubble_full_width=False,
            )

            # ---------- 输入区域 ----------
            with gr.Row():
                # 用户输入文本框
                input_box = gr.Textbox(
                    label="输入消息",
                    placeholder="请输入您想说的话...",
                    scale=4,
                    show_label=False,
                )

                # 发送按钮
                submit_btn = gr.Button("发送", scale=1, variant="primary")

            # ---------- 操作按钮 ----------
            with gr.Row():
                clear_btn = gr.Button("🗑️ 清空对话", variant="secondary")

            # ---------- 事件绑定 ----------

            def respond(message: str, chat_history: List):
                """
                处理用户输入并生成回复

                Args:
                    message: 用户输入的消息
                    chat_history: Gradio 聊天历史

                Returns:
                    Tuple: (清空的输入框, 更新后的聊天历史)
                """
                if not message.strip():
                    return "", chat_history

                # 生成回复
                response = self.chat(message)

                # 更新聊天历史
                chat_history.append((message, response))

                return "", chat_history

            def clear_chat():
                """清空对话历史"""
                self.clear_history()
                return []

            # 绑定提交事件（点击发送按钮）
            submit_btn.click(
                fn=respond,
                inputs=[input_box, chatbot],
                outputs=[input_box, chatbot]
            )

            # 绑定回车键提交
            input_box.submit(
                fn=respond,
                inputs=[input_box, chatbot],
                outputs=[input_box, chatbot]
            )

            # 绑定清空按钮
            clear_btn.click(
                fn=clear_chat,
                outputs=[chatbot]
            )

            # ---------- 使用说明 ----------
            gr.Markdown(
                """
                ---
                **使用说明**:
                - 在输入框中输入您想咨询的问题，按回车或点击"发送"按钮
                - 点击"清空对话"可以开始新的对话
                - 模型会以专业心理咨询师的角度回复您

                **注意**: 本系统仅供学习和研究使用，不能替代专业的心理咨询服务。
                """
            )

        return demo

    def run(self, port: int = 7860):
        """
        启动 Gradio 服务

        Args:
            port: 服务端口号

        启动后会显示本地访问地址，如果启用了 share 还会显示公开分享链接。
        """
        print("=" * 60)
        print("启动 Gradio 服务")
        print("=" * 60)

        # 创建界面
        demo = self.create_interface()

        # 打印访问信息
        print(f"本地访问地址: http://localhost:{port}")
        if self.share:
            print("正在创建公开分享链接...")
        print("按 Ctrl+C 停止服务")
        print()

        # 启动服务
        demo.launch(
            server_name="0.0.0.0",  # 允许外部访问
            server_port=port,
            share=self.share,
            show_error=True,
        )


# ==================== 主函数 ====================

def main():
    """
    主函数 - 解析命令行参数并启动服务

    支持的命令行参数：
    --lora-path: LoRA 权重路径
    --share: 是否公开分享
    --port: 服务端口
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="MentalChat Gradio 对话界面",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    # 基本用法
    python chat/app.py

    # 指定 LoRA 权重
    python chat/app.py --lora-path output/checkpoints/final

    # 启用公开分享
    python chat/app.py --share --port 8080
        """
    )

    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="LoRA 权重路径（可选，不指定则使用基座模型）"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="创建 Gradio 公开分享链接"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="服务端口号（默认: 7860）"
    )

    # 解析参数
    args = parser.parse_args()

    # 创建并启动界面
    interface = MentalChatInterface(
        lora_path=args.lora_path,
        share=args.share
    )
    interface.run(port=args.port)

    return 0


# ==================== 程序入口 ====================

if __name__ == "__main__":
    sys.exit(main())
