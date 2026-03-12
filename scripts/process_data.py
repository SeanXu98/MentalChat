#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理脚本
将原始 CSV 数据转换为 ChatML 格式的训练数据
支持 AI 辅助的数据清理和增强（使用 Qwen API）
"""

import os
import sys
import json
import argparse
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import random

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 尝试导入 pandas，如果没有则使用 csv
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    import csv
    HAS_PANDAS = False

# 尝试导入 requests 用于 API 调用
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ==================== 配置 ====================

# 系统提示词（System Prompt）
SYSTEM_PROMPT = """你是一名专业的心理咨询客服，请根据来访者的问题，给出专业、共情的回复。

回复要求：
1. 首先表达对来访者情绪的理解和接纳
2. 分析来访者可能面临的问题
3. 提供具体、可行的建议
4. 以开放式问题或鼓励性话语引导来访者继续表达

注意：保持温和、专业的语气，避免过于绝对的建议，尊重来访者的感受。"""


# ==================== Qwen API 配置 ====================

class QwenAPI:
    """Qwen API 调用类（支持阿里云 DashScope）"""

    def __init__(self, api_key: Optional[str] = None, model: str = "qwen-plus"):
        """
        初始化 Qwen API

        Args:
            api_key: DashScope API Key（可从环境变量 DASHSCOPE_API_KEY 获取）
            model: 模型名称，可选 qwen-turbo, qwen-plus, qwen-max
        """
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        self.model = model
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        self.max_retries = 3
        self.retry_delay = 1.0

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> Optional[str]:
        """
        调用 Qwen API 进行对话

        Args:
            messages: 对话消息列表
            temperature: 生成温度

        Returns:
            生成的回复文本，失败返回 None
        """
        if not self.api_key:
            print("警告: 未配置 DASHSCOPE_API_KEY，跳过 AI 增强")
            return None

        if not HAS_REQUESTS:
            print("警告: 未安装 requests 库，跳过 AI 增强")
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1024
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                elif response.status_code == 429:
                    # 速率限制，等待重试
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    print(f"API 错误: {response.status_code} - {response.text}")
                    return None

            except requests.exceptions.Timeout:
                print(f"请求超时，重试 {attempt + 1}/{self.max_retries}")
                time.sleep(self.retry_delay)
            except Exception as e:
                print(f"API 调用异常: {e}")
                return None

        return None


# ==================== AI 数据清理 ====================

def ai_clean_response(qwen_api: QwenAPI, user_input: str, assistant_response: str) -> Optional[str]:
    """
    使用 AI 清理和优化回复

    Args:
        qwen_api: Qwen API 实例
        user_input: 用户输入
        assistant_response: 原始回复

    Returns:
        优化后的回复
    """
    prompt = f"""你是一个心理咨询数据清洗助手。请检查并优化以下心理咨询回复。

用户输入：
{user_input}

原始回复：
{assistant_response}

请按照以下标准优化回复（如果原回复已经很好，可以直接返回原文）：
1. 确保语气温和、专业
2. 增加共情表达
3. 提供更具体的建议
4. 避免过于绝对的建议
5. 保持适当的回复长度（不要过于简短或冗长）

请直接返回优化后的回复内容，不要添加任何解释或标注："""

    messages = [
        {"role": "system", "content": "你是一个专业的心理咨询数据清洗助手。"},
        {"role": "user", "content": prompt}
    ]

    return qwen_api.chat(messages, temperature=0.3)


# ==================== AI 数据增强 ====================

def ai_paraphrase_input(qwen_api: QwenAPI, user_input: str, num_variations: int = 2) -> List[str]:
    """
    使用 AI 改写用户输入，生成同义变体

    Args:
        qwen_api: Qwen API 实例
        user_input: 原始用户输入
        num_variations: 生成变体数量

    Returns:
        改写后的输入列表
    """
    prompt = f"""请将以下心理咨询来访者的问题改写成 {num_variations} 个意思相近但表达不同的版本。

原始问题：
{user_input}

要求：
1. 保持原意不变
2. 改变表达方式和用词
3. 可以调整语气（如更焦虑、更困惑等）
4. 每个版本一行，不要编号

请直接输出改写后的问题，每行一个："""

    messages = [
        {"role": "system", "content": "你是一个专业的心理咨询数据增强助手。"},
        {"role": "user", "content": prompt}
    ]

    result = qwen_api.chat(messages, temperature=0.8)

    if result:
        variations = [line.strip() for line in result.strip().split('\n') if line.strip()]
        return variations[:num_variations]
    return []


def ai_enhance_response(qwen_api: QwenAPI, user_input: str, original_response: str) -> Optional[str]:
    """
    使用 AI 增强回复质量

    Args:
        qwen_api: Qwen API 实例
        user_input: 用户输入
        original_response: 原始回复

    Returns:
        增强后的回复
    """
    prompt = f"""你是一名资深心理咨询师。请根据来访者的输入，生成一个更专业、更有共情的回复。

来访者输入：
{user_input}

参考回复（可借鉴其思路）：
{original_response}

请生成一个更完善的回复，要求：
1. 首先表达共情和理解
2. 简要分析来访者可能的问题
3. 提供2-3个具体可行的建议
4. 以开放式问题或鼓励性话语结尾
5. 语气温和、专业，避免说教

请直接输出回复内容："""

    messages = [
        {"role": "system", "content": "你是一名资深心理咨询师，擅长提供专业、共情的心理支持。"},
        {"role": "user", "content": prompt}
    ]

    return qwen_api.chat(messages, temperature=0.7)


def ai_generate_follow_up(qwen_api: QwenAPI, conversation: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """
    使用 AI 生成对话的后续轮次

    Args:
        qwen_api: Qwen API 实例
        conversation: 现有对话历史

    Returns:
        生成的新一轮对话 {"user": "...", "assistant": "..."}
    """
    conv_text = "\n".join([
        f"{'来访者' if msg['role'] == 'user' else '咨询师'}：{msg['content']}"
        for msg in conversation if msg['role'] != 'system'
    ])

    prompt = f"""基于以下心理咨询对话，请生成一个合理的后续对话轮次（来访者继续提问，咨询师给予回复）。

对话历史：
{conv_text}

要求：
1. 来访者的问题要自然延续之前的话题
2. 咨询师的回复要专业、共情
3. 对话要有实际意义，不要泛泛而谈

请按以下格式输出（JSON格式）：
{{"user": "来访者的问题", "assistant": "咨询师的回复"}}"""

    messages = [
        {"role": "system", "content": "你是一名专业的心理咨询对话生成助手。"},
        {"role": "user", "content": prompt}
    ]

    result = qwen_api.chat(messages, temperature=0.8)

    if result:
        try:
            # 尝试解析 JSON
            json_match = re.search(r'\{[^}]+\}', result, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data
        except json.JSONDecodeError:
            pass

    return None


def ai_scenario_expansion(qwen_api: QwenAPI, user_input: str, scenario_type: str) -> Optional[Tuple[str, str]]:
    """
    基于现有问题生成不同场景的变体

    Args:
        qwen_api: Qwen API 实例
        user_input: 原始用户输入
        scenario_type: 场景类型（如"青少年"、"职场"、"家庭"等）

    Returns:
        (新用户输入, 新回复)
    """
    prompt = f"""请将以下心理咨询问题改编为【{scenario_type}】场景，并生成相应的专业回复。

原始问题：
{user_input}

要求：
1. 将问题改编为 {scenario_type} 场景
2. 生成适合该场景的专业回复
3. 回复要体现共情和专业性

请按以下 JSON 格式输出：
{{"input": "改编后的问题", "output": "专业回复"}}"""

    messages = [
        {"role": "system", "content": "你是一名专业的心理咨询场景生成助手。"},
        {"role": "user", "content": prompt}
    ]

    result = qwen_api.chat(messages, temperature=0.8)

    if result:
        try:
            json_match = re.search(r'\{[^}]+\}', result, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return (data.get("input", ""), data.get("output", ""))
        except json.JSONDecodeError:
            pass

    return None


# ==================== 数据增强主流程 ====================

def augment_data(
    data: List[Dict[str, Any]],
    qwen_api: Optional[QwenAPI],
    strategies: List[str] = None,
    augment_ratio: float = 0.5,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    数据增强主流程

    Args:
        data: 原始数据
        qwen_api: Qwen API 实例（None 则跳过 AI 增强）
        strategies: 增强策略列表
        augment_ratio: 增强比例（0-1）
        verbose: 是否输出详细信息

    Returns:
        增强后的数据（原始 + 增强）
    """
    if strategies is None:
        strategies = ["paraphrase", "enhance"]  # 默认策略

    if qwen_api is None:
        if verbose:
            print("  未配置 API，跳过 AI 增强")
        return data

    augmented_data = data.copy()
    augment_count = int(len(data) * augment_ratio)

    if verbose:
        print(f"\n[AI 数据增强]")
        print(f"  原始数据: {len(data)} 条")
        print(f"  增强策略: {strategies}")
        print(f"  目标增强: {augment_count} 条")

    # 随机选择要增强的样本
    indices = random.sample(range(len(data)), min(augment_count, len(data)))

    success_count = 0
    for i, idx in enumerate(indices):
        record = data[idx]
        user_input = record.get('Input', '') or record.get('input', '')
        assistant_output = record.get('Output', '') or record.get('output', '')

        strategy = random.choice(strategies)

        try:
            if strategy == "paraphrase":
                # 同义改写
                variations = ai_paraphrase_input(qwen_api, user_input, num_variations=1)
                if variations:
                    new_record = record.copy()
                    new_record['Input'] = variations[0]
                    new_record['_augmented'] = 'paraphrase'
                    augmented_data.append(new_record)
                    success_count += 1

            elif strategy == "enhance":
                # 回复增强
                enhanced_response = ai_enhance_response(qwen_api, user_input, assistant_output)
                if enhanced_response:
                    new_record = record.copy()
                    new_record['Output'] = enhanced_response
                    new_record['_augmented'] = 'enhance'
                    augmented_data.append(new_record)
                    success_count += 1

            elif strategy == "clean":
                # 回复清理
                cleaned_response = ai_clean_response(qwen_api, user_input, assistant_output)
                if cleaned_response:
                    # 直接更新原记录
                    augmented_data[idx] = record.copy()
                    augmented_data[idx]['Output'] = cleaned_response
                    augmented_data[idx]['_cleaned'] = True
                    success_count += 1

            # 进度显示
            if verbose and (i + 1) % 10 == 0:
                print(f"  进度: {i + 1}/{len(indices)}, 成功: {success_count}")

            # 避免触发速率限制
            time.sleep(0.5)

        except Exception as e:
            if verbose:
                print(f"  增强失败 (索引 {idx}): {e}")

    if verbose:
        print(f"  ✓ 增强完成: 新增 {success_count} 条")
        print(f"  ✓ 最终数据: {len(augmented_data)} 条")

    return augmented_data


# ==================== 数据加载 ====================

def load_csv_data(filepath: str) -> List[Dict[str, Any]]:
    """加载 CSV 数据"""
    if HAS_PANDAS:
        df = pd.read_csv(filepath)
        return df.to_dict('records')
    else:
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data


# ==================== 数据探查 ====================

def explore_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """数据探查与统计"""
    print("=" * 60)
    print("数据探查报告")
    print("=" * 60)

    # 基本统计
    total_records = len(data)
    print(f"\n[基本信息]")
    print(f"  总记录数: {total_records}")

    # 按对话分组
    conversations = defaultdict(list)
    for record in data:
        conv_id = record.get('Conversation ID') or record.get('conversation_id')
        if conv_id is not None:
            conversations[conv_id].append(record)

    num_conversations = len(conversations)
    print(f"  对话总数: {num_conversations}")

    # 对话轮次分布
    turn_counts = [len(turns) for turns in conversations.values()]
    avg_turns = sum(turn_counts) / len(turn_counts) if turn_counts else 0
    max_turns = max(turn_counts) if turn_counts else 0
    min_turns = min(turn_counts) if turn_counts else 0

    print(f"\n[对话轮次分布]")
    print(f"  平均轮次: {avg_turns:.2f}")
    print(f"  最大轮次: {max_turns}")
    print(f"  最小轮次: {min_turns}")

    # 文本长度分布
    input_lengths = []
    output_lengths = []

    for record in data:
        input_text = record.get('Input', '') or record.get('input', '')
        output_text = record.get('Output', '') or record.get('output', '')
        input_lengths.append(len(str(input_text)))
        output_lengths.append(len(str(output_text)))

    print(f"\n[Input 长度分布]")
    print(f"  平均长度: {sum(input_lengths)/len(input_lengths):.1f} 字符")
    print(f"  最大长度: {max(input_lengths)}")
    print(f"  最小长度: {min(input_lengths)}")

    print(f"\n[Output 长度分布]")
    print(f"  平均长度: {sum(output_lengths)/len(output_lengths):.1f} 字符")
    print(f"  最大长度: {max(output_lengths)}")
    print(f"  最小长度: {min(output_lengths)}")

    # 异常数据检测
    print(f"\n[异常数据检测]")

    empty_input = sum(1 for l in input_lengths if l == 0)
    empty_output = sum(1 for l in output_lengths if l == 0)
    too_short_input = sum(1 for l in input_lengths if 0 < l < 5)
    too_long_input = sum(1 for l in input_lengths if l > 500)
    too_long_output = sum(1 for l in output_lengths if l > 1000)

    if empty_input > 0:
        print(f"  ⚠ Input 为空: {empty_input} 条")
    if empty_output > 0:
        print(f"  ⚠ Output 为空: {empty_output} 条")
    if too_short_input > 0:
        print(f"  ⚠ Input 过短(<5字符): {too_short_input} 条")
    if too_long_input > 0:
        print(f"  ⚠ Input 过长(>500字符): {too_long_input} 条")
    if too_long_output > 0:
        print(f"  ⚠ Output 过长(>1000字符): {too_long_output} 条")

    return {
        "total_records": total_records,
        "num_conversations": num_conversations,
        "avg_turns": avg_turns,
        "empty_input": empty_input,
        "empty_output": empty_output,
    }


# ==================== 数据清洗 ====================

def clean_text(text: str) -> str:
    """清洗文本"""
    if not text:
        return ""

    # 去除首尾空白
    text = text.strip()

    # 统一换行符
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # 去除多余空格
    text = re.sub(r' +', ' ', text)

    # 去除多余换行
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text


def is_valid_record(record: Dict[str, Any]) -> Tuple[bool, str]:
    """检查记录是否有效"""
    input_text = record.get('Input', '') or record.get('input', '')
    output_text = record.get('Output', '') or record.get('output', '')

    input_text = str(input_text).strip()
    output_text = str(output_text).strip()

    if not input_text:
        return False, "Input 为空"
    if not output_text:
        return False, "Output 为空"
    if len(input_text) < 5:
        return False, "Input 过短"
    if len(output_text) < 10:
        return False, "Output 过短"

    return True, ""


def clean_data(data: List[Dict[str, Any]], verbose: bool = True) -> List[Dict[str, Any]]:
    """清洗数据"""
    if verbose:
        print("\n" + "=" * 60)
        print("数据清洗")
        print("=" * 60)

    cleaned_data = []
    removed_counts = defaultdict(int)

    for record in data:
        is_valid, reason = is_valid_record(record)
        if is_valid:
            # 清洗文本
            cleaned_record = record.copy()
            if 'Input' in cleaned_record:
                cleaned_record['Input'] = clean_text(str(cleaned_record.get('Input', '')))
            if 'Output' in cleaned_record:
                cleaned_record['Output'] = clean_text(str(cleaned_record.get('Output', '')))
            if 'input' in cleaned_record:
                cleaned_record['input'] = clean_text(str(cleaned_record.get('input', '')))
            if 'output' in cleaned_record:
                cleaned_record['output'] = clean_text(str(cleaned_record.get('output', '')))
            cleaned_data.append(cleaned_record)
        else:
            removed_counts[reason] += 1

    if verbose:
        print(f"  原始数据: {len(data)} 条")
        print(f"  清洗后数据: {len(cleaned_data)} 条")
        print(f"  移除数据: {len(data) - len(cleaned_data)} 条")
        if removed_counts:
            print("\n  移除原因:")
            for reason, count in removed_counts.items():
                print(f"    - {reason}: {count} 条")

    return cleaned_data


# ==================== 格式转换 ====================

def convert_to_chatml_single_turn(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """转换为 ChatML 单轮对话格式"""
    chatml_data = []

    for record in data:
        input_text = record.get('Input', '') or record.get('input', '')
        output_text = record.get('Output', '') or record.get('output', '')

        chatml_item = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(input_text).strip()},
                {"role": "assistant", "content": str(output_text).strip()}
            ]
        }
        chatml_data.append(chatml_item)

    return chatml_data


def convert_to_chatml_multi_turn(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """转换为 ChatML 多轮对话格式"""
    # 按对话ID分组
    conversations = defaultdict(list)
    for record in data:
        conv_id = record.get('Conversation ID') or record.get('conversation_id')
        if conv_id is not None:
            conversations[conv_id].append(record)

    chatml_data = []

    for conv_id, turns in conversations.items():
        # 按轮次ID排序
        turns.sort(key=lambda x: int(x.get('Turn ID', x.get('turn_id', 0))))

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for turn in turns:
            input_text = turn.get('Input', '') or turn.get('input', '')
            output_text = turn.get('Output', '') or turn.get('output', '')

            messages.append({"role": "user", "content": str(input_text).strip()})
            messages.append({"role": "assistant", "content": str(output_text).strip()})

        chatml_item = {
            "conversation_id": str(conv_id),
            "messages": messages
        }
        chatml_data.append(chatml_item)

    return chatml_data


# ==================== 数据集划分 ====================

def split_dataset(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """划分数据集"""
    random.seed(seed)
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    n_total = len(shuffled_data)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)

    train_data = shuffled_data[:n_train]
    valid_data = shuffled_data[n_train:n_train + n_valid]
    test_data = shuffled_data[n_train + n_valid:]

    return train_data, valid_data, test_data


# ==================== 数据保存 ====================

def save_jsonl(data: List[Dict[str, Any]], filepath: str):
    """保存为 JSONL 格式"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def save_json(data: List[Dict[str, Any]], filepath: str):
    """保存为 JSON 格式"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ==================== 主流程 ====================

def process_data(
    input_file: str,
    output_dir: str,
    mode: str = "multi",
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    explore_only: bool = False,
    augment: bool = False,
    augment_ratio: float = 0.3,
    augment_strategies: List[str] = None,
    api_key: str = None,
    api_model: str = "qwen-plus"
):
    """
    主处理流程

    Args:
        input_file: 输入 CSV 文件路径
        output_dir: 输出目录
        mode: 转换模式 ("single" 单轮, "multi" 多轮)
        train_ratio: 训练集比例
        valid_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        explore_only: 仅进行数据探查
        augment: 是否启用 AI 数据增强
        augment_ratio: 增强比例
        augment_strategies: 增强策略
        api_key: Qwen API Key
        api_model: Qwen 模型名称
    """
    print("=" * 60)
    print("心理咨询对话数据处理 - ChatML 格式转换")
    print("=" * 60)
    print(f"输入文件: {input_file}")
    print(f"输出目录: {output_dir}")
    print(f"转换模式: {mode}-turn")
    print(f"AI 增强: {'启用' if augment else '禁用'}")
    print()

    # 1. 加载数据
    print("1. 加载数据...")
    data = load_csv_data(input_file)
    print(f"   加载完成: {len(data)} 条记录")

    # 2. 数据探查
    print("\n2. 数据探查...")
    stats = explore_data(data)

    if explore_only:
        print("\n[仅探查模式] 处理完成")
        return

    # 3. 数据清洗
    print("\n3. 数据清洗...")
    cleaned_data = clean_data(data)

    # 4. AI 数据增强（可选）
    if augment:
        print("\n4. AI 数据增强...")
        qwen_api = QwenAPI(api_key=api_key, model=api_model)
        augmented_data = augment_data(
            cleaned_data,
            qwen_api=qwen_api,
            strategies=augment_strategies,
            augment_ratio=augment_ratio,
            verbose=True
        )
    else:
        augmented_data = cleaned_data
        print("\n4. AI 数据增强: 跳过")

    # 5. 格式转换
    print(f"\n5. 格式转换 ({mode}-turn ChatML)...")
    if mode == "single":
        chatml_data = convert_to_chatml_single_turn(augmented_data)
    else:
        chatml_data = convert_to_chatml_multi_turn(augmented_data)
    print(f"   转换完成: {len(chatml_data)} 条对话")

    # 6. 数据集划分
    print("\n6. 数据集划分...")
    train_data, valid_data, test_data = split_dataset(
        chatml_data,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        seed=seed
    )
    print(f"   训练集: {len(train_data)} 条 ({len(train_data)/len(chatml_data)*100:.1f}%)")
    print(f"   验证集: {len(valid_data)} 条 ({len(valid_data)/len(chatml_data)*100:.1f}%)")
    print(f"   测试集: {len(test_data)} 条 ({len(test_data)/len(chatml_data)*100:.1f}%)")

    # 7. 保存数据
    print("\n7. 保存数据...")

    # 保存为 JSONL 格式（推荐，适合训练）
    save_jsonl(train_data, os.path.join(output_dir, "train.jsonl"))
    save_jsonl(valid_data, os.path.join(output_dir, "valid.jsonl"))
    save_jsonl(test_data, os.path.join(output_dir, "test.jsonl"))

    print(f"   ✓ train.jsonl ({len(train_data)} 条)")
    print(f"   ✓ valid.jsonl ({len(valid_data)} 条)")
    print(f"   ✓ test.jsonl ({len(test_data)} 条)")

    # 8. 数据示例
    print("\n8. 数据示例 (ChatML 格式):")
    print("-" * 40)
    if chatml_data:
        sample = chatml_data[0]
        print(json.dumps(sample, ensure_ascii=False, indent=2)[:500] + "...")

    # 9. 完成
    print("\n" + "=" * 60)
    print("✓ 数据处理完成！")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  - {output_dir}/train.jsonl")
    print(f"  - {output_dir}/valid.jsonl")
    print(f"  - {output_dir}/test.jsonl")
    print(f"\n数据统计:")
    print(f"  - 原始数据: {stats['total_records']} 条")
    print(f"  - 清洗后: {len(cleaned_data)} 条")
    if augment:
        print(f"  - 增强后: {len(augmented_data)} 条")
    print(f"  - 最终对话: {len(chatml_data)} 组")


def main():
    parser = argparse.ArgumentParser(
        description="数据处理脚本 - 转换为 ChatML 格式，支持 AI 数据增强",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基础数据处理
  python scripts/process_data.py --mode multi

  # 仅探查数据
  python scripts/process_data.py --explore-only

  # 启用 AI 数据增强（需要配置 DASHSCOPE_API_KEY 环境变量）
  python scripts/process_data.py --augment --augment-ratio 0.5

  # 使用指定 API Key 和模型
  python scripts/process_data.py --augment --api-key YOUR_KEY --api-model qwen-plus

增强策略说明:
  paraphrase  - 同义改写：改写用户输入，生成新的训练样本
  enhance     - 回复增强：优化回复内容，使其更专业
  clean       - 回复清理：修复语法错误，标准化格式
"""
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "data.csv"),
        help="输入 CSV 文件路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed"),
        help="输出目录"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "multi"],
        default="multi",
        help="转换模式: single(单轮对话), multi(多轮对话，默认)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="训练集比例 (默认: 0.8)"
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.1,
        help="验证集比例 (默认: 0.1)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="测试集比例 (默认: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)"
    )
    parser.add_argument(
        "--explore-only",
        action="store_true",
        help="仅进行数据探查，不进行转换"
    )

    # AI 数据增强参数
    parser.add_argument(
        "--augment",
        action="store_true",
        help="启用 AI 数据增强（需要 Qwen API）"
    )
    parser.add_argument(
        "--augment-ratio",
        type=float,
        default=0.3,
        help="增强比例 (默认: 0.3，即增强 30%% 的数据)"
    )
    parser.add_argument(
        "--augment-strategies",
        type=str,
        nargs="+",
        choices=["paraphrase", "enhance", "clean"],
        default=["paraphrase", "enhance"],
        help="增强策略: paraphrase(同义改写), enhance(回复增强), clean(清理)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="DashScope API Key（也可通过 DASHSCOPE_API_KEY 环境变量设置）"
    )
    parser.add_argument(
        "--api-model",
        type=str,
        default="qwen-plus",
        choices=["qwen-turbo", "qwen-plus", "qwen-max"],
        help="Qwen 模型: qwen-turbo(快), qwen-plus(平衡), qwen-max(强)"
    )

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return 1

    # 处理数据
    process_data(
        input_file=args.input,
        output_dir=args.output_dir,
        mode=args.mode,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        explore_only=args.explore_only,
        augment=args.augment,
        augment_ratio=args.augment_ratio,
        augment_strategies=args.augment_strategies,
        api_key=args.api_key,
        api_model=args.api_model
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
