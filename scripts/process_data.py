#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理脚本
============

本脚本将原始 CSV 数据转换为 ChatML 格式的训练数据，用于模型微调。

主要功能：
    - 数据探查：统计分析原始数据的质量和分布
    - 数据清洗：移除无效数据，标准化文本格式
    - 格式转换：将 CSV 转换为 ChatML 格式（支持单轮和多轮对话）
    - 数据集划分：自动划分训练集、验证集、测试集
    - AI 数据增强：可选的数据增强功能

运行方法：
    # 多轮对话模式（推荐）
    python scripts/process_data.py --mode multi

    # 单轮对话模式
    python scripts/process_data.py --mode single

    # 仅探查数据（不进行转换）
    python scripts/process_data.py --explore-only

    # 启用 AI 数据增强（使用 GLM API，推荐）
    python scripts/process_data.py --augment --augment-ratio 0.3 --api-type glm

    # 启用 AI 数据增强（使用 Qwen API）
    python scripts/process_data.py --augment --augment-ratio 0.3 --api-type qwen

输入格式：
    CSV 文件，应包含以下字段：
    - Conversation ID / conversation_id: 对话唯一标识
    - Turn ID / turn_id: 对话轮次标识
    - Input / input: 用户输入
    - Output / output: 咨询师回复

输出格式：
    JSONL 文件（ChatML 格式），每行一个对话：
    {
        "conversation_id": "xxx",
        "messages": [
            {"role": "system", "content": "系统提示词"},
            {"role": "user", "content": "用户消息"},
            {"role": "assistant", "content": "咨询师回复"}
        ]
    }

作者：MentalChat 项目组
"""

# ==================== 标准库导入 ====================
import os
import sys
import json
import argparse
import re
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

# ==================== 项目内部导入 ====================
# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import config

# ==================== 第三方库导入 ====================
# 尝试导入 pandas（可选依赖）
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    import csv
    HAS_PANDAS = False


# ==================== 工具函数 ====================

def ensure_dir(path: str):
    """
    确保目录存在

    Args:
        path: 目录路径，如果不存在则自动创建
    """
    os.makedirs(path, exist_ok=True)


def read_jsonl(filepath: str) -> List[Dict]:
    """
    读取 JSONL 文件

    JSONL 格式是每行一个 JSON 对象，便于流式处理大文件。

    Args:
        filepath: JSONL 文件路径

    Returns:
        解析后的数据列表
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def write_jsonl(data: List[Dict], filepath: str):
    """
    写入 JSONL 文件

    Args:
        data: 要写入的数据列表
        filepath: 输出文件路径
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


# ==================== 数据加载 ====================

def load_csv_data(filepath: str) -> List[Dict[str, Any]]:
    """
    加载 CSV 数据文件

    优先使用 pandas（如果可用），否则使用标准库 csv 模块。

    Args:
        filepath: CSV 文件路径

    Returns:
        包含所有记录的字典列表
    """
    if HAS_PANDAS:
        # 使用 pandas 读取（更快，支持更多格式）
        df = pd.read_csv(filepath)
        return df.to_dict('records')
    else:
        # 使用标准库 csv 模块作为备选
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(dict(row))
        return data


# ==================== 数据探查 ====================

def explore_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    数据探查与统计分析

    对原始数据进行全面的统计分析，包括：
        - 基本统计（记录数、对话数）
        - 对话轮次分布
        - 文本长度分布
        - 异常数据检测

    Args:
        data: 原始数据列表

    Returns:
        统计信息字典
    """
    print("=" * 60)
    print("数据探查报告")
    print("=" * 60)

    # ========== 基本统计 ==========
    total_records = len(data)
    print(f"\n[基本信息]")
    print(f"  总记录数: {total_records}")

    # 按对话 ID 分组统计
    conversations = defaultdict(list)
    for record in data:
        # 兼容不同的字段名格式
        conv_id = record.get('Conversation ID') or record.get('conversation_id')
        if conv_id is not None:
            conversations[conv_id].append(record)

    num_conversations = len(conversations)
    print(f"  对话总数: {num_conversations}")

    # ========== 对话轮次分布 ==========
    turn_counts = [len(turns) for turns in conversations.values()]
    avg_turns = sum(turn_counts) / len(turn_counts) if turn_counts else 0
    max_turns = max(turn_counts) if turn_counts else 0
    min_turns = min(turn_counts) if turn_counts else 0

    print(f"\n[对话轮次分布]")
    print(f"  平均轮次: {avg_turns:.2f}")
    print(f"  最大轮次: {max_turns}")
    print(f"  最小轮次: {min_turns}")

    # ========== 文本长度分布 ==========
    input_lengths = []
    output_lengths = []
    for record in data:
        # 兼容不同的字段名格式
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

    # ========== 异常数据检测 ==========
    print(f"\n[异常数据检测]")
    empty_input = sum(1 for l in input_lengths if l == 0)
    empty_output = sum(1 for l in output_lengths if l == 0)
    too_short_input = sum(1 for l in input_lengths if 0 < l < 5)
    too_long_input = sum(1 for l in input_lengths if l > 500)
    too_long_output = sum(1 for l in output_lengths if l > 1000)

    # 统计各类异常
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
    """
    清洗文本内容

    处理内容包括：
        - 移除首尾空白
        - 统一换行符格式
        - 压缩多余空格
        - 压缩多余换行

    Args:
        text: 原始文本

    Returns:
        清洗后的文本
    """
    if not text:
        return ""

    text = str(text).strip()
    text = text.replace('\r\n', '\n').replace('\r', '\n')  # 统一换行符
    text = re.sub(r' +', ' ', text)                        # 压缩多余空格
    text = re.sub(r'\n{3,}', '\n\n', text)                 # 压缩多余换行

    return text


def is_valid_record(record: Dict[str, Any]) -> Tuple[bool, str]:
    """
    检查记录是否有效

    验证规则：
        - Input 和 Output 不能为空
        - Input 长度至少 5 个字符
        - Output 长度至少 10 个字符

    Args:
        record: 数据记录

    Returns:
        Tuple[bool, str]: (是否有效, 无效原因)
    """
    # 获取输入和输出（兼容不同字段名）
    input_text = record.get('Input', '') or record.get('input', '')
    output_text = record.get('Output', '') or record.get('output', '')
    input_text = str(input_text).strip()
    output_text = str(output_text).strip()

    # 验证规则
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
    """
    清洗数据集

    处理流程：
        1. 遍历所有记录
        2. 验证每条记录的有效性
        3. 清洗有效记录的文本内容
        4. 统计并报告移除的记录

    Args:
        data: 原始数据列表
        verbose: 是否输出详细信息

    Returns:
        清洗后的数据列表
    """
    if verbose:
        print("\n" + "=" * 60)
        print("数据清洗")
        print("=" * 60)

    cleaned_data = []
    removed_counts = defaultdict(int)

    for record in data:
        is_valid, reason = is_valid_record(record)

        if is_valid:
            # 创建清洗后的记录
            cleaned_record = record.copy()

            # 清洗文本字段（兼容不同字段名）
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
            # 记录移除原因
            removed_counts[reason] += 1

    # 打印清洗统计
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
    """
    转换为 ChatML 单轮对话格式

    每条记录转换为一个独立的对话，适用于不需要上下文的场景。

    Args:
        data: 清洗后的数据列表

    Returns:
        ChatML 格式的数据列表

    输出格式：
        {
            "messages": [
                {"role": "system", "content": "系统提示词"},
                {"role": "user", "content": "用户输入"},
                {"role": "assistant", "content": "咨询师回复"}
            ]
        }
    """
    chatml_data = []
    system_prompt = config.data.system_prompt

    for record in data:
        # 获取输入和输出（兼容不同字段名）
        input_text = record.get('Input', '') or record.get('input', '')
        output_text = record.get('Output', '') or record.get('output', '')

        # 构建 ChatML 格式
        chatml_item = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(input_text).strip()},
                {"role": "assistant", "content": str(output_text).strip()}
            ]
        }
        chatml_data.append(chatml_item)

    return chatml_data


def convert_to_chatml_multi_turn(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    转换为 ChatML 多轮对话格式

    将同一对话 ID 的多轮交互合并为一个完整的对话。
    这种格式更接近真实场景，模型可以学习上下文理解。

    Args:
        data: 清洗后的数据列表

    Returns:
        ChatML 格式的数据列表

    输出格式：
        {
            "conversation_id": "xxx",
            "messages": [
                {"role": "system", "content": "系统提示词"},
                {"role": "user", "content": "第一轮用户输入"},
                {"role": "assistant", "content": "第一轮回复"},
                {"role": "user", "content": "第二轮用户输入"},
                {"role": "assistant", "content": "第二轮回复"}
            ]
        }
    """
    system_prompt = config.data.system_prompt

    # ========== 按对话 ID 分组 ==========
    conversations = defaultdict(list)
    for record in data:
        conv_id = record.get('Conversation ID') or record.get('conversation_id')
        if conv_id is not None:
            conversations[conv_id].append(record)

    chatml_data = []

    for conv_id, turns in conversations.items():
        # 按轮次 ID 排序
        turns.sort(key=lambda x: int(x.get('Turn ID', x.get('turn_id', 0))))

        # 构建消息列表
        messages = [{"role": "system", "content": system_prompt}]

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
    """
    划分数据集

    将数据随机打乱后按比例划分为训练集、验证集和测试集。

    Args:
        data: 完整数据集
        train_ratio: 训练集比例（默认 0.8）
        valid_ratio: 验证集比例（默认 0.1）
        test_ratio: 测试集比例（默认 0.1）
        seed: 随机种子（保证可复现）

    Returns:
        Tuple[train_data, valid_data, test_data]

    注意：
        - 三个比例之和应该等于 1.0
        - 使用固定种子保证结果可复现
    """
    random.seed(seed)

    # 随机打乱数据
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    # 计算划分点
    n_total = len(shuffled_data)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)

    # 划分数据集
    train_data = shuffled_data[:n_train]
    valid_data = shuffled_data[n_train:n_train + n_valid]
    test_data = shuffled_data[n_train + n_valid:]

    return train_data, valid_data, test_data


# ==================== AI 数据增强 ====================

# 注意：API 客户端类（QwenAPI、GLMAPI）在 scripts/augmentation.py 中定义
# 这里直接导入使用，避免代码重复
from scripts.augmentation import QwenAPI, GLMAPI, create_api_client


def augment_data(
    data: List[Dict[str, Any]],
    api,
    strategies: List[str] = None,
    augment_ratio: float = 0.3,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    数据增强

    使用 AI 模型对数据进行增强，生成更多的训练样本。

    Args:
        data: 原始数据列表
        api: API 客户端实例（支持 QwenAPI 或 GLMAPI）
        strategies: 增强策略列表（目前只支持 paraphrase）
        augment_ratio: 增强比例
        verbose: 是否输出详细信息

    Returns:
        增强后的数据列表（包含原始数据）
    """
    import time

    # 检查 API 是否可用
    if api is None or api.api_key is None:
        if verbose:
            print("   未配置 API Key，跳过数据增强")
        return data

    if strategies is None:
        strategies = ["paraphrase"]

    # 复制原始数据
    augmented_data = data.copy()

    # 计算需要增强的数据量
    num_to_augment = int(len(data) * augment_ratio)
    indices = random.sample(range(len(data)), min(num_to_augment, len(data)))

    # ========== 打印增强任务信息 ==========
    if verbose:
        print(f"   增强策略: {', '.join(strategies)}")
        print(f"   目标数量: {num_to_augment} 条 ({augment_ratio*100:.1f}%)")

    # ========== 测试 API 连通性 ==========
    if verbose:
        print(f"   测试 API 连通性...")
    try:
        # 使用 raise_on_error=True 来获取详细错误信息
        test_result = api.call("你好，请回复'测试成功'", max_tokens=50, raise_on_error=True)
        if test_result:
            if verbose:
                # 截断过长的响应
                response_preview = test_result[:50] + "..." if len(test_result) > 50 else test_result
                print(f"   ✓ API 连接正常")
                print(f"   响应示例: {response_preview}")
                print(f"   开始处理...")
                print()
        else:
            print(f"   ✗ API 返回为空，请检查 API Key 和模型配置")
            print(f"   提示: 确保已设置正确的环境变量 ZHIPUAI_API_KEY 或 DASHSCOPE_API_KEY")
            return data
    except Exception as e:
        print(f"   ✗ API 连接失败: {e}")
        print(f"   提示: 请检查网络连接、API Key 和模型名称是否正确")
        return data

    # ========== 执行增强（带进度显示）==========
    success_count = 0
    fail_count = 0
    error_messages = []  # 收集错误信息
    consecutive_failures = 0  # 连续失败计数
    max_consecutive_failures = 10  # 连续失败超过此次数则停止
    start_time = time.time()

    for i, idx in enumerate(indices):
        record = data[idx]
        user_input = record.get('Input', '') or record.get('input', '')
        user_output = record.get('Output', '') or record.get('output', '')

        for strategy in strategies:
            prompt = None
            target_field = None
            strategy_name = None

            if strategy == "paraphrase":
                # 同义改写策略：改写用户输入
                prompt = f"请将以下用户问题改写为意思相近但表达不同的版本，只需输出改写后的问题：\n\n{user_input}"
                target_field = 'Input'
                strategy_name = 'paraphrase'

            elif strategy == "enhance":
                # 回复增强策略：优化咨询师回复
                prompt = f"""你是一位专业的心理咨询师。请优化以下咨询回复，使其更加专业、更有共情心。
要求：
1. 保持原意不变
2. 使用更温暖、更专业的表达
3. 适当增加共情语句
4. 只输出优化后的回复

用户问题：{user_input}

原回复：{user_output}

优化后的回复："""
                target_field = 'Output'
                strategy_name = 'enhance'

            elif strategy == "scenario":
                # 场景扩展策略：将问题改编为不同场景
                prompt = f"""请将以下心理咨询问题改写为不同的生活场景，保持核心心理问题不变。
要求：
1. 改变具体的情境描述
2. 保持问题的心理本质
3. 只输出改写后的问题

原问题：{user_input}

改写后的问题："""
                target_field = 'Input'
                strategy_name = 'scenario'

            # 如果有有效的策略，执行 API 调用
            if prompt and target_field:
                try:
                    # 使用 raise_on_error=True 来获取详细错误
                    result = api.call(prompt, max_tokens=300, raise_on_error=True)

                    if result and len(result.strip()) > 10:
                        new_record = record.copy()
                        new_record[target_field] = result.strip()
                        new_record['_augmented'] = strategy_name
                        augmented_data.append(new_record)
                        success_count += 1
                        consecutive_failures = 0  # 成功后重置连续失败计数
                    else:
                        fail_count += 1
                        consecutive_failures += 1
                        error_msg = f"返回为空" if result is None else f"返回过短({len(result.strip())}字符)"
                        error_messages.append(f"第 {i+1} 条 [{strategy}]: {error_msg}")
                except Exception as e:
                    fail_count += 1
                    consecutive_failures += 1
                    error_messages.append(f"第 {i+1} 条 [{strategy}]: {str(e)}")

                # 检查连续失败次数，避免无效的 API 调用
                if consecutive_failures >= max_consecutive_failures:
                    print()  # 换行
                    print()
                    print(f"   ⚠ 连续失败 {consecutive_failures} 次，停止增强")
                    print(f"   最后错误: {error_messages[-1] if error_messages else '未知'}")
                    print(f"   请检查 API 配置是否正确")
                    break  # 跳出策略循环

        # 检查是否需要跳出外层循环
        if consecutive_failures >= max_consecutive_failures:
            break  # 跳出数据循环

        # ========== 显示进度和样例（每100条显示一条） ==========
        if verbose:
            current = i + 1
            elapsed = time.time() - start_time
            avg_time = elapsed / current if current > 0 else 0
            remaining = avg_time * (num_to_augment - current)

            # 计算进度百分比
            progress = current / num_to_augment * 100

            # 创建进度条
            bar_length = 30
            filled = int(bar_length * current / num_to_augment)
            bar = '█' * filled + '░' * (bar_length - filled)

            # 格式化时间
            if remaining > 60:
                time_str = f"{int(remaining // 60)}分{int(remaining % 60)}秒"
            else:
                time_str = f"{int(remaining)}秒"

            # 每处理100条打印一条样例
            if success_count > 0 and success_count % 100 == 0:
                # 蟥找最近成功增强的记录
                last_record = augmented_data[-1]
                strategy_type = last_record.get('_augmented', 'unknown')
                original_input = data[idx].get('Input', '') or data[idx].get('input', '')[:60]
                new_content = last_record.get('Input', '') or last_record.get('Output', '')

                print()  # 换行
                print(f"   📝 样例 #{success_count} [{strategy_type}]:")
                print(f"      原始: {original_input}")
                print(f"      增强: {new_content}")

            # 打印进度（覆盖上一行）
            print(f"\r   [{bar}] {current}/{num_to_augment} ({progress:.1f}%) | "
                  f"成功: {success_count} | 失败: {fail_count} | "
                  f"预计剩余: {time_str}    ", end='', flush=True)

    # ========== 打印完成信息 ==========
    if verbose:
        total_time = time.time() - start_time
        if total_time > 60:
            time_str = f"{total_time // 60:.0f}分{total_time % 60:.0f}秒"
        else:
            time_str = f"{total_time:.1f}秒"

        print()  # 换行
        print()
        print(f"   ✓ 增强完成:")
        print(f"     - 成功: {success_count} 条")
        print(f"     - 失败: {fail_count} 条")
        if (success_count + fail_count) > 0:
            print(f"     - 成功率: {success_count/(success_count+fail_count)*100:.1f}%")
        print(f"     - 耗时: {time_str}")
        print(f"     - 数据总量: {len(data)} → {len(augmented_data)} 条")

        # 显示错误详情（最多显示10条）
        if fail_count > 0 and error_messages:
            print()
            print(f"   错误详情 (显示前10条):")
            for msg in error_messages[:10]:
                print(f"     - {msg}")
            if len(error_messages) > 10:
                print(f"     ... 还有 {len(error_messages) - 10} 条错误")

    return augmented_data


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
    api_type: str = "glm",
    api_key: str = None,
    api_model: str = None,
    verbose: bool = True
):
    """
    主处理流程

    完整的数据处理流程，包括：
        1. 加载数据
        2. 数据探查
        3. 数据清洗
        4. AI 数据增强（可选）
        5. 格式转换
        6. 数据集划分
        7. 保存数据

    Args:
        input_file: 输入 CSV 文件路径
        output_dir: 输出目录
        mode: 转换模式，"single" 或 "multi"
        train_ratio: 训练集比例
        valid_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        explore_only: 是否仅进行数据探查
        augment: 是否启用 AI 数据增强
        augment_ratio: 增强比例
        augment_strategies: 增强策略列表
        api_type: API 类型（"glm" 或 "qwen"）
        api_key: API 密钥（未提供时从环境变量读取）
        api_model: 模型名称（未提供时使用默认模型）
        verbose: 是否输出详细信息
    """
    # ========== 打印配置信息 ==========
    print("=" * 60)
    print("心理咨询对话数据处理 - ChatML 格式转换")
    print("=" * 60)
    print(f"输入文件: {input_file}")
    print(f"输出目录: {output_dir}")
    print(f"转换模式: {mode}-turn")
    print(f"AI 增强: {'启用 (' + api_type.upper() + ')' if augment else '禁用'}")
    print()

    # ========== 1. 加载数据 ==========
    if verbose:
        print("1. 加载数据...")
    data = load_csv_data(input_file)
    print(f"   加载完成: {len(data)} 条记录")

    # ========== 2. 数据探查 ==========
    if verbose:
        print("\n2. 数据探查...")
    stats = explore_data(data)

    # 如果仅探查模式，到此结束
    if explore_only:
        print("\n[仅探查模式] 处理完成")
        return

    # ========== 3. 数据清洗 ==========
    if verbose:
        print("\n3. 数据清洗...")
    cleaned_data = clean_data(data, verbose=verbose)

    # ========== 4. AI 数据增强（可选）==========
    if verbose:
        print("\n4. AI 数据增强...")
    if augment:
        # 使用工厂函数创建 API 客户端
        api_client = create_api_client(api_type=api_type, api_key=api_key, model=api_model)

        if api_client is not None:
            # 设置默认模型名称（用于显示）
            display_model = api_model or ("glm-4.7" if api_type.lower() == "glm" else "qwen-plus")
            print(f"   使用 {api_type.upper()} API，模型: {display_model}")

            # 执行数据增强
            augmented_data = augment_data(
                cleaned_data,
                api=api_client,
                strategies=augment_strategies,
                augment_ratio=augment_ratio,
                verbose=verbose
            )
        else:
            augmented_data = cleaned_data
            print("   跳过（API 客户端创建失败）")
    else:
        augmented_data = cleaned_data
        print("   跳过")

    # ========== 5. 格式转换 ==========
    if verbose:
        print(f"\n5. 格式转换 ({mode}-turn ChatML)...")
    if mode == "single":
        chatml_data = convert_to_chatml_single_turn(augmented_data)
    else:
        chatml_data = convert_to_chatml_multi_turn(augmented_data)
    print(f"   转换完成: {len(chatml_data)} 条对话")

    # ========== 6. 数据集划分 ==========
    if verbose:
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

    # ========== 7. 保存数据 ==========
    if verbose:
        print("\n7. 保存数据...")
    write_jsonl(train_data, os.path.join(output_dir, "train.jsonl"))
    write_jsonl(valid_data, os.path.join(output_dir, "valid.jsonl"))
    write_jsonl(test_data, os.path.join(output_dir, "test.jsonl"))
    print(f"   ✓ train.jsonl ({len(train_data)} 条)")
    print(f"   ✓ valid.jsonl ({len(valid_data)} 条)")
    print(f"   ✓ test.jsonl ({len(test_data)} 条)")

    # ========== 8. 数据示例 ==========
    if verbose and chatml_data:
        print("\n8. 数据示例 (ChatML 格式):")
        print("-" * 40)
        sample = chatml_data[0]
        print(json.dumps(sample, ensure_ascii=False, indent=2)[:500] + "...")

    # ========== 9. 完成 ==========
    print("\n" + "=" * 60)
    print("✓ 数据处理完成！")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  - {output_dir}/train.jsonl")
    print(f"  - {output_dir}/valid.jsonl")
    print(f"  - {output_dir}/test.jsonl")


# ==================== 主函数 ====================

def main():
    """
    主函数 - 解析命令行参数并执行数据处理

    支持的命令行参数：
        --input: 输入 CSV 文件
        --output-dir: 输出目录
        --mode: 转换模式（single/multi）
        --explore-only: 仅进行数据探查
        --augment: 启用 AI 数据增强
        --augment-ratio: 增强比例
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="数据处理脚本 - 转换为 ChatML 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
    # 多轮对话模式（推荐）
    python scripts/process_data.py --mode multi

    # 单轮对话模式
    python scripts/process_data.py --mode single

    # 仅探查数据
    python scripts/process_data.py --explore-only

    # 启用 AI 数据增强（使用 GLM API，推荐）
    python scripts/process_data.py --augment --augment-ratio 0.3 --api-type glm

    # 启用 AI 数据增强（使用 Qwen API）
    python scripts/process_data.py --augment --augment-ratio 0.3 --api-type qwen

    # 使用多种增强策略
    python scripts/process_data.py --augment --strategies paraphrase enhance

    # 使用全部增强策略
    python scripts/process_data.py --augment --strategies paraphrase enhance scenario
        """
    )

    # 定义命令行参数
    parser.add_argument(
        "--input",
        type=str,
        default=config.data.raw_data_path,
        help="输入 CSV 文件路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.data.processed_data_dir,
        help="输出目录"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "multi"],
        default="multi",
        help="转换模式: single(单轮), multi(多轮，默认)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=config.data.train_ratio,
        help="训练集比例"
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=config.data.valid_ratio,
        help="验证集比例"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=config.data.test_ratio,
        help="测试集比例"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.training.seed,
        help="随机种子"
    )
    parser.add_argument(
        "--explore-only",
        action="store_true",
        help="仅进行数据探查"
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="启用 AI 数据增强"
    )
    parser.add_argument(
        "--augment-ratio",
        type=float,
        default=config.augment.augment_ratio,
        help="增强比例（默认 0.3）"
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        choices=["paraphrase", "enhance", "scenario"],
        default=["paraphrase"],
        help="增强策略（可多选）: paraphrase(同义改写), enhance(回复增强), scenario(场景扩展)"
    )
    parser.add_argument(
        "--api-type",
        type=str,
        choices=["glm", "qwen"],
        default="glm",
        help="API 类型: glm(智谱AI，推荐) 或 qwen(阿里云)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API Key（未提供时从环境变量读取）"
    )
    parser.add_argument(
        "--api-model",
        type=str,
        default=None,
        help="API 模型名称（默认: glm-4.7 或 qwen-plus）"
    )

    # 解析参数
    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return 1

    # 执行数据处理
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
        augment_strategies=args.strategies,
        api_type=args.api_type,
        api_key=args.api_key,
        api_model=args.api_model
    )

    return 0


# ==================== 程序入口 ====================

if __name__ == "__main__":
    sys.exit(main())
