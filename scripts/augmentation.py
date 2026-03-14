#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据增强模块
============

本模块提供心理咨询对话数据的 AI 辅助增强功能，使用阿里云 DashScope Qwen API 实现。

主要功能：
    - 同义改写（Paraphrase）：改写用户输入，生成多样化的训练样本
    - 回复增强（Enhance）：优化回复内容，使其更专业、更有共情
    - 回复清理（Clean）：修复语法错误，标准化格式
    - 场景扩展（Scenario Expansion）：将问题改编为不同场景

使用方法：
    from scripts.augmentation import DataAugmenter, QwenAPI

    # 初始化
    api = QwenAPI(api_key="your-api-key")
    augmenter = DataAugmenter(api)

    # 执行增强
    augmented = augmenter.augment(data, strategies=["paraphrase", "enhance"])

API 密钥获取：
    1. 访问阿里云 DashScope 控制台
    2. 创建 API Key
    3. 设置环境变量: export DASHSCOPE_API_KEY='your-api-key'

作者：MentalChat 项目组
"""

# ==================== 标准库导入 ====================
import os
import json
import time
import random
import re
import sys
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path

# ==================== 第三方库导入 ====================
# 尝试导入 requests，如果失败则使用 urllib
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    import urllib.request
    import urllib.error
    HAS_REQUESTS = False

# ==================== 项目内部导入 ====================
# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import config


# ==================== 增强配置类 ====================

@dataclass
class AugmentationConfig:
    """
    数据增强配置类

    使用 dataclass 定义，便于管理和序列化配置参数。

    属性：
        enabled: 是否启用数据增强
        augment_ratio: 增强比例（0-1），表示对多少比例的数据进行增强
        strategies: 增强策略列表，可选值：paraphrase, enhance, scenario, clean
        api_key: DashScope API 密钥
        api_model: 使用的 Qwen 模型名称
        api_base_url: API 基础 URL
        scenario_types: 场景扩展时使用的场景类型列表
        max_retries: API 调用最大重试次数
        retry_delay: 重试间隔（秒）
        request_timeout: 请求超时时间（秒）
    """
    # 是否启用增强
    enabled: bool = False

    # 增强比例 (0-1)，例如 0.3 表示对 30% 的数据进行增强
    augment_ratio: float = 0.3

    # 增强策略列表
    strategies: List[str] = field(default_factory=lambda: ["paraphrase", "enhance"])

    # Qwen API 配置
    api_key: Optional[str] = None
    api_model: str = "qwen-plus"  # 可选: qwen-turbo, qwen-plus, qwen-max
    api_base_url: str = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

    # 增强场景类型（用于场景扩展策略）
    scenario_types: List[str] = field(default_factory=lambda: ["青少年", "职场", "家庭", "学业"])

    # 请求配置
    max_retries: int = 3           # 最大重试次数
    retry_delay: float = 1.0       # 重试间隔（秒）
    request_timeout: int = 30      # 请求超时（秒）


# ==================== Qwen API 封装类 ====================

class QwenAPI:
    """
    阿里云 DashScope Qwen API 封装类

    该类封装了与阿里云 DashScope API 的交互逻辑，用于调用 Qwen 系列模型进行数据增强。

    属性：
        api_key (str): API 密钥
        model (str): 模型名称
        base_url (str): API 基础 URL

    使用示例：
        >>> api = QwenAPI(api_key="your-api-key")
        >>> if api.is_available():
        ...     result = api.call("你好", temperature=0.7)
        ...     print(result)

    注意：
        API 密钥可以从环境变量 DASHSCOPE_API_KEY 获取，也可以在初始化时传入。
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "qwen-plus",
        base_url: str = None
    ):
        """
        初始化 API 客户端

        Args:
            api_key: DashScope API 密钥（可选，默认从环境变量 DASHSCOPE_API_KEY 获取）
            model: Qwen 模型名称，可选值：qwen-turbo, qwen-plus, qwen-max
            base_url: API 基础 URL（可选，使用默认值即可）
        """
        # 获取 API 密钥（优先使用传入的密钥，其次使用环境变量）
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.model = model
        self.base_url = base_url or "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

        # 检查 API 密钥是否可用
        if not self.api_key:
            print("警告: 未设置 API 密钥，AI 增强功能将被禁用")
            print("请设置环境变量 DASHSCOPE_API_KEY 或在初始化时传入 api_key")

    def is_available(self) -> bool:
        """
        检查 API 是否可用

        Returns:
            bool: 如果 API 密钥已设置则返回 True，否则返回 False
        """
        return self.api_key is not None

    def call(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> Optional[str]:
        """
        调用 Qwen API 生成文本

        Args:
            prompt: 用户提示文本
            system_prompt: 系统提示文本（可选，用于设定模型角色）
            temperature: 温度参数，控制输出的随机性（0-2），值越大越随机
            max_tokens: 最大生成 token 数

        Returns:
            生成的文本内容，如果调用失败则返回 None

        调用流程：
        1. 构建请求头和请求体
        2. 发送 POST 请求到 DashScope API
        3. 解析响应并返回结果
        4. 如果失败则自动重试（最多 3 次）
        """
        if not self.is_available():
            return None

        # ========== 构建请求头 ==========
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # ========== 构建消息列表 ==========
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # ========== 构建请求体 ==========
        payload = {
            "model": self.model,
            "input": {
                "messages": messages
            },
            "parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "result_format": "message"  # 使用消息格式返回结果
            }
        }

        # ========== 发送请求（带重试机制）==========
        for attempt in range(3):  # 最多重试 3 次
            try:
                if HAS_REQUESTS:
                    # 使用 requests 库发送请求
                    response = requests.post(
                        self.base_url,
                        headers=headers,
                        json=payload,
                        timeout=30
                    )
                    response.raise_for_status()  # 检查 HTTP 错误
                    result = response.json()
                else:
                    # 使用 urllib 发送请求（requests 不可用时的备选方案）
                    req = urllib.request.Request(
                        self.base_url,
                        data=json.dumps(payload).encode('utf-8'),
                        headers=headers,
                        method='POST'
                    )
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        result = json.loads(resp.read().decode('utf-8'))

                # ========== 解析响应 ==========
                # DashScope API 有两种响应格式
                if "output" in result and "choices" in result["output"]:
                    # 新版消息格式
                    return result["output"]["choices"][0]["message"]["content"]
                elif "output" in result and "text" in result["output"]:
                    # 旧版文本格式
                    return result["output"]["text"]
                else:
                    print(f"API 响应格式异常: {result}")
                    return None

            except Exception as e:
                print(f"API 调用失败 (尝试 {attempt + 1}/3): {e}")
                time.sleep(1.0 * (attempt + 1))  # 指数退避

        return None


# ==================== 增强策略函数 ====================

def clean_response(response: str) -> str:
    """
    清理和标准化回复文本

    该函数用于清理原始回复中的格式问题，使输出更加规范。

    Args:
        response: 原始回复文本

    Returns:
        清理后的回复文本

    处理内容：
        1. 移除多余的空白字符
        2. 修复重复的标点符号
        3. 确保句末有标点
    """
    # 移除多余的空白字符（保留单个空格）
    response = re.sub(r'\s+', ' ', response).strip()

    # 修复常见的标点符号问题（移除重复标点）
    response = re.sub(r'[,，]{2,}', '，', response)
    response = re.sub(r'[。。]{2,}', '。', response)
    response = re.sub(r'[?？]{2,}', '？', response)
    response = re.sub(r'[!！]{2,}', '！', response)

    # 确保句末有标点符号
    if response and response[-1] not in '。！？，、；：…』」")':
        response += '。'

    return response


def paraphrase_input(qwen_api: QwenAPI, text: str) -> Optional[str]:
    """
    同义改写用户输入

    使用 Qwen 模型对用户输入进行同义改写，生成语义相同但表达不同的新样本。
    这有助于增加训练数据的多样性，提高模型的泛化能力。

    Args:
        qwen_api: Qwen API 实例
        text: 原始用户输入文本

    Returns:
        改写后的输入文本，如果失败则返回 None

    改写要求：
        1. 保持问题的核心含义不变
        2. 使用不同的词汇和句式
        3. 保持口语化、自然的表达
    """
    # 构建提示词
    prompt = f"""请将以下心理咨询场景中的用户输入进行同义改写，保持原意但使用不同的表达方式。
要求：
1. 保持问题的核心含义不变
2. 使用不同的词汇和句式
3. 保持口语化、自然的表达
4. 只输出改写后的内容，不要有任何解释

原始输入：{text}

改写后："""

    # 设定系统角色
    system_prompt = "你是一个专业的心理咨询数据增强助手，擅长进行同义改写。"

    # 调用 API（使用较高的温度以增加多样性）
    result = qwen_api.call(prompt, system_prompt=system_prompt, temperature=0.8)
    return result.strip() if result else None


def enhance_response(qwen_api: QwenAPI, response: str, context: str = "") -> Optional[str]:
    """
    增强回复内容

    使用 Qwen 模型优化心理咨询回复，使其更加专业、温暖、有共情力。
    这有助于提高训练数据的质量。

    Args:
        qwen_api: Qwen API 实例
        response: 原始回复文本
        context: 上下文信息（用户输入），可选

    Returns:
        增强后的回复文本，如果失败则返回 None

    增强要求：
        1. 增加对来访者情绪的理解和接纳
        2. 提供更具体、可行的建议
        3. 使用温暖、支持性的语言
        4. 保持专业边界
    """
    # 构建提示词
    prompt = f"""请优化以下心理咨询回复，使其更加专业、温暖、有共情力。

上下文：{context if context else '无'}

原始回复：{response}

优化要求：
1. 增加对来访者情绪的理解和接纳
2. 提供更具体、可行的建议
3. 使用温暖、支持性的语言
4. 保持专业边界
5. 只输出优化后的回复，不要有任何解释

优化后的回复："""

    # 设定系统角色
    system_prompt = "你是一个专业的心理咨询师，擅长提供温暖、专业的心理支持。"

    # 调用 API
    result = qwen_api.call(prompt, system_prompt=system_prompt, temperature=0.7)

    # 清理并返回结果
    return clean_response(result) if result else None


def scenario_expansion(
    qwen_api: QwenAPI,
    input_text: str,
    response: str,
    scenario: str
) -> Optional[Dict[str, str]]:
    """
    场景扩展：将对话改编到不同场景

    将原始对话改编到指定的场景（如青少年、职场、家庭等），
    以增加训练数据在不同场景下的覆盖度。

    Args:
        qwen_api: Qwen API 实例
        input_text: 原始用户输入
        response: 原始咨询师回复
        scenario: 目标场景名称

    Returns:
        包含新输入和回复的字典，格式为 {"input": "...", "response": "..."}
        如果失败则返回 None

    改编要求：
        1. 保持心理问题的核心不变
        2. 将背景调整为目标场景
        3. 调整用词以符合该场景的特点
    """
    # 构建提示词
    prompt = f"""请将以下心理咨询对话改编为"{scenario}"场景。

原始对话：
用户：{input_text}
咨询师：{response}

要求：
1. 保持心理问题的核心不变
2. 将背景调整为{scenario}场景
3. 调整用词以符合该场景的特点
4. 咨询师的回复也应相应调整
5. 输出格式为 JSON: {{"input": "改编后的用户输入", "response": "改编后的咨询师回复"}}

改编结果："""

    # 设定系统角色
    system_prompt = "你是一个专业的心理咨询数据增强助手，擅长场景改编。"

    # 调用 API
    result = qwen_api.call(prompt, system_prompt=system_prompt, temperature=0.8)

    # 解析 JSON 响应
    if result:
        try:
            # 移除可能的 markdown 代码块标记
            result = re.sub(r'```json\s*', '', result)
            result = re.sub(r'```\s*', '', result)

            # 解析 JSON
            data = json.loads(result)

            if "input" in data and "response" in data:
                return {
                    "input": data["input"],
                    "response": clean_response(data["response"])
                }
        except json.JSONDecodeError:
            pass

    return None


# ==================== 数据增强器类 ====================

class DataAugmenter:
    """
    数据增强器类

    该类提供多种增强策略来扩充心理咨询对话训练数据。

    属性：
        api (QwenAPI): Qwen API 实例
        augment_ratio (float): 增强比例
        strategies (List[str]): 增强策略列表
        scenario_types (List[str]): 场景类型列表
        stats (Dict): 增强统计信息

    支持的增强策略：
        - paraphrase: 同义改写用户输入
        - enhance: 增强回复内容
        - scenario: 场景扩展
        - clean: 清理回复格式

    使用示例：
        >>> augmenter = DataAugmenter(api_key="your-api-key")
        >>> augmented_data = augmenter.augment(original_data)
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "qwen-plus",
        augment_ratio: float = 0.3,
        strategies: List[str] = None,
        scenario_types: List[str] = None
    ):
        """
        初始化数据增强器

        Args:
            api_key: DashScope API 密钥（可选，默认从环境变量获取）
            model: Qwen 模型名称
            augment_ratio: 增强比例（0-1），表示对多少比例的数据进行增强
            strategies: 增强策略列表，可选值：paraphrase, enhance, scenario, clean
            scenario_types: 场景类型列表（用于场景扩展策略）
        """
        # 初始化 API 客户端
        self.api = QwenAPI(api_key=api_key, model=model)

        # 增强配置
        self.augment_ratio = augment_ratio
        self.strategies = strategies or ["paraphrase", "enhance"]
        self.scenario_types = scenario_types or ["青少年", "职场", "家庭", "学业"]

        # 统计信息（用于追踪增强效果）
        self.stats = {
            "total_processed": 0,    # 总处理数
            "paraphrase_success": 0, # 同义改写成功数
            "enhance_success": 0,    # 回复增强成功数
            "scenario_success": 0,   # 场景扩展成功数
            "errors": 0              # 错误数
        }

    def augment_single(
        self,
        input_text: str,
        response: str,
        strategies: List[str] = None
    ) -> List[Dict[str, str]]:
        """
        增强单条数据

        对单条对话数据应用指定的增强策略。

        Args:
            input_text: 用户输入文本
            response: 咨询师回复文本
            strategies: 使用的策略列表（None 则使用默认策略）

        Returns:
            增强后的数据列表，每条数据包含 input, response, augment_strategy 字段
        """
        strategies = strategies or self.strategies
        augmented = []

        for strategy in strategies:
            try:
                if strategy == "paraphrase":
                    # ========== 同义改写策略 ==========
                    new_input = paraphrase_input(self.api, input_text)
                    if new_input:
                        augmented.append({
                            "input": new_input,
                            "response": response,
                            "augment_strategy": "paraphrase"
                        })
                        self.stats["paraphrase_success"] += 1

                elif strategy == "enhance":
                    # ========== 回复增强策略 ==========
                    new_response = enhance_response(self.api, response, input_text)
                    if new_response:
                        augmented.append({
                            "input": input_text,
                            "response": new_response,
                            "augment_strategy": "enhance"
                        })
                        self.stats["enhance_success"] += 1

                elif strategy == "scenario":
                    # ========== 场景扩展策略 ==========
                    scenario = random.choice(self.scenario_types)
                    new_data = scenario_expansion(self.api, input_text, response, scenario)
                    if new_data:
                        new_data["augment_strategy"] = f"scenario_{scenario}"
                        augmented.append(new_data)
                        self.stats["scenario_success"] += 1

                elif strategy == "clean":
                    # ========== 清理策略（不需要 API）==========
                    cleaned_response = clean_response(response)
                    if cleaned_response != response:
                        augmented.append({
                            "input": input_text,
                            "response": cleaned_response,
                            "augment_strategy": "clean"
                        })

            except Exception as e:
                print(f"增强策略 {strategy} 失败: {e}")
                self.stats["errors"] += 1

        return augmented

    def augment(
        self,
        data: List[Dict[str, Any]],
        strategies: List[str] = None,
        augment_ratio: float = None,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        批量增强数据

        对数据集应用增强策略，生成更多的训练样本。

        Args:
            data: 原始数据列表，每条数据应包含 input/Input 和 response/Output 字段
            strategies: 增强策略列表（None 则使用默认策略）
            augment_ratio: 增强比例（None 则使用默认比例）
            verbose: 是否输出详细的处理信息

        Returns:
            增强后的数据列表（包含原始数据和新增数据）
        """
        # 检查 API 是否可用
        if not self.api.is_available():
            print("API 不可用，跳过数据增强")
            return data

        # 使用传入的参数或默认值
        strategies = strategies or self.strategies
        augment_ratio = augment_ratio or self.augment_ratio

        # ========== 计算需要增强的数据量 ==========
        n_augment = int(len(data) * augment_ratio)

        # 随机选择要增强的数据索引
        indices_to_augment = random.sample(range(len(data)), min(n_augment, len(data)))

        # 复制原始数据作为结果的基础
        result = list(data)
        self.stats["total_processed"] = len(indices_to_augment)

        # 打印增强配置信息
        if verbose:
            print(f"开始数据增强...")
            print(f"  原始数据量: {len(data)}")
            print(f"  增强比例: {augment_ratio}")
            print(f"  增强策略: {strategies}")
            print(f"  预计增强数量: {n_augment}")

        # ========== 遍历并增强数据 ==========
        for i, idx in enumerate(indices_to_augment):
            item = data[idx]

            # 获取输入和回复（兼容不同的字段名）
            input_text = item.get("input", item.get("Input", ""))
            response = item.get("response", item.get("Output", ""))

            # 跳过无效数据
            if not input_text or not response:
                continue

            # 执行增强
            augmented = self.augment_single(input_text, response, strategies)
            result.extend(augmented)

            # 定期输出进度
            if verbose and (i + 1) % 50 == 0:
                print(f"  已处理: {i + 1}/{len(indices_to_augment)}")

        # ========== 打印增强结果 ==========
        if verbose:
            print(f"\n增强完成!")
            print(f"  最终数据量: {len(result)}")
            print(f"  新增数据量: {len(result) - len(data)}")
            print(f"  同义改写成功: {self.stats['paraphrase_success']}")
            print(f"  回复增强成功: {self.stats['enhance_success']}")
            print(f"  场景扩展成功: {self.stats['scenario_success']}")
            if self.stats['errors'] > 0:
                print(f"  错误数: {self.stats['errors']}")

        return result

    def get_stats(self) -> Dict[str, int]:
        """
        获取增强统计信息

        Returns:
            包含各项统计数据的字典
        """
        return self.stats.copy()


# ==================== 便捷函数 ====================

def create_augmenter(
    api_key: str = None,
    augment_ratio: float = 0.3,
    strategies: List[str] = None
) -> DataAugmenter:
    """
    创建数据增强器的便捷函数

    这是一个工厂函数，用于快速创建 DataAugmenter 实例。

    Args:
        api_key: DashScope API 密钥
        augment_ratio: 增强比例（0-1）
        strategies: 增强策略列表

    Returns:
        配置好的 DataAugmenter 实例

    使用示例：
        >>> augmenter = create_augmenter(augment_ratio=0.5)
        >>> augmented = augmenter.augment(data)
    """
    return DataAugmenter(
        api_key=api_key,
        augment_ratio=augment_ratio,
        strategies=strategies
    )


# ==================== 测试代码 ====================

if __name__ == "__main__":
    """
    数据增强模块测试入口

    运行方式：
        python scripts/augmentation.py

    测试内容：
        1. 清理功能测试（不需要 API）
        2. API 连接测试
        3. 同义改写测试
        4. 回复增强测试
    """
    print("=" * 60)
    print("数据增强模块测试")
    print("=" * 60)

    # ========== 测试清理功能 ==========
    test_response = "你好，，我觉得很焦虑。。。"
    cleaned = clean_response(test_response)
    print(f"\n清理测试:")
    print(f"  原始: {test_response}")
    print(f"  清理后: {cleaned}")

    # ========== 测试 API 连接 ==========
    api = QwenAPI()
    if api.is_available():
        print("\n✓ API 可用，测试增强功能...")

        # 测试同义改写
        test_input = "最近工作压力很大，经常失眠"
        print(f"\n同义改写测试:")
        print(f"  原始: {test_input}")
        paraphrased = paraphrase_input(api, test_input)
        print(f"  改写: {paraphrased}")

        # 测试回复增强
        test_response = "我理解你的感受，建议你尝试放松一下。"
        print(f"\n回复增强测试:")
        print(f"  原始: {test_response}")
        enhanced = enhance_response(api, test_response, test_input)
        print(f"  增强: {enhanced}")

    else:
        print("\n✗ API 不可用（未设置 DASHSCOPE_API_KEY）")
        print("  要启用 AI 增强功能，请设置环境变量:")
        print("  export DASHSCOPE_API_KEY='your-api-key'")

    print("\n" + "=" * 60)
    print("测试完成")
