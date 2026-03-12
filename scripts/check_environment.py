#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境验证脚本
================

用途：
    用于检查项目所需的所有依赖是否正确安装，以及硬件环境是否满足要求。
    在开始训练之前，必须先运行此脚本确保环境配置正确。

运行方法：
    python scripts/check_environment.py

检查内容：
    1. Python 版本（要求 >=3.10）
    2. GPU & CUDA 可用性
    3. 所有必需的依赖包
    4. 磁盘空间（建议 >=50GB）
    5. 网络连接（Hugging Face 访问）

作者：MentalChat 项目组
"""

import sys
import os
import platform
import subprocess
from typing import Dict, List, Tuple


# ==================== 颜色输出配置 ====================
class Colors:
    """
    终端颜色输出类

    使用 ANSI 转义码实现终端彩色输出，让检查结果更直观。
    - GREEN: 绿色，表示通过
    - RED: 红色，表示失败
    - YELLOW: 黄色，表示警告
    - BLUE: 蓝色，表示标题
    """
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(title: str):
    """
    打印格式化的标题

    Args:
        title: 要显示的标题文本
    """
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{title.center(60)}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}\n")


def print_status(name: str, status: bool, extra_info: str = ""):
    """
    打印检查项的状态

    Args:
        name: 检查项名称
        status: True 表示通过，False 表示失败
        extra_info: 附加信息（如版本号、具体数值等）
    """
    status_str = f"{Colors.GREEN}✓ 通过{Colors.END}" if status else f"{Colors.RED}✗ 失败{Colors.END}"
    print(f"  {name}: {status_str} {extra_info}")


def print_warning(msg: str):
    """
    打印警告信息

    Args:
        msg: 警告内容
    """
    print(f"{Colors.YELLOW}  ⚠ {msg}{Colors.END}")


def print_info(msg: str):
    """
    打印普通信息

    Args:
        msg: 信息内容
    """
    print(f"    {msg}")


def check_python_version() -> Tuple[bool, str]:
    """
    检查 Python 版本是否满足要求

    大模型微调需要 Python 3.10+ 以支持最新的类型注解和特性。

    Returns:
        Tuple[bool, str]: (是否通过, 版本字符串)
    """
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    is_ok = version.major == 3 and version.minor >= 10
    return is_ok, version_str


def check_cuda() -> Tuple[bool, str]:
    """
    检查 CUDA 是否可用

    通过运行 nvidia-smi 命令来检测 GPU 和 CUDA 驱动。

    Returns:
        Tuple[bool, str]: (是否通过, GPU 信息)
    """
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            # 解析 GPU 信息，提取常见 GPU 型号
            lines = result.stdout.split('\n')
            gpu_info = ""
            for line in lines:
                if 'RTX' in line or 'A100' in line or 'V100' in line:
                    gpu_info = line.strip()
                    break
            return True, gpu_info if gpu_info else "GPU detected"
        return False, "nvidia-smi not found"
    except Exception as e:
        return False, str(e)


def check_torch_cuda() -> Tuple[bool, str, Dict]:
    """
    检查 PyTorch CUDA 支持

    验证 PyTorch 是否能正确识别和使用 GPU。

    Returns:
        Tuple[bool, str, Dict]: (是否通过, 状态信息, 详细信息字典)
    """
    try:
        import torch
        cuda_available = torch.cuda.is_available()

        info = {
            "torch_version": torch.__version__,
            "cuda_available": cuda_available,
        }

        if cuda_available:
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "N/A"
            info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB" if torch.cuda.device_count() > 0 else "N/A"

        return cuda_available, "CUDA可用" if cuda_available else "CUDA不可用", info
    except ImportError:
        return False, "PyTorch未安装", {}
    except Exception as e:
        return False, str(e), {}


def check_packages() -> Dict[str, Tuple[bool, str]]:
    """
    检查所有必需的依赖包

    验证核心依赖是否安装且版本符合要求。

    Returns:
        Dict[str, Tuple[bool, str]]: {包名: (是否通过, 版本信息)}
    """
    # 核心依赖及其最低版本要求
    packages = {
        # 核心依赖 - 大模型微调必备
        "torch": "2.0.0",           # PyTorch 深度学习框架
        "transformers": "4.36.0",   # Hugging Face 模型库
        "peft": "0.7.0",            # 轻量化微调库（LoRA/QLoRA）
        "accelerate": "0.25.0",     # 分布式训练支持
        "datasets": "2.15.0",       # 数据处理
        "bitsandbytes": "0.41.0",   # 4-bit 量化库
        # 数据处理
        "pandas": "2.0.0",          # 数据清洗
        "numpy": "1.24.0",          # 数值计算
        # API & UI
        "fastapi": "0.104.0",       # API 服务
        "gradio": "4.0.0",          # 可视化界面
        # 其他
        "tqdm": "4.65.0",           # 进度条
        "safetensors": "0.4.0",     # 安全模型保存格式
    }

    results = {}
    for package, min_version in packages.items():
        try:
            # 特殊处理 bitsandbytes
            if package == "bitsandbytes":
                import bitsandbytes as bnb
                version = bnb.__version__
            else:
                module = __import__(package)
                version = getattr(module, "__version__", "unknown")

            # 版本检查
            is_ok = True
            if version != "unknown":
                try:
                    from packaging import version as pkg_version
                    is_ok = pkg_version.parse(version) >= pkg_version.parse(min_version)
                except:
                    pass

            results[package] = (is_ok, f"v{version}")
        except ImportError:
            results[package] = (False, "未安装")

    return results


def check_disk_space() -> Tuple[bool, str]:
    """
    检查磁盘空间

    模型微调需要足够的磁盘空间存储：
    - 基座模型权重（约 15GB）
    - LoRA 权重（约 100MB）
    - 训练 checkpoint（每个约 5GB）
    - 数据集（约 100MB）

    Returns:
        Tuple[bool, str]: (是否通过, 空间信息)
    """
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free / (1024**3)
        is_ok = free_gb >= 50  # 至少50GB
        return is_ok, f"可用空间: {free_gb:.1f} GB"
    except Exception as e:
        return False, str(e)


def run_gpu_test() -> Tuple[bool, str]:
    """
    运行简单的 GPU 计算测试

    通过矩阵乘法测试 GPU 是否能正常进行计算。
    这是验证 CUDA 环境正确配置的关键步骤。

    Returns:
        Tuple[bool, str]: (是否通过, 测试结果信息)
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "CUDA不可用"

        # 简单的矩阵乘法测试
        device = torch.device("cuda:0")
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        c = torch.mm(a, b)
        torch.cuda.synchronize()  # 等待计算完成

        return True, "GPU计算测试通过"
    except Exception as e:
        return False, str(e)


def check_huggingface_access() -> Tuple[bool, str]:
    """
    检查 Hugging Face 访问

    验证是否能访问 Hugging Face 下载模型。
    国内网络环境可能需要配置镜像。

    Returns:
        Tuple[bool, str]: (是否通过, 访问状态)
    """
    try:
        import requests
        response = requests.get("https://huggingface.co", timeout=10)
        if response.status_code == 200:
            return True, "可访问"
        return False, f"状态码: {response.status_code}"
    except Exception as e:
        return False, str(e)

def main():
    """
    主函数 - 执行所有环境检查

    检查流程：
    1. 系统信息显示
    2. Python 版本检查
    3. GPU & CUDA 检查
    4. 依赖包检查
    5. 磁盘空间检查
    6. 网络连接检查
    7. 结果总结
    """
    print_header("心理咨询客服对话模型微调 - 环境检查")

    all_passed = True

    # ========== 1. 系统信息 ==========
    print_header("1. 系统信息")
    print_info(f"操作系统: {platform.system()} {platform.release()}")
    print_info(f"架构: {platform.machine()}")
    print_info(f"Python 路径: {sys.executable}")

    # ========== 2. Python 版本检查 ==========
    print_header("2. Python 版本检查")
    py_ok, py_version = check_python_version()
    print_status("Python 版本", py_ok, f"(当前: {py_version}, 要求: >=3.10)")
    if not py_ok:
        all_passed = False
        print_warning("请升级 Python 到 3.10 或更高版本")

    # ========== 3. CUDA/GPU 检查 ==========
    print_header("3. GPU & CUDA 检查")
    cuda_ok, cuda_info = check_cuda()
    print_status("nvidia-smi", cuda_ok, f"({cuda_info})")

    torch_cuda_ok, torch_cuda_msg, torch_info = check_torch_cuda()
    print_status("PyTorch CUDA", torch_cuda_ok, f"({torch_cuda_msg})")

    if torch_info:
        print_info(f"PyTorch 版本: {torch_info.get('torch_version', 'N/A')}")
        if torch_info.get('cuda_available'):
            print_info(f"CUDA 版本: {torch_info.get('cuda_version', 'N/A')}")
            print_info(f"GPU 数量: {torch_info.get('gpu_count', 'N/A')}")
            print_info(f"GPU 型号: {torch_info.get('gpu_name', 'N/A')}")
            print_info(f"GPU 显存: {torch_info.get('gpu_memory', 'N/A')}")

    if torch_cuda_ok:
        gpu_ok, gpu_msg = run_gpu_test()
        print_status("GPU 计算测试", gpu_ok, f"({gpu_msg})")
        if not gpu_ok:
            all_passed = False

    # ========== 4. 依赖包检查 ==========
    print_header("4. 依赖包检查")
    packages = check_packages()
    for pkg, (is_ok, version) in packages.items():
        print_status(pkg, is_ok, f"({version})")
        if not is_ok:
            all_passed = False

    # ========== 5. 磁盘空间检查 ==========
    print_header("5. 磁盘空间检查")
    disk_ok, disk_info = check_disk_space()
    print_status("磁盘空间", disk_ok, f"({disk_info})")
    if not disk_ok:
        print_warning("建议至少保留 50GB 可用空间")

    # ========== 6. 网络检查 ==========
    print_header("6. 网络连接检查")
    hf_ok, hf_msg = check_huggingface_access()
    print_status("Hugging Face", hf_ok, f"({hf_msg})")
    if not hf_ok:
        print_warning("无法访问 Hugging Face，可能需要配置镜像")

    # ========== 7. 结果总结 ==========
    print_header("检查结果总结")
    if all_passed and torch_cuda_ok:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ 所有检查通过！环境已准备就绪。{Colors.END}")
        print_info("下一步: 运行 python scripts/download_model.py 下载模型")
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠ 部分检查未通过，请根据上述提示进行修复。{Colors.END}")
        if not torch_cuda_ok:
            print_warning("GPU/CUDA 未正确配置，训练可能无法进行")

    print()
    return all_passed and torch_cuda_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
