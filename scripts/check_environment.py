#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境验证脚本
用于检查项目所需的所有依赖是否正确安装，以及硬件环境是否满足要求
"""

import sys
import os
import platform
import subprocess
from typing import Dict, List, Tuple

# 颜色输出
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(title: str):
    """打印标题"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{title.center(60)}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.END}\n")

def print_status(name: str, status: bool, extra_info: str = ""):
    """打印状态"""
    status_str = f"{Colors.GREEN}✓ 通过{Colors.END}" if status else f"{Colors.RED}✗ 失败{Colors.END}"
    print(f"  {name}: {status_str} {extra_info}")

def print_warning(msg: str):
    """打印警告"""
    print(f"{Colors.YELLOW}  ⚠ {msg}{Colors.END}")

def print_info(msg: str):
    """打印信息"""
    print(f"    {msg}")

def check_python_version() -> Tuple[bool, str]:
    """检查 Python 版本"""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    is_ok = version.major == 3 and version.minor >= 10
    return is_ok, version_str

def check_cuda() -> Tuple[bool, str]:
    """检查 CUDA 是否可用"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            # 解析 GPU 信息
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
    """检查 PyTorch CUDA 支持"""
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
    """检查所有必需的包"""
    packages = {
        # 核心依赖
        "torch": "2.0.0",
        "transformers": "4.36.0",
        "peft": "0.7.0",
        "accelerate": "0.25.0",
        "datasets": "2.15.0",
        "bitsandbytes": "0.41.0",
        # 数据处理
        "pandas": "2.0.0",
        "numpy": "1.24.0",
        # API & UI
        "fastapi": "0.104.0",
        "gradio": "4.0.0",
        # 其他
        "tqdm": "4.65.0",
        "safetensors": "0.4.0",
    }

    results = {}
    for package, min_version in packages.items():
        try:
            if package == "bitsandbytes":
                import bitsandbytes as bnb
                version = bnb.__version__
            else:
                module = __import__(package)
                version = getattr(module, "__version__", "unknown")

            # 简单的版本检查
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
    """检查磁盘空间"""
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free / (1024**3)
        is_ok = free_gb >= 50  # 至少50GB
        return is_ok, f"可用空间: {free_gb:.1f} GB"
    except Exception as e:
        return False, str(e)

def run_gpu_test() -> Tuple[bool, str]:
    """运行简单的GPU计算测试"""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "CUDA不可用"

        # 简单的矩阵乘法测试
        device = torch.device("cuda:0")
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        c = torch.mm(a, b)
        torch.cuda.synchronize()

        return True, "GPU计算测试通过"
    except Exception as e:
        return False, str(e)

def check_huggingface_access() -> Tuple[bool, str]:
    """检查 Hugging Face 访问"""
    try:
        import requests
        response = requests.get("https://huggingface.co", timeout=10)
        if response.status_code == 200:
            return True, "可访问"
        return False, f"状态码: {response.status_code}"
    except Exception as e:
        return False, str(e)

def main():
    """主函数"""
    print_header("心理咨询客服对话模型微调 - 环境检查")

    all_passed = True

    # 1. 系统信息
    print_header("1. 系统信息")
    print_info(f"操作系统: {platform.system()} {platform.release()}")
    print_info(f"架构: {platform.machine()}")
    print_info(f"Python 路径: {sys.executable}")

    # 2. Python 版本检查
    print_header("2. Python 版本检查")
    py_ok, py_version = check_python_version()
    print_status("Python 版本", py_ok, f"(当前: {py_version}, 要求: >=3.10)")
    if not py_ok:
        all_passed = False
        print_warning("请升级 Python 到 3.10 或更高版本")

    # 3. CUDA/GPU 检查
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

    # 4. 依赖包检查
    print_header("4. 依赖包检查")
    packages = check_packages()
    for pkg, (is_ok, version) in packages.items():
        print_status(pkg, is_ok, f"({version})")
        if not is_ok:
            all_passed = False

    # 5. 磁盘空间检查
    print_header("5. 磁盘空间检查")
    disk_ok, disk_info = check_disk_space()
    print_status("磁盘空间", disk_ok, f"({disk_info})")
    if not disk_ok:
        print_warning("建议至少保留 50GB 可用空间")

    # 6. 网络检查
    print_header("6. 网络连接检查")
    hf_ok, hf_msg = check_huggingface_access()
    print_status("Hugging Face", hf_ok, f"({hf_msg})")
    if not hf_ok:
        print_warning("无法访问 Hugging Face，可能需要配置镜像")

    # 总结
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
