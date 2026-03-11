# 环境搭建指南

本文档指导完成心理咨询客服对话模型微调项目的环境搭建与验证。

## 1. 系统要求

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| 操作系统 | Linux (Ubuntu 20.04/22.04) | Ubuntu 22.04 |
| GPU | NVIDIA RTX 3090 (16GB) | NVIDIA RTX 4090 (24GB) |
| CUDA | 11.8+ | 12.x |
| Python | 3.10+ | 3.10.x |
| 磁盘空间 | 50GB | 100GB+ |
| 内存 | 16GB | 32GB |

## 2. 快速开始

### 2.1 创建虚拟环境
```bash
# 使用 conda 创建虚拟环境
conda create -n mental_chat python=3.10 -y
conda activate mental_chat

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
```

### 2.2 安装 PyTorch
```bash
# CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU 版本（仅用于测试，不推荐用于训练）
pip install torch torchvision
```

### 2.3 安装其他依赖
```bash
pip install -r requirements.txt
```

### 2.4 验证环境
```bash
python scripts/check_environment.py
```

### 2.5 下载模型
```bash
# 从 HuggingFace 下载（需要科学上网）
python scripts/download_model.py

# 使用国内镜像
python scripts/download_model.py --mirror
```

### 2.6 验证 QLoRA 配置
```bash
python scripts/verify_qlora.py
```

## 3. 详细步骤

### 3.1 CUDA 安装验证
```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 CUDA 版本
nvcc --version
```

### 3.2 依赖说明
| 包名 | 用途 | 必要说明 |
|------|------|--------|
| torch | 深度学习框架 | 必须与CUDA版本匹配 |
| transformers | 模型加载与训练 | HuggingFace核心库 |
| peft | 轻量化微调 | LoRA/QLoRA实现 |
| accelerate | 分布式训练 | 自动设备分配 |
| bitsandbytes | 量化支持 | 4-bit量化必需 |
| datasets | 数据处理 | HuggingFace数据集 |
| pandas | 数据清洗 | CSV处理 |
| gradio | 可视化界面 | Demo演示 |
| fastapi | API服务 | 部署服务 |

### 3.3 Hugging Face 镜像配置（国内用户）
```bash
# 临时使用
export HF_ENDPOINT=https://hf-mirror.com

# 或在代码中设置
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

### 3.4 模型下载
模型约 14GB，下载时间取决于网络速度。

```bash
# 完整命令
python scripts/download_model.py \
    --model Qwen/Qwen2-7B-Instruct \
    --save-dir ./models/base \
    --mirror  # 使用国内镜像
```

## 4. 常见问题

### 4.1 CUDA out of memory
- 减小 batch_size
- 增加 gradient_accumulation_steps
- 检查是否有其他程序占用显存

### 4.2 bitsandbytes 安装失败
```bash
# Linux
pip install bitsandbytes

# 如果失败，尝试从源码安装
pip install git+https://github.com/TimDettmers/bitsandbytes.git
```

### 4.3 模型下载中断
- 重新运行下载脚本，会自动断点续传
- 或手动从镜像站下载

### 4.4 依赖冲突
```bash
# 清理环境重新安装
pip uninstall torch transformers peft accelerate bitsandbytes -y
pip install -r requirements.txt
```

## 5. 验证清单
完成以下检查后，环境搭建完成：
- [ ] Python 3.10+ 已安装
- [ ] PyTorch 2.0+ 已安装，CUDA 可用
- [ ] 所有依赖包已安装
- [ ] GPU 显存 >= 16GB
- [ ] 磁盘空间 >= 50GB
- [ ] Qwen2-7B-Instruct 模型已下载
- [ ] QLoRA 4-bit 量化验证通过

## 6. 下一步
环境搭建完成后，继续执行：
1. 数据处理（阶段二）
2. 模型训练（阶段三）
3. 模型评估（阶段四）
4. 部署演示（阶段五）

## 7. 项目目录结构
```
MentalChat/
├── configs/              # 配置文件
│   └── config.py
├── data/                 # 数据目录
│   ├── raw/              # 原始数据
│   │   └── data.csv
│   └── processed/        # 处理后数据
│       ├── train.json
│       ├── valid.json
│       └── test.json
├── models/               # 模型目录
│   ├── base/             # 基座模型
│   ├── lora/             # LoRA 权重
│   └── merged/           # 合并后模型
├── outputs/              # 输出目录
│   ├── checkpoints/      # 训练检查点
│   ├── logs/             # 训练日志
│   └── results/          # 评估结果
├── scripts/              # 脚本文件
│   ├── check_environment.py
│   ├── download_model.py
│   └── verify_qlora.py
├── docs/                 # 文档目录
│   └── setup.md
├── requirements.txt      # 依赖清单
└── README.md
```
