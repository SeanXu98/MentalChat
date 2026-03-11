#!/bin/bash
# ============================================================
# AutoDL 环境快速配置脚本
# 在 AutoDL 实例上运行此脚本
# ============================================================

set -e

echo "============================================================"
echo "AutoDL 环境配置 - 心理咨询客服对话模型微调项目"
echo "============================================================"

# 项目目录
PROJECT_DIR="/root/autodl-tmp/MentalChat"

# 1. 创建虚拟环境
echo ""
echo "[1/5] 创建 Python 虚拟环境..."
if [ ! -d "/root/miniconda3/envs/mental_chat" ]; then
    # 检查 conda 是否存在
    if command -v conda &> /dev/null; then
        conda create -n mental_chat python=3.10 -y
        echo "  ✓ Conda 环境创建成功"
    else
        # 使用 venv
        python3 -m venv /root/autodl-tmp/mental_chat_env
        echo "  ✓ venv 环境创建成功"
    fi
else
    echo "  ✓ 虚拟环境已存在"
fi

# 2. 激活环境
echo ""
echo "[2/5] 激活虚拟环境..."
if command -v conda &> /dev/null; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate mental_chat
    PYTHON_BIN="/root/miniconda3/envs/mental_chat/bin/python"
else
    source /root/autodl-tmp/mental_chat_env/bin/activate
    PYTHON_BIN="/root/autodl-tmp/mental_chat_env/bin/python"
fi
echo "  Python: $PYTHON_BIN"
echo "  版本: $($PYTHON_BIN --version)"

# 3. 安装 PyTorch（CUDA 12.1）
echo ""
echo "[3/5] 安装 PyTorch (CUDA 12.1)..."
$PYTHON_BIN -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
echo "  ✓ PyTorch 安装完成"

# 4. 克隆项目（如果不存在）
echo ""
echo "[4/5] 准备项目文件..."
if [ ! -d "$PROJECT_DIR" ]; then
    mkdir -p $PROJECT_DIR
    echo "  ✓ 项目目录创建成功: $PROJECT_DIR"
else
    echo "  ✓ 项目目录已存在"
fi

# 5. 创建 requirements.txt
echo ""
echo "[5/5] 创建依赖文件..."
cat > $PROJECT_DIR/requirements.txt << 'EOF'
# 核心深度学习框架
torch>=2.0.0
torchvision>=0.15.0

# Hugging Face 核心库
transformers>=4.36.0
peft>=0.7.0
accelerate>=0.25.0
datasets>=2.15.0
bitsandbytes>=0.41.0
safetensors>=0.4.0

# 数据处理
pandas>=2.0.0
numpy>=1.24.0

# API 服务
fastapi>=0.104.0
uvicorn>=0.24.0

# 可视化界面
gradio>=4.0.0

# 工具库
tqdm>=4.65.0
EOF
echo "  ✓ requirements.txt 创建成功"

# 安装依赖
echo ""
echo "============================================================"
echo "安装项目依赖..."
echo "============================================================"
$PYTHON_BIN -m pip install -r $PROJECT_DIR/requirements.txt -q
echo "  ✓ 依赖安装完成"

# 验证安装
echo ""
echo "============================================================"
echo "验证环境..."
echo "============================================================"
echo "PyTorch 版本: $($PYTHON_BIN -c 'import torch; print(torch.__version__)')"
echo "CUDA 可用: $($PYTHON_BIN -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU 设备: $($PYTHON_BIN -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "GPU 显存: $($PYTHON_BIN -c 'import torch; print(f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB" if torch.cuda.is_available() else "N/A")')"

echo ""
echo "============================================================"
echo "✓ 环境配置完成！"
echo "============================================================"
echo ""
echo "PyCharm 连接信息:"
echo "  - 解释器路径: $PYTHON_BIN"
echo "  - 项目路径: $PROJECT_DIR"
echo ""
echo "下一步:"
echo "  1. 将本地项目文件上传到 $PROJECT_DIR"
echo "  2. 下载模型: python scripts/download_model.py --mirror"
echo "  3. 验证 QLoRA: python scripts/verify_qlora.py"
