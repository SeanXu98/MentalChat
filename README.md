# 心理咨询客服对话模型微调项目

> 本项目使用 QLoRA 技术对 Qwen2.5-7B-Instruct 进行轻量化微调，使其适配心理咨询客服场景。

---

## 目录

1. [项目背景与目标](#一项目背景与目标)
2. [微调背景知识](#二微调背景知识)
3. [技术栈与环境](#三技术栈与环境)
4. [项目结构](#四项目结构)
5. [快速开始](#五快速开始)
6. [详细执行步骤](#六详细执行步骤)
7. [常见问题](#七常见问题)
8. [面试要点](#八面试要点)

---

## 一、项目背景与目标

### 1.1 项目背景

通用大模型在心理咨询客服场景存在明显痛点：

| 痛点 | 具体表现 | 影响 |
|------|---------|------|
| 专业度不足 | 对抑郁、焦虑等问题的回复缺乏行业规范 | 可能给出不恰当建议 |
| 共情能力弱 | 无法贴合来访者情绪给出安抚性话术 | 用户体验差 |
| 格式不统一 | 输出无规律，无固定话术框架 | 无法对接客服流程 |
| 成本过高 | 全量微调显存占用大、周期长 | 中小企业难以落地 |

### 1.2 项目目标

| 维度 | 具体目标 |
|------|---------|
| 效果 | 回答专业度、共情贴合度 ≥90%；话术格式合规率 ≥95% |
| 工程 | 单卡 RTX 4090（24G）可稳定训练；推理延迟 ≤100ms |
| 落地 | 封装 API 接口；配套可视化 Demo |

### 1.3 核心产出

- **微调模型**：适配心理咨询场景的 Qwen2.5-7B 模型
- **LoRA 权重**：约 100MB，便于存储和部署
- **API 服务**：FastAPI 封装的推理接口
- **可视化 Demo**：Gradio 构建的演示界面

---

## 二、微调背景知识

### 2.1 为什么选择 LoRA/QLoRA？

**全量微调的痛点**：
- 显存占用高：7B 模型全量微调需 80G+ 显存
- 训练成本高：周期长（7天+），算力昂贵
- 易过拟合：小样本场景下破坏模型通用能力

**LoRA 的优势**：
```
原始权重矩阵 W (4096 × 4096)
         ↓ LoRA 分解
低秩矩阵 A (4096 × 16) + 低秩矩阵 B (16 × 4096)

可训练参数：16 × (4096 + 4096) = 131,072
原始参数：4096 × 4096 = 16,777,216
参数比例：约 0.78%（减少 99% 以上）
```

**QLoRA 的优化**：
- 4-bit 量化：显存从 20G 降至 11G
- NF4 格式：专为大模型设计，精度损失 <5%
- 双量化：进一步节省约 0.5GB 显存

### 2.2 模型精度与显存计算

#### 2.2.1 精度类型说明

| 精度类型 | 每参数字节数 | 数值范围 | 适用场景 |
|---------|------------|---------|---------|
| FP32 (全精度) | 4 bytes | ±3.4×10³⁸ | 科学计算、训练 |
| FP16 (半精度) | 2 bytes | ±65,504 | 训练、推理 |
| BF16 (脑浮点) | 2 bytes | 同 FP32 范围 | 训练（推荐） |
| INT8 (8位整型) | 1 byte | -128 ~ 127 | 量化推理 |
| INT4 / NF4 | 0.5 bytes | -8 ~ 7 | 极限量化 |

#### 2.2.2 显存占用计算公式

```
模型显存占用 ≈ 参数量 × 每参数字节数

以 Qwen2.5-7B 为例（实际参数量约 7.6B）：
┌────────────────────────────────────────────────────────┐
│  FP32: 7.6B × 4 bytes = 30.4 GB  (全精度，极少使用)      │
│  FP16: 7.6B × 2 bytes = 15.2 GB  (HuggingFace 默认)    │
│  BF16: 7.6B × 2 bytes = 15.2 GB  (训练推荐)             │
│  INT8: 7.6B × 1 byte  = 7.6 GB   (8-bit 量化)          │
│  INT4: 7.6B × 0.5 byte = 3.8 GB  (4-bit 量化)          │
└────────────────────────────────────────────────────────┘
```

#### 2.2.3 本项目基座模型说明

**下载的 Qwen2.5-7B-Instruct 是什么精度？**

> **FP16/BF16 混合精度**，这就是为什么模型文件约 15GB。

HuggingFace 上的模型权重通常以 FP16 或 BF16 格式存储（ safetensors 格式），这是业界标准做法：
- 精度足够：保持模型性能
- 体积适中：比 FP32 小一半
- 兼容性好：大多数框架直接支持

**训练时的显存占用**：

| 阶段 | 显存组成 | 7B 模型估算 |
|------|---------|-----------|
| 加载模型 | FP16 权重 | ~15 GB |
| + 梯度 | FP16 梯度 | +15 GB |
| + 优化器状态 | FP32 动量 | +30 GB |
| + 激活值 | 中间计算 | +5-10 GB |
| **全量微调总计** | | **~65-70 GB** |

**QLoRA 训练时**：

| 阶段 | 显存组成 | 7B 模型估算 |
|------|---------|-----------|
| 加载模型 | 4-bit 量化权重 | ~4 GB |
| + LoRA 权重 | FP16 低秩矩阵 | +0.1 GB |
| + 梯度 | 仅 LoRA 部分 | +0.1 GB |
| + 优化器状态 | 8-bit 优化器 | +0.5 GB |
| + 激活值 | 中间计算 | +3-5 GB |
| **QLoRA 微调总计** | | **~8-10 GB** |

### 2.3 核心概念详解

#### 2.2.1 Transformers 库

```
┌─────────────────────────────────────────────────────┐
│                  Transformers 库                     │
├─────────────────────────────────────────────────────┤
│  AutoTokenizer      文本分词器                        │
│  AutoModelForCausalLM   因果语言模型（文本生成）        │
│  TrainingArguments  训练参数配置                      │
│  Trainer            训练器封装                        │
│  BitsAndBytesConfig 量化配置                         │
└─────────────────────────────────────────────────────┘
```

**关键类说明**：

| 类名 | 作用 | 项目用途 |
|------|------|---------|
| AutoTokenizer | 将文本转换为 token | 处理输入输出文本 |
| AutoModelForCausalLM | 加载生成式模型 | 加载 Qwen2.5-7B |
| BitsAndBytesConfig | 配置量化参数 | 实现 4-bit 量化 |

#### 2.2.2 PEFT 库

```
┌─────────────────────────────────────────────────────┐
│                    PEFT 库                           │
├─────────────────────────────────────────────────────┤
│  LoraConfig         LoRA 参数配置                     │
│  get_peft_model     将 LoRA 应用到模型                │
│  prepare_model_for_kbit_training  准备量化训练        │
└─────────────────────────────────────────────────────┘
```

**LoRA 关键参数深度解析**：

#### ① r (秩 / Rank)

```
含义：LoRA 低秩分解的秩，决定了可训练参数的数量

原始权重 W: (4096, 4096) = 16,777,216 参数
LoRA 分解: W' = W + BA
  - 矩阵 A: (4096, r)    r=16 时 = 65,536 参数
  - 矩阵 B: (r, 4096)    r=16 时 = 65,536 参数
  - 总参数: 2 × 4096 × r = 131,072 参数 (r=16)

不同 r 值对比：
┌─────┬──────────────┬───────────┬────────────────┐
│  r  │  可训练参数   │  显存占用  │     效果       │
├─────┼──────────────┼───────────┼────────────────┤
│  8  │   65,536     │  更低     │ 表达能力有限   │
│ 16  │  131,072     │  适中     │ 推荐：平衡点   │
│ 32  │  262,144     │  较高     │ 收益递减       │
│ 64  │  524,288     │  高       │ 过拟合风险     │
└─────┴──────────────┴───────────┴────────────────┘

为什么选 r=16？
- 实验证明：r=16 在大多数任务上已接近全量微调效果
- r>32 后性能提升不明显，但显存和计算成本线性增长
- r<8 时表达能力不足，难以学到复杂的领域知识
```

#### ② lora_alpha (缩放因子)

```
含义：控制 LoRA 更新的缩放比例，影响学习强度

实际更新公式：ΔW = (lora_alpha / r) × B × A
              缩放系数 = alpha / r

示例计算：
- r=16, alpha=32 → 缩放系数 = 32/16 = 2.0
- r=16, alpha=16 → 缩放系数 = 16/16 = 1.0
- r=16, alpha=64 → 缩放系数 = 64/16 = 4.0

为什么 alpha 通常设为 2×r？
├── 经验法则：alpha=2r 是最稳定的设置
├── 缩放系数=2：适度的更新强度，不会过激
├── 配合 learning_rate=2e-4：形成合理的有效学习率
└── 太大会导致训练不稳定，太小则学习太慢

调节建议：
- 如果 loss 震荡 → 减小 alpha 或降低 learning_rate
- 如果 loss 下降太慢 → 增大 alpha 或提高 learning_rate
```

#### ③ target_modules (目标模块)

```
含义：指定要应用 LoRA 的模型层

Transformer 注意力结构：
                    ┌─────────────────┐
                    │   Attention     │
                    ├─────────────────┤
  Query (Q) ───────►│  q_proj         │◄── LoRA 重点
  Key (K) ─────────►│  k_proj         │
  Value (V) ───────►│  v_proj         │◄── LoRA 重点
  Output ─────────►│  o_proj         │
                    └─────────────────┘

为什么选择 q_proj 和 v_proj？
├── q_proj (Query)：决定"关注什么"
│   └── 影响模型对问题的理解方式
├── v_proj (Value)：决定"提取什么信息"
│   └── 影响模型生成回复的内容选择
├── 实验结论：q_proj + v_proj 覆盖 80% 的效果
└── 微调更多层（如 k_proj, o_proj）收益有限但显存增加

扩展选项：
- 最小配置：["q_proj", "v_proj"] ← 推荐
- 中等配置：["q_proj", "k_proj", "v_proj", "o_proj"]
- 最大配置：加上 MLP 层 ["gate_proj", "up_proj", "down_proj"]
```

#### ④ lora_dropout (Dropout 率)

```
含义：LoRA 层的 Dropout 比例，用于正则化

作用机制：
训练时：随机"丢弃"一定比例的神经元，防止过拟合
推理时：使用全部神经元

为什么设 0.05？
├── 小数据集容易过拟合，需要正则化
├── 0.05 是轻度正则化，不会过度限制学习能力
├── 配合 3 个 epoch：防止模型过度记忆训练数据
└── 太大（如 0.1-0.2）会降低模型收敛速度

调节建议：
- 数据量 < 1000 条 → dropout=0.1
- 数据量 1000-10000 条 → dropout=0.05 ← 推荐
- 数据量 > 10000 条 → dropout=0.0 或 0.02
```

### 2.3 数据格式：ChatML

ChatML 是 OpenAI 提出的对话格式，被 Qwen 等模型广泛采用：

```json
{
  "messages": [
    {"role": "system", "content": "你是一名专业的心理咨询客服..."},
    {"role": "user", "content": "我最近很迷茫..."},
    {"role": "assistant", "content": "看到你面临的困境..."}
  ]
}
```

**System Prompt 设计**：
```
你是一名专业的心理咨询客服，请根据来访者的问题，给出专业、共情的回复。

回复要求：
1. 首先表达对来访者情绪的理解和接纳
2. 分析来访者可能面临的问题
3. 提供具体、可行的建议
4. 以开放式问题或鼓励性话语引导来访者继续表达

注意：保持温和、专业的语气，避免过于绝对的建议，尊重来访者的感受。
```

### 2.4 训练参数解读

| 参数 | 设置值 | 含义 | 为什么这样设置 |
|------|-------|------|---------------|
| num_train_epochs | 3 | 训练轮数 | 小样本避免过拟合 |
| per_device_train_batch_size | 4 | 单卡批次大小 | 适配 24G 显存 |
| gradient_accumulation_steps | 4 | 梯度累积 | 等效 batch_size=16 |
| learning_rate | 2e-4 | 学习率 | LoRA 常用范围 |
| fp16 | True | 混合精度 | 节省显存、加速训练 |
| optim | paged_adamw_8bit | 优化器 | 8-bit 优化器节省显存 |

---

## 三、技术栈与环境

### 3.1 技术选型

| 类别 | 选型 | 选型理由 |
|------|------|---------|
| 基座模型 | Qwen2.5-7B-Instruct | 中文适配优、参数量适中、开源免费 |
| 微调方案 | QLoRA (4-bit) | 显存占用低、效果有保障 |
| 训练框架 | Transformers + PEFT | 成熟稳定、文档完善 |
| 数据处理 | Pandas + JSON | 高效处理表格和结构化数据 |
| 硬件 | RTX 4090 (24G) | 性价比高、适合单卡训练 |
| 部署 | FastAPI + Gradio | 高性能 API + 快速 Demo |

### 3.2 环境要求

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU | RTX 3090 (24G) | RTX 4090 (24G) |
| CPU | 8核 | 16核 |
| 内存 | 32GB | 64GB |
| 磁盘 | 50GB | 100GB SSD |
| Python | 3.10 | 3.10+ |
| CUDA | 11.8 | 12.x |

### 3.3 核心依赖

```txt
# 深度学习核心
torch>=2.0.0
transformers>=4.36.0
peft>=0.7.0
accelerate>=0.25.0
bitsandbytes>=0.41.0

# 数据处理
pandas>=2.0.0
numpy>=1.24.0
datasets>=2.15.0

# API & UI
fastapi>=0.104.0
gradio>=4.0.0
uvicorn>=0.24.0

# 工具
tqdm>=4.65.0
safetensors>=0.4.0
requests>=2.31.0
```

---

## 四、项目结构

```
MentalChat/
├── configs/
│   └── config.py              # 统一配置文件
├── data/
│   ├── raw/                   # 原始数据
│   │   └── data.csv
│   └── processed/             # 处理后的数据
│       ├── train.jsonl
│       ├── valid.jsonl
│       └── test.jsonl
├── models/
│   ├── base/                  # 基座模型
│   │   └── Qwen_Qwen2.5-7B-Instruct/
│   └── lora/                  # LoRA 权重
├── scripts/
│   ├── check_environment.py   # 环境检查
│   ├── download_model.py      # 模型下载
│   ├── process_data.py        # 数据处理
│   ├── train.py               # 训练脚本
│   ├── evaluate.py            # 评估脚本
│   ├── inference.py           # 推理脚本
│   └── verify_qlora.py        # QLoRA 验证
├── outputs/
│   ├── checkpoints/           # 训练检查点
│   └── logs/                  # 训练日志
├── requirements.txt           # 依赖清单
└── README.md                  # 本文档
```

---

## 五、快速开始

### 5.1 环境搭建（AutoDL 服务器）

```bash
# 1. 进入项目目录
cd /root/MentalChat

# 2. 配置 conda 镜像（清华源）
cat > ~/.condarc << 'EOF'
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
EOF

# 3. 初始化 conda
conda init bash && source ~/.bashrc

# 4. 创建虚拟环境
conda create -n mental_chat python=3.10 -y
conda activate mental_chat

# 5. 安装 PyTorch（清华镜像）
pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple

# 6. 安装依赖（清华镜像）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 7. 验证环境
python scripts/check_environment.py
```

### 5.2 下载模型

**模型下载路径说明**：

| 项目 | 说明 |
|------|------|
| 基座模型名称 | `Qwen/Qwen2.5-7B-Instruct` |
| 本地保存路径 | `models/base/Qwen_Qwen2.5-7B-Instruct/` |
| 模型精度 | **FP16/BF16 混合精度**（HuggingFace 默认格式） |
| 模型大小 | 约 **15GB**（7.6B 参数 × 2 bytes = 15.2GB） |

> **为什么是 15GB？**
> HuggingFace 上的模型默认以 FP16/BF16 半精度存储，每个参数占 2 字节。
> 7.6B 参数 × 2 bytes ≈ 15.2 GB，加上配置文件和分词器，总共约 15GB。

**下载命令**：

```bash
# 设置 HuggingFace 镜像（国内必须）
export HF_ENDPOINT=https://hf-mirror.com

# 方式1：使用脚本下载（推荐）
python scripts/download_model.py --mirror

# 方式2：手动下载到指定路径
# 模型会保存到: /root/MentalChat/models/base/Qwen_Qwen2.5-7B-Instruct/
```

**下载后的目录结构**：

```
models/base/Qwen_Qwen2.5-7B-Instruct/
├── config.json              # 模型配置
├── generation_config.json   # 生成配置
├── model-00001-of-00004.safetensors  # 模型权重（分片）
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
├── model.safetensors.index.json
├── tokenizer.json           # 分词器
├── tokenizer_config.json
├── special_tokens_map.json
└── vocab.json
```

### 5.3 处理数据

```bash
# 数据探查
python scripts/process_data.py --explore-only

# 数据处理（多轮对话格式）
python scripts/process_data.py --mode multi

# 启用 AI 数据增强（可选）
export DASHSCOPE_API_KEY="your-api-key"
python scripts/process_data.py --mode multi --augment --augment-ratio 0.3
```

### 5.4 验证 QLoRA

```bash
python scripts/verify_qlora.py
```

### 5.5 启动训练

```bash
# 使用 tmux 保持长时任务
tmux new -s finetune
conda activate mental_chat
python scripts/train.py

# 断开会话：Ctrl+B 然后按 D
# 重新连接：tmux attach -t finetune
```

---

## 六、详细执行步骤

### 6.1 阶段一：环境搭建与验证（1天）

| 任务 | 命令 | 预期结果 |
|------|------|---------|
| 硬件确认 | `nvidia-smi` | 显示 GPU 信息 |
| Python 环境 | `conda create -n mental_chat python=3.10` | 环境创建成功 |
| 依赖安装 | `pip install -r requirements.txt` | 无报错 |
| 环境验证 | `python scripts/check_environment.py` | 所有检查通过 |
| 模型下载 | `python scripts/download_model.py --mirror` | 保存到 `models/base/Qwen_Qwen2.5-7B-Instruct/` |
| QLoRA 验证 | `python scripts/verify_qlora.py` | 验证通过（显存约 6GB） |

### 6.2 阶段二：数据处理（1-2天）

| 任务 | 命令 | 产出 |
|------|------|------|
| 数据探查 | `python scripts/process_data.py --explore-only` | 数据统计报告 |
| 数据清洗 | 内置于 process_data.py | 清洗后的数据 |
| 格式转换 | `python scripts/process_data.py --mode multi` | ChatML 格式数据 |
| 数据集划分 | 内置于 process_data.py | train/valid/test.jsonl |

**数据增强（可选）**：

```bash
# 配置 API Key
export DASHSCOPE_API_KEY="sk-xxx"

# 启用增强
python scripts/process_data.py --mode multi --augment --augment-ratio 0.3
```

| 增强策略 | 说明 | 效果 |
|---------|------|------|
| paraphrase | 同义改写用户输入 | 增加输入多样性 |
| enhance | 增强回复质量 | 提升回复专业性 |
| clean | 清理和标准化 | 修复语法错误 |

### 6.3 阶段三：模型微调训练（2天）

**训练配置**：

```python
# LoRA 配置
r = 16
lora_alpha = 32
target_modules = ["q_proj", "v_proj"]
lora_dropout = 0.05

# QLoRA 配置
load_in_4bit = True
bnb_4bit_quant_type = "nf4"
bnb_4bit_compute_dtype = "float16"

# 训练配置
num_train_epochs = 3
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
learning_rate = 2e-4
```

**训练监控**：

```bash
# 实时监控 GPU
watch -n 1 nvidia-smi

# 查看训练日志
tail -f outputs/logs/training.log

# TensorBoard（如果启用）
tensorboard --logdir outputs/logs
```

**训练调优参考**：

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| Loss 不下降 | 学习率过低 | 提高到 5e-4 |
| Loss 震荡 | 学习率过高 | 降低到 1e-4 |
| 验证 Loss 高 | 过拟合 | 减少 epoch、增加 dropout |
| 显存溢出 | batch_size 过大 | 减小到 2 |

### 6.4 阶段四：模型评估（1天）

**自动化评估**：

```bash
python scripts/evaluate.py --model-path outputs/checkpoints/best
```

**人工评估维度**：

| 维度 | 权重 | 目标值 |
|------|------|-------|
| 专业度 | 30% | ≥90% |
| 共情能力 | 30% | ≥90% |
| 格式规范 | 20% | ≥95% |
| 语言流畅 | 10% | ≥95% |
| 安全性 | 10% | 100% |

### 6.5 阶段五：部署与演示（1天）

**启动 API 服务**：

```bash
python scripts/inference.py --mode api --port 8000
```

**启动 Demo 界面**：

```bash
python scripts/inference.py --mode gradio --port 7860
```

---

## 七、常见问题

### 7.1 环境问题

**Q: conda activate 报错？**
```bash
conda init bash
source ~/.bashrc
# 或使用替代命令
source activate mental_chat
```

**Q: CUDA 不可用？**
```bash
# 检查 PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 重新安装对应版本
pip install torch torchvision --index-url https://mirrors.aliyun.com/pytorch-wheels/cu121
```

**Q: bitsandbytes 安装失败？**
```bash
# 需要在 Linux 环境安装
pip install bitsandbytes -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**Q: 磁盘空间不足？**

下载模型前先清理各类缓存，通常可释放 10-30GB 空间：

```bash
# ========== 清理 pip 缓存 ==========
# 查看缓存大小
pip cache info

# 清理所有 pip 缓存（可释放数GB）
pip cache purge

# ========== 清理 conda 缓存 ==========
# 查看缓存大小
conda clean --dry-run --all

# 清理所有 conda 缓存（包、tar包、索引缓存）
conda clean --all -y

# ========== 清理 HuggingFace 缓存 ==========
# HF 缓存通常在 ~/.cache/huggingface/
du -sh ~/.cache/huggingface/  # 查看大小
rm -rf ~/.cache/huggingface/  # 清理（如果之前下载失败会有残留）

# ========== 清理系统缓存 ==========
# 清理 apt 缓存（Ubuntu/Debian）
sudo apt clean
sudo apt autoremove -y

# 清理日志文件
sudo journalctl --vacuum-size=100M

# 清理 tmp 目录
sudo rm -rf /tmp/*

# ========== 查看磁盘使用情况 ==========
# 查看各目录占用
du -sh /* 2>/dev/null | sort -hr | head -20

# 查看当前目录占用
du -sh * | sort -hr

# ========== 大文件查找 ==========
# 查找大于 100MB 的文件
find / -type f -size +100M 2>/dev/null | head -20
```

**预计释放空间**：

| 缓存类型 | 通常大小 | 清理命令 |
|---------|---------|---------|
| pip 缓存 | 2-10 GB | `pip cache purge` |
| conda 缓存 | 3-15 GB | `conda clean --all -y` |
| HF 缓存 | 0-30 GB | `rm -rf ~/.cache/huggingface/` |
| apt 缓存 | 1-5 GB | `sudo apt clean` |
| 日志文件 | 0.5-2 GB | `sudo journalctl --vacuum-size=100M` |

### 7.2 训练问题

**Q: 显存溢出 (OOM)？**
- 减小 `per_device_train_batch_size` 到 2
- 增加 `gradient_accumulation_steps` 到 8
- 启用更激进的量化

**Q: Loss 不下降？**
- 检查学习率（尝试 1e-4 到 5e-4）
- 检查数据格式是否正确
- 增加训练轮数

**Q: 训练很慢？**
- 确认使用 fp16 混合精度
- 检查数据加载是否有瓶颈
- 使用更快的存储（SSD）

### 7.3 推理问题

**Q: 回复质量差？**
- 检查是否加载了 LoRA 权重
- 调整 temperature（0.7-0.9）
- 调整 top_p（0.8-0.95）

**Q: 推理速度慢？**
- 启用 4-bit 量化推理
- 考虑使用 vLLM 框架
- 实现批处理推理

---

## 八、面试要点

### 8.1 项目背景类

**Q: 为什么做这个项目？**
> 通用大模型在心理咨询场景存在专业度不足、共情能力弱、格式不统一等问题，通过微调可以让模型更好地适配这一垂直场景。

**Q: 为什么选择 LoRA/QLoRA 而非全量微调？**
> 全量微调显存占用高（80G+）、易过拟合、成本高。QLoRA 通过 4-bit 量化和低秩适配，将显存降至 11G，单卡即可训练，且精度损失可控（<5%）。

### 8.2 技术细节类

**Q: LoRA 的 r=16 是如何确定的？**
> 通过实验验证：r=8 效果不足（专业度仅 85%），r=32 显存过高（15G+），r=16 是效果与成本的最优平衡点。

**Q: 为什么选择 q_proj 和 v_proj？**
> 这两个模块是注意力层的核心，直接影响模型对语义和情绪的理解。微调它们能快速适配场景，同时控制可训练参数。

**Q: QLoRA 和 LoRA 的区别？**
> 核心区别是基座模型精度：LoRA 是 32-bit，QLoRA 是 4-bit。显存从 20G 降至 11G，精度损失仅 2%。

### 8.3 工程实践类

**Q: 如何保证数据质量？**
> 通过去重、纠错、敏感词过滤、格式标准化等多步处理，并进行人工抽样检查（100 条样本）。

**Q: 如何评估模型效果？**
> 自动评估（Perplexity、BLEU）+ 人工评估（专业度、共情能力、格式规范）+ 对比评估（与基座模型对比）。

**Q: 如何优化推理性能？**
> 4-bit 量化推理、Flash Attention、vLLM 框架、批处理推理，可将延迟控制在 100ms 以内。

---

## 附录

### A. 国内镜像源

| 类型 | 镜像地址 |
|------|---------|
| 清华 PyPI | https://pypi.tuna.tsinghua.edu.cn/simple |
| 阿里云 PyPI | https://mirrors.aliyun.com/pypi/simple/ |
| 阿里云 PyTorch | https://mirrors.aliyun.com/pytorch-wheels/ |
| HF 镜像 | https://hf-mirror.com |

### B. 推荐阅读

- [QLoRA 论文](https://arxiv.org/abs/2305.14314)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [PEFT 文档](https://huggingface.co/docs/peft)
- [Transformers 文档](https://huggingface.co/docs/transformers)

### C. 项目时间规划

```
第1天：环境搭建与验证
第2-3天：数据处理
第4-5天：模型微调训练
第6-7天：模型评估与优化
第8天：部署与演示
```

---

*文档版本：v2.0*
*适用场景：心理咨询客服对话模型微调项目*
