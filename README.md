# 心理咨询客服对话模型微调项目

> 本项目使用 QLoRA 技术对 Qwen2.5-7B-Instruct 进行轻量化微调，使其适配心理咨询客服场景。

---

## 目录

- [项目背景与目标](#一项目背景与目标)
- [微调背景知识](#二微调背景知识)
- [技术栈与环境](#三技术栈与环境)
- [项目结构](#四项目结构)
- [快速开始](#五快速开始)
- [详细执行步骤](#六详细执行步骤)
- [数据增强说明](#七数据增强说明)
- [常见问题](#八常见问题)

---

## 一、项目背景与目标

### 1.1 项目背景

通用大模型在心理咨询客服场景存在明显痛点：

| 痛点 | 具体表现 |
|------|----------|
| 缺乏专业知识 | 无法识别心理学术语，难以提供专业建议 |
| 共情能力不足 | 回复过于机械，缺乏情感理解和人文关怀 |
| 安全边界模糊 | 可能给出危险建议或无法识别危机情况（如自杀倾向） |
| 场景适配差 | 通用回复不够贴近心理咨询对话风格和节奏 |

### 1.2 项目目标

- **专业化**：让模型掌握心理咨询领域知识和对话技巧
- **共情化**：提升模型的情感理解和表达能力
- **安全性**：模型能够识别危机情况并给出恰当回应
- **实用性**：满足实际客服场景的对话需求

### 1.3 应用场景

- 在线心理咨询平台
- 心理健康热线辅助系统
- 企业员工心理援助计划（EAP）
- 心理健康教育问答系统

---

## 二、微调背景知识

> 本章详细介绍大模型微调的核心概念、原理和技术细节，帮助读者深入理解 QLoRA 微调的底层机制。

### 2.1 什么是微调 (Fine-tuning)

#### 2.1.1 基本概念

微调（Fine-tuning）是指在预训练模型的基础上，使用特定领域或任务的数据进行进一步训练，使模型适应特定场景。

**为什么需要微调？**

1. **领域适配**：预训练模型是通用的，特定领域（如医疗、法律、心理咨询）需要专门的知识
2. **风格定制**：调整模型的输出风格，使其符合特定应用场景
3. **任务优化**：针对特定任务（如分类、生成、对话）进行优化
4. **成本效益**：相比从头训练，微调可以用较少的数据和计算资源获得更好的效果

#### 2.1.2 微调的分类

| 类型 | 说明 | 参数量 | 适用场景 |
|------|------|--------|----------|
| **全量微调** | 更新所有模型参数 | 100% | 大规模数据、高性能硬件 |
| **参数高效微调 (PEFT)** | 只更新部分参数 | <1% | 资源受限场景 |
| **提示微调 (Prompt Tuning)** | 只优化提示向量 | 极少 | 简单任务 |
| **前缀微调 (Prefix Tuning)** | 在每层添加可学习前缀 | ~0.1% | 生成任务 |

### 2.2 PEFT 框架详解

#### 2.2.1 什么是 PEFT

PEFT (Parameter-Efficient Fine-Tuning) 是 Hugging Face 提供的参数高效微调框架，包含多种微调方法：

```
PEFT 方法分类
├── LoRA 系列
│   ├── LoRA (Low-Rank Adaptation)
│   ├── QLoRA (Quantized LoRA)
│   └── AdaLoRA (Adaptive LoRA)
├── Prompt 系列
│   ├── Prompt Tuning
│   ├── Prefix Tuning
│   └── P-Tuning
└── Adapter 系列
    ├── Adapter
    ├── AdapterHub
    └── IA³
```

#### 2.2.2 PEFT 的优势

| 优势 | 说明 |
|------|------|
| **显存效率** | 只需存储少量可训练参数，大幅降低显存需求 |
| **存储效率** | 微调后的权重文件很小（MB 级别），便于分享和部署 |
| **训练速度** | 参数少，反向传播更快 |
| **多任务支持** | 可以为不同任务训练不同的适配器，共享基座模型 |
| **避免灾难性遗忘** | 基座模型参数不变，保留原有能力 |

### 2.3 LoRA 原理详解

#### 2.3.1 LoRA 的核心思想

LoRA (Low-Rank Adaptation) 的核心假设：**模型微调过程中的权重更新具有低秩特性**。

假设预训练模型的权重矩阵为 $W_0 \in \mathbb{R}^{d \times k}$，微调后的权重为 $W = W_0 + \Delta W$。LoRA 认为 $\Delta W$ 可以分解为两个低秩矩阵的乘积：

$$\Delta W = B \times A$$

其中：
- $B \in \mathbb{R}^{d \times r}$：降维矩阵
- $A \in \mathbb{R}^{r \times k}$：升维矩阵
- $r \ll \min(d, k)$：秩（rank），通常取 4-64

#### 2.3.2 LoRA 的数学推导

**参数量对比：**

| 方法 | 参数量 | 示例 (d=k=4096) |
|------|--------|-----------------|
| 全量微调 | $d \times k$ | 16,777,216 |
| LoRA (r=16) | $d \times r + r \times k$ | 131,072 |
| 压缩比 | - | **128 倍** |

**前向传播：**

```python
# 原始线性层
output = W0 @ x

# LoRA 增强后
output = W0 @ x + (B @ A) @ x
output = W0 @ x + B @ (A @ x)  # 计算效率更高
```

**缩放因子：**

LoRA 引入缩放因子 $\alpha$ 来控制适配器的影响：

$$h = W_0 x + \frac{\alpha}{r} BAx$$

通常设置 $\alpha = 2r$，使得缩放因子为 2。

#### 2.3.3 LoRA 应用于 Transformer

在 Transformer 中，LoRA 通常应用于注意力层的投影矩阵：

```
Transformer Block
├── Self-Attention
│   ├── Q_proj (Query)    ← LoRA 目标
│   ├── K_proj (Key)
│   ├── V_proj (Value)    ← LoRA 目标
│   └── O_proj (Output)
├── MLP
│   ├── gate_proj         ← LoRA 目标 (可选)
│   ├── up_proj           ← LoRA 目标 (可选)
│   └── down_proj         ← LoRA 目标 (可选)
└── LayerNorm (不应用 LoRA)
```

**常见配置：**

| 配置 | target_modules | 参数量 | 效果 |
|------|----------------|--------|------|
| 最小配置 | ["q_proj", "v_proj"] | 最少 | 基础效果 |
| 推荐配置 | ["q_proj", "v_proj", "k_proj", "o_proj"] | 中等 | 较好效果 |
| 完整配置 | ["all_linear_layers"] | 最多 | 最佳效果 |

#### 2.3.4 LoRA 的初始化

LoRA 的初始化策略确保训练开始时不影响基座模型：

- **矩阵 A**：使用随机高斯分布初始化 $A \sim \mathcal{N}(0, \sigma^2)$
- **矩阵 B**：初始化为零 $B = 0$

这样 $\Delta W = B \times A = 0$，训练开始时 $W = W_0$。

### 2.4 QLoRA 原理详解

#### 2.4.1 QLoRA 的创新点

QLoRA 在 LoRA 的基础上引入了三项创新：

```
QLoRA = LoRA + 量化技术

创新点：
1. 4-bit NormalFloat (NF4) 量化
2. 双重量化 (Double Quantization)
3. 分页优化器 (Paged Optimizers)
```

#### 2.4.2 4-bit NormalFloat (NF4) 量化

**量化基础：**

量化是将高精度浮点数映射到低精度表示的过程。

| 数据类型 | 位数 | 数值范围 | 精度 |
|----------|------|----------|------|
| FP32 | 32 | ±3.4e38 | ~7位有效数字 |
| FP16 | 16 | ±65504 | ~3位有效数字 |
| BF16 | 16 | ±3.4e38 | ~2位有效数字 |
| INT8 | 8 | -128 ~ 127 | 整数 |
| **NF4** | **4** | **-1 ~ 1** | **信息论最优** |

**NF4 的优势：**

NF4 是针对正态分布权重设计的信息论最优量化数据类型：

1. **信息论最优**：对于正态分布的权重，NF4 提供最优的量化精度
2. **无需校准**：不需要额外的校准数据
3. **零样本量化**：可以直接量化预训练模型

**量化过程：**

```
FP16 权重 → 归一化 → 量化到 NF4 → 存储

示例：
原始权重: 0.1234 (FP16)
归一化: 0.1234 / absmax = 0.1234 / 2.5 = 0.04936
量化: 找到最接近的 NF4 值 (0.0469)
存储: NF4 index (7) + absmax (2.5)
```

#### 2.4.3 双重量化 (Double Quantization)

**问题：** 量化的缩放因子（absmax）本身也需要存储，占用额外显存。

**解决方案：** 对缩放因子再次量化。

```
普通量化：
  权重 (FP16) → 量化 → NF4 权重
  每个 block 存储一个 FP16 缩放因子

双重量化：
  权重 (FP16) → 第一次量化 → NF4 权重 + FP16 缩放因子
  FP16 缩放因子 → 第二次量化 → NF8 缩放因子 + FP32 二次缩放因子

显存节省：
  每个 block 节省: 2 bytes (FP16) - 0.5 bytes (NF8) = 1.5 bytes
  对于 7B 模型: 节省约 0.5GB
```

#### 2.4.4 分页优化器 (Paged Optimizers)

**问题：** 训练过程中显存使用会有峰值，导致 OOM。

**解决方案：** 使用 CPU 内存作为溢出缓冲区。

```
GPU 显存使用模式：
├── 模型权重 (固定)
├── LoRA 参数 (固定)
├── 梯度 (变化)
├── 优化器状态 (变化，主要峰值来源)
└── 激活值 (变化)

分页优化器：
  当 GPU 显存不足时 → 将优化器状态移到 CPU 内存
  当需要更新时 → 移回 GPU
```

### 2.5 显存占用计算

#### 2.5.1 模型显存估算

**基本公式：**

```
模型显存 ≈ 参数量 × 每参数字节数
```

| 精度 | 每参数字节数 | 7B 模型显存 | 13B 模型显存 |
|------|--------------|-------------|--------------|
| FP32 | 4 bytes | ~28 GB | ~52 GB |
| FP16/BF16 | 2 bytes | ~14 GB | ~26 GB |
| INT8 | 1 byte | ~7 GB | ~13 GB |
| **4-bit (NF4)** | **0.5 bytes** | **~3.5 GB** | **~6.5 GB** |

#### 2.5.2 训练显存估算

训练显存包括多个部分：

```
总显存 = 模型权重 + 梯度 + 优化器状态 + 激活值 + 临时缓存
```

**各部分详解：**

| 组件 | 全量微调 | LoRA | QLoRA |
|------|----------|------|-------|
| 模型权重 | 14GB (FP16) | 14GB (FP16) | 3.5GB (4-bit) |
| 梯度 | 14GB | ~0.1GB (LoRA) | ~0.1GB (LoRA) |
| 优化器状态 | 28GB (Adam) | ~0.2GB | ~0.2GB |
| 激活值 | 4-8GB | 4-8GB | 4-8GB |
| **总计** | **~60GB** | **~20GB** | **~6-8GB** |

#### 2.5.3 实际计算示例

以 Qwen2.5-7B 为例：

```python
# 模型参数
params = 7_000_000_000  # 7B

# 1. 4-bit 量化模型权重
model_memory = params * 0.5 / (1024**3)  # ≈ 3.26 GB

# 2. LoRA 参数 (r=16, target_modules=["q_proj", "v_proj"])
# 假设每个投影矩阵维度为 4096x4096
lora_params = 2 * 32 * 4096 * 16 * 2  # 2 modules, 2 matrices each
lora_memory = lora_params * 2 / (1024**3)  # ≈ 0.008 GB

# 3. 优化器状态 (AdamW: 2 states per param)
optimizer_memory = lora_params * 2 * 2 / (1024**3)  # ≈ 0.016 GB

# 4. 激活值 (取决于 batch_size 和 sequence_length)
# batch_size=4, seq_len=2048, hidden_dim=4096
activation_memory = 4 * 2048 * 4096 * 32 / (1024**3)  # ≈ 4 GB (估算)

# 总计
total = model_memory + lora_memory + optimizer_memory + activation_memory
# ≈ 7.3 GB
```

### 2.6 训练技巧与最佳实践

#### 2.6.1 学习率选择

| 微调方法 | 推荐学习率 | 说明 |
|----------|------------|------|
| 全量微调 | 1e-5 ~ 5e-5 | 较小学习率避免破坏预训练知识 |
| LoRA | 1e-4 ~ 5e-4 | 可以使用较大学习率 |
| QLoRA | 1e-4 ~ 2e-3 | 更大的学习率通常效果更好 |

#### 2.6.2 LoRA Rank 选择

| Rank | 参数量 | 适用场景 |
|------|--------|----------|
| r=4 | 最少 | 简单任务、数据少 |
| r=8 | 较少 | 通用任务 |
| **r=16** | **中等** | **推荐默认值** |
| r=32 | 较多 | 复杂任务 |
| r=64 | 最多 | 需要最大表达能力 |

**经验法则：** 从 r=16 开始，如果效果不好再尝试更大的值。

#### 2.6.3 梯度累积

当显存不足以支持较大 batch_size 时，可以使用梯度累积模拟大批次：

```python
# 实际 batch_size = 2
# 梯度累积步数 = 8
# 等效 batch_size = 2 * 8 = 16

# 每 8 步更新一次参数
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / gradient_accumulation_steps
    loss.backward()

    if (i + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### 2.6.4 混合精度训练

使用 FP16/BF16 混合精度可以减少显存占用并加速训练：

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    fp16=True,  # 使用 FP16 混合精度
    # 或 bf16=True,  # 使用 BF16（推荐，如果硬件支持）
)
```

### 2.7 为什么选择 Qwen2.5-7B-Instruct

| 特点 | 说明 |
|------|------|
| 中文能力强 | 在中文理解和生成方面表现优秀，C-Eval 等中文榜单领先 |
| 指令遵循好 | 能够准确理解和执行复杂指令，对话能力强 |
| 开源免费 | Apache 2.0 协议，可商用 |
| 社区活跃 | Hugging Face 下载量高，文档完善，生态丰富 |
| 模型大小适中 | 7B 参数在效果和效率间取得平衡，适合个人和小团队 |
| 长上下文支持 | 支持 32K 上下文，适合长对话场景 |
| 工具调用能力 | 支持函数调用，可扩展性强 |

---

## 三、技术栈与环境

### 3.1 技术栈

| 类别 | 技术 | 版本要求 | 说明 |
|------|------|----------|------|
| 基座模型 | Qwen/Qwen2.5-7B-Instruct | - | 阿里通义千问系列 |
| 深度学习框架 | PyTorch | >= 2.0.0 | 核心训练框架 |
| 模型库 | Transformers | >= 4.36.0 | Hugging Face 模型库 |
| 微调框架 | PEFT | >= 0.7.0 | 参数高效微调 |
| 量化工具 | BitsAndBytes | >= 0.41.0 | 4-bit 量化支持 |
| 训练加速 | Accelerate | >= 0.25.0 | 分布式训练支持 |
| 数据处理 | Pandas, Datasets | >= 2.0.0 | 数据加载和处理 |
| 可视化界面 | Gradio | >= 4.0.0 | Web 界面 |

### 3.2 硬件要求

| 资源 | 最低配置 | 推荐配置 | 说明 |
|------|----------|----------|------|
| GPU | RTX 3060 (12GB) | RTX 4090 (24GB) | QLoRA 训练至少需要 6GB 显存 |
| 内存 | 16GB | 64GB | 数据加载和缓存 |
| 磁盘 | 50GB SSD | 100GB NVMe | 模型存储和检查点 |

**不同模型大小的显存需求：**

| 模型 | QLoRA 训练 | LoRA 训练 | 推理 (4-bit) |
|------|------------|-----------|--------------|
| 7B | ~8GB | ~20GB | ~4GB |
| 13B | ~12GB | ~35GB | ~7GB |
| 34B | ~20GB | ~80GB | ~18GB |
| 70B | ~40GB | ~160GB | ~38GB |

### 3.3 软件环境

```
Python >= 3.10
CUDA >= 12.0
cuDNN >= 8.0
```

### 3.4 推荐平台

| 平台 | 特点 | 价格参考 |
|------|------|----------|
| AutoDL | 国内平台，性价比高，适合学习 | RTX 4090 ~2元/小时 |
| 阿里云 PAI | 企业级，稳定可靠 | 按需计费 |
| Lambda Labs | 海外平台，A100/H100 资源丰富 | A100 ~0.5美元/小时 |

---

## 四、项目结构

```
MentalChat/
├── config/                          # 配置模块
│   ├── __init__.py                  # 模块导出
│   └── config.py                    # 统一配置文件（唯一配置源）
│
├── data/                            # 数据模块
│   ├── raw/                         # 原始数据
│   │   └── data.csv                 # CSV 格式原始数据
│   └── processed/                   # 处理后的数据
│       ├── train.jsonl              # 训练集 (ChatML 格式)
│       ├── valid.jsonl              # 验证集
│       └── test.jsonl               # 测试集
│
├── scripts/                         # 核心代码模块
│   ├── check_environment.py         # 环境检查脚本
│   ├── download_model.py            # 模型下载脚本
│   ├── process_data.py              # 数据处理脚本
│   ├── augmentation.py              # 数据增强模块
│   ├── train.py                     # 模型训练脚本
│   ├── evaluate.py                  # 模型评估脚本
│   ├── inference.py                 # 模型推理脚本
│   └── verify_qlora.py              # QLoRA 配置验证
│
├── output/                          # 输出目录
│   ├── checkpoints/                 # 训练检查点
│   │   ├── checkpoint-500/          # 中间检查点
│   │   └── final/                   # 最终模型
│   └── logs/                        # 训练日志
│       └── tensorboard/             # TensorBoard 日志
│
├── chat/                            # 对话界面
│   └── app.py                       # Gradio 应用
│
├── /root/autodl-tmp/MentalChat/models/   # 模型目录（AutoDL 数据盘）
│   ├── base/                        # 基座模型（约 15GB）
│   │   └── Qwen_Qwen2.5-7B-Instruct/
│   └── lora/                        # LoRA 权重
│
├── requirements.txt                 # 依赖清单
└── README.md                        # 本文档
```

### 4.1 目录说明

| 目录 | 用途 | 备注 |
|------|------|------|
| `config/` | 存放所有配置参数 | 唯一配置源，避免配置分散 |
| `data/raw/` | 存放原始 CSV 数据 | 需要按规范格式准备 |
| `data/processed/` | 存放处理后的 JSONL 数据 | 由脚本自动生成 |
| `scripts/` | 所有训练相关代码 | 核心功能实现 |
| `output/` | 训练输出和日志 | 自动创建，可配置 |
| `chat/` | Gradio 对话界面 | 用于模型演示 |

---

## 五、快速开始

### 5.1 环境搭建

```bash
# 1. 克隆项目（或在 AutoDL 上上传）
cd /root
git clone <repository-url>
cd MentalChat

# 2. 创建虚拟环境
conda create -n mental_chat python=3.10 -y
conda activate mental_chat

# 3. 安装 PyTorch（根据 CUDA 版本选择）
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. 安装其他依赖
pip install -r requirements.txt

# 5. 检查环境
python scripts/check_environment.py
```

### 5.2 数据准备

#### 5.2.1 基础数据处理

```bash
# 步骤 1: 将原始数据放入 data/raw/ 目录
# 数据格式要求：CSV 文件，包含 Conversation ID, Turn ID, Input, Output 字段

# 步骤 2: 查看数据探查结果（可选）
python scripts/process_data.py --explore-only

# 步骤 3: 处理数据（多轮对话模式）
python scripts/process_data.py --mode multi
```

#### 5.2.2 AI 数据增强（可选）

数据增强可以扩充训练数据，提升模型效果。

**快速开始：**

```bash
# 基础用法（使用默认设置）
python scripts/process_data.py --mode multi --augment

# 指定增强比例和 API
python scripts/process_data.py --mode multi --augment --augment-ratio 0.3 --api-type qwen
```

**选择 API 提供商：**

```bash
# 使用 Qwen API（阿里云）
export DASHSCOPE_API_KEY='your-api-key'
python scripts/process_data.py --mode multi --augment --api-type qwen

# 使用 GLM API（智谱 AI）
export ZHIPUAI_API_KEY='your-api-key'
python scripts/process_data.py --mode multi --augment --api-type glm
```

| API | 获取地址 | 推荐模型 |
|-----|---------|---------|
| Qwen | https://dashscope.console.aliyun.com/ | qwen-plus |
| GLM | https://open.bigmodel.cn/ | glm-4.7 |

**选择增强策略：**

```bash
# 单一策略
python scripts/process_data.py --mode multi --augment --strategies paraphrase

# 多策略组合
python scripts/process_data.py --mode multi --augment --strategies paraphrase enhance

# 全部策略
python scripts/process_data.py --mode multi --augment --strategies paraphrase enhance scenario
```

| 策略 | 说明 |
|------|------|
| `paraphrase` | 同义改写用户输入，增加表达多样性（默认） |
| `enhance` | 优化咨询师回复，使回复更专业、更有共情心 |
| `scenario` | 场景扩展，将问题改编为不同生活场景 |

**完整参数示例：**

```bash
# 使用 Qwen API，30% 增强比例，启用改写+优化两种策略
python scripts/process_data.py --mode multi \
    --augment \
    --augment-ratio 0.3 \
    --api-type qwen \
    --strategies paraphrase enhance
```

### 5.3 模型下载

```bash
# 下载基座模型到本地
python scripts/download_model.py

# 或使用镜像加速
python scripts/download_model.py --use-mirror
```

### 5.4 开始训练

```bash
# 使用默认配置训练
python scripts/train.py

# 自定义参数训练
python scripts/train.py --epochs 5 --learning-rate 1e-4 --batch-size 2

# 从检查点恢复训练
python scripts/train.py --resume-from output/checkpoints/checkpoint-1000
```

### 5.5 模型评估与推理

```bash
# 评估模型
python scripts/evaluate.py --lora-path output/checkpoints/final

# 命令行交互式推理
python scripts/inference.py --lora-path output/checkpoints/final --interactive

# 单次输入推理
python scripts/inference.py --lora-path output/checkpoints/final --input "我最近感觉很焦虑"

# 启动 Gradio 对话界面
python chat/app.py
# 访问 http://localhost:7860
```

### 5.6 模型对比评估（推荐）

完成微调后，强烈建议对比微调前后的模型效果，评估微调价值。

```bash
# 对比微调模型和基座模型
python scripts/compare_models.py --lora-path output/checkpoints/final

# 快速测试（限制样本数）
python scripts/compare_models.py --lora-path output/checkpoints/final --max-samples 50

# 保存详细对比结果
python scripts/compare_models.py --lora-path output/checkpoints/final --save-details
```

**评估维度：**

| 指标 | 说明 | 期望效果 |
|------|------|--------|
| ROUGE-1/2/L | 文本相似度 | 微调后应略高 |
| 专业术语覆盖 | 心理咨询关键词使用率 | 微调后应更高 |
| 平均回复长度 | 回复详细程度 | 微调后可能更长 |
| 回复样例 | 直观对比回复质量 | 人工评估参考 |

**输出示例:**
```
======================================================================
模型对比评估报告
======================================================================
测试样本: 100 条

------------------------------------------------------------------------------
【指标对比】
------------------------------------------------------------------------------
指标              基座模型        微调后模型        提升
------------------------------------------------------------------------------
ROUGE-1          0.356          0.412          +15.7%
ROUGE-2          0.142          0.187          +31.7%
ROUGE-L          0.289          0.345          +19.4%
专业术语覆盖率     23.4%          45.6%          +22.2pp
平均回复长度       42.3字         78.5字         +36.2字

------------------------------------------------------------------------------
【回复样例对比】
------------------------------------------------------------------------------
用户输入: 我最近总是感觉很焦虑，不知道该怎么办...

基座模型回复:
  作为一个AI，我不能给出医疗建议...建议您咨询专业心理医生...

微调后模型回复:
  我理解你现在的焦虑感受，这种感觉确实让人很不安。你愿意多和我

聊聊是什么让你感到焦虑的吗？我们可以一起找到应对的方法。
```

---

## 六、详细执行步骤

### 6.1 数据准备

#### 数据格式要求

原始数据应为 CSV 格式，包含以下字段：

| 字段名 | 说明 | 示例 |
|--------|------|------|
| Conversation ID | 对话唯一标识 | conv_001 |
| Turn ID | 对话轮次 | 1, 2, 3... |
| Input | 用户输入 | "最近工作压力很大" |
| Output | 咨询师回复 | "我理解你的感受..." |

#### 数据处理流程

```bash
# 单轮对话模式（每条数据独立）
python scripts/process_data.py --mode single

# 多轮对话模式（保持对话连贯性）
python scripts/process_data.py --mode multi
```

处理后的数据格式（ChatML）：

```json
{
  "messages": [
    {"role": "system", "content": "你是一名专业的心理咨询客服..."},
    {"role": "user", "content": "我最近感觉很焦虑"},
    {"role": "assistant", "content": "我理解你现在的感受..."}
  ]
}
```

### 6.2 训练配置详解

在 `config/config.py` 中调整训练参数。以下是所有参数的详细说明：

#### 6.2.1 模型配置 (ModelConfig)

```python
@dataclass
class ModelConfig:
    # 基座模型名称（Hugging Face 模型 ID）
    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct"

    # 基座模型本地路径（下载后的存放位置）
    base_model_path: str = "/root/autodl-tmp/MentalChat/models/base/Qwen_Qwen2.5-7B-Instruct"

    # 是否使用本地模型（False 则从 HF 下载）
    use_local_model: bool = True

    # 信任远程代码（某些模型需要）
    trust_remote_code: bool = True
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `base_model_name` | str | "Qwen/Qwen2.5-7B-Instruct" | Hugging Face 上的模型标识符 |
| `base_model_path` | str | "/root/autodl-tmp/..." | 模型本地存储路径 |
| `use_local_model` | bool | True | 是否优先使用本地模型 |
| `trust_remote_code` | bool | True | 是否执行模型仓库中的代码 |

#### 6.2.2 LoRA 配置 (LoRAConfig)

```python
@dataclass
class LoRAConfig:
    # LoRA 秩（rank），核心参数
    r: int = 16

    # LoRA 缩放因子
    lora_alpha: int = 32

    # 目标模块列表
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Dropout 概率
    lora_dropout: float = 0.05

    # 偏置项处理方式
    bias: str = "none"

    # 任务类型
    task_type: str = "CAUSAL_LM"
```

| 参数 | 类型 | 默认值 | 详细说明 |
|------|------|--------|----------|
| `r` | int | 16 | **LoRA 秩**，决定低秩矩阵的维度。r 越大，可训练参数越多，表达能力越强，但显存占用也越高。<br>• r=4-8：简单任务，数据量小<br>• r=16-32：推荐默认值，平衡效果和效率<br>• r=64+：复杂任务，需要更强的表达能力 |
| `lora_alpha` | int | 32 | **缩放因子**，控制 LoRA 适配器对原始权重的贡献程度。实际缩放系数为 `alpha/r`。通常设置为 `2r`，即缩放系数为 2。 |
| `target_modules` | List[str] | ["q_proj", "v_proj"] | **目标模块**，指定在哪些层应用 LoRA。<br>• 最小：["q_proj", "v_proj"]<br>• 推荐：["q_proj", "v_proj", "k_proj", "o_proj"]<br>• 完整：["all_linear_layers"] |
| `lora_dropout` | float | 0.05 | **Dropout 概率**，防止过拟合。通常 0.05-0.1 效果较好。数据量大时可设为 0。 |
| `bias` | str | "none" | **偏置项处理**。<br>• "none"：不训练偏置（推荐）<br>• "all"：训练所有偏置<br>• "lora_only"：只训练 LoRA 层偏置 |
| `task_type` | str | "CAUSAL_LM" | **任务类型**。因果语言模型用 "CAUSAL_LM"，序列到序列用 "SEQ_2_SEQ_LM"。 |

**LoRA 参数量计算**：

```
每个目标模块的 LoRA 参数量 = 2 × hidden_dim × r
（2 是因为有两个矩阵 A 和 B）

总参数量 = len(target_modules) × num_layers × 2 × hidden_dim × r

示例（Qwen2.5-7B）：
- hidden_dim = 3584
- num_layers = 28
- target_modules = ["q_proj", "v_proj"] (2个)
- r = 16

总参数量 = 2 × 28 × 2 × 3584 × 16 ≈ 6.4M 参数
```

#### 6.2.3 量化配置 (QuantizationConfig)

```python
@dataclass
class QuantizationConfig:
    # 是否使用 4-bit 量化
    load_in_4bit: bool = True

    # 4-bit 量化类型
    bnb_4bit_quant_type: str = "nf4"

    # 是否使用双重量化
    bnb_4bit_use_double_quant: bool = True

    # 计算数据类型
    bnb_4bit_compute_dtype: str = "float16"
```

| 参数 | 类型 | 默认值 | 详细说明 |
|------|------|--------|----------|
| `load_in_4bit` | bool | True | **是否启用 4-bit 量化**。启用后模型权重以 4-bit 存储，大幅降低显存占用。 |
| `bnb_4bit_quant_type` | str | "nf4" | **量化数据类型**。<br>• "nf4"：NormalFloat4，信息论最优，适合正态分布权重（推荐）<br>• "fp4"：Float4，通用浮点格式 |
| `bnb_4bit_use_double_quant` | bool | True | **是否启用双重量化**。对量化常数再次量化，额外节省约 0.5GB 显存（7B 模型）。 |
| `bnb_4bit_compute_dtype` | str | "float16" | **计算精度**。前向和反向传播时使用的精度。<br>• "float16"：平衡精度和速度<br>• "bfloat16"：更大动态范围（推荐 A100/H100）<br>• "float32"：最高精度，显存占用大 |

#### 6.2.4 训练配置 (TrainingConfig)

```python
@dataclass
class TrainingConfig:
    # ==================== 基础训练参数 ====================
    # 训练轮数
    num_train_epochs: int = 3

    # 每设备批次大小
    per_device_train_batch_size: int = 4

    # 每设备评估批次大小
    per_device_eval_batch_size: int = 4

    # 梯度累积步数
    gradient_accumulation_steps: int = 4

    # 学习率
    learning_rate: float = 2e-4

    # ==================== 优化器参数 ====================
    # 优化器类型
    optim: str = "paged_adamw_8bit"

    # 权重衰减
    weight_decay: float = 0.01

    # 最大梯度范数
    max_grad_norm: float = 1.0

    # ==================== 学习率调度 ====================
    # 学习率调度器类型
    lr_scheduler_type: str = "cosine"

    # 预热步数占比
    warmup_ratio: float = 0.03

    # 预热步数（优先于 warmup_ratio）
    warmup_steps: int = 0

    # ==================== 精度与显存 ====================
    # 混合精度训练
    fp16: bool = False
    bf16: bool = True

    # 梯度检查点
    gradient_checkpointing: bool = True

    # ==================== 日志与保存 ====================
    # 日志记录步数
    logging_steps: int = 10

    # 评估步数
    eval_steps: int = 100

    # 模型保存步数
    save_steps: int = 500

    # 保存模型数量上限
    save_total_limit: int = 3

    # ==================== 序列长度 ====================
    # 最大序列长度
    max_total_length: int = 2048
```

**详细参数说明：**

| 参数 | 类型 | 默认值 | 详细说明 |
|------|------|--------|----------|
| **基础训练参数** ||||
| `num_train_epochs` | int | 3 | **训练轮数**。完整遍历训练数据的次数。<br>• 数据量大：1-3 轮<br>• 数据量小：3-5 轮<br>• 监控验证集 loss，避免过拟合 |
| `per_device_train_batch_size` | int | 4 | **每 GPU 批次大小**。每次前向传播处理的样本数。<br>• 显存受限时减小此值<br>• 配合梯度累积使用 |
| `gradient_accumulation_steps` | int | 4 | **梯度累积步数**。累积多个小批次的梯度后更新参数。<br>• 等效批次 = batch_size × gradient_accumulation_steps<br>• 例：batch_size=2, accumulation=8 → 等效 batch_size=16 |
| `learning_rate` | float | 2e-4 | **学习率**。控制参数更新步长。<br>• QLoRA 推荐：1e-4 ~ 5e-4<br>• LoRA 推荐：1e-4 ~ 3e-4<br>• 全量微调：1e-5 ~ 5e-5 |
| **优化器参数** ||||
| `optim` | str | "paged_adamw_8bit" | **优化器类型**。<br>• "paged_adamw_8bit"：分页 8-bit Adam，省显存（推荐）<br>• "adamw_torch"：标准 AdamW<br>• "adafactor"：更省显存，但效果可能稍差 |
| `weight_decay` | float | 0.01 | **权重衰减**。L2 正则化系数，防止过拟合。<br>• 通常 0.01-0.1<br>• 数据量大时可增大 |
| `max_grad_norm` | float | 1.0 | **梯度裁剪**。防止梯度爆炸。<br>• 通常 1.0<br>• 训练不稳定时可减小到 0.3 |
| **学习率调度** ||||
| `lr_scheduler_type` | str | "cosine" | **学习率调度器**。<br>• "cosine"：余弦退火，平滑降低（推荐）<br>• "linear"：线性降低<br>• "constant"：恒定不变<br>• "polynomial"：多项式衰减 |
| `warmup_ratio` | float | 0.03 | **预热比例**。训练开始时学习率从 0 逐渐增加到设定值。<br>• 通常 0.03-0.1<br>• 有助于稳定训练初期 |
| **精度与显存** ||||
| `fp16` | bool | False | **FP16 混合精度**。使用 16-bit 浮点数计算。<br>• 省显存，加速训练<br>• 可能存在精度损失 |
| `bf16` | bool | True | **BF16 混合精度**。使用 Brain Float 16。<br>• 比 FP16 更大动态范围<br>• 推荐 A100/H100/RTX 3090+ 使用<br>• fp16 和 bf16 通常只开一个 |
| `gradient_checkpointing` | bool | True | **梯度检查点**。以计算换显存。<br>• 重计算激活值而非存储<br>• 节省 30-50% 显存<br>• 训练速度降低 20-30% |
| **日志与保存** ||||
| `logging_steps` | int | 10 | **日志记录频率**。每隔多少步记录一次 loss。 |
| `eval_steps` | int | 100 | **评估频率**。每隔多少步在验证集上评估。 |
| `save_steps` | int | 500 | **保存频率**。每隔多少步保存检查点。 |
| `save_total_limit` | int | 3 | **检查点数量上限**。只保留最近的 N 个检查点，节省磁盘空间。 |
| **序列长度** ||||
| `max_total_length` | int | 2048 | **最大序列长度**。输入 + 输出的总 token 数。<br>• 影响显存占用<br>• 根据对话长度调整<br>• 过长会截断 |

#### 6.2.5 推荐配置组合

**配置一：最省显存（单卡 12GB）**
```python
per_device_train_batch_size = 1
gradient_accumulation_steps = 16
gradient_checkpointing = True
max_total_length = 1024
optim = "paged_adamw_8bit"
```

**配置二：平衡配置（单卡 24GB）**
```python
per_device_train_batch_size = 2
gradient_accumulation_steps = 8
gradient_checkpointing = True
max_total_length = 2048
optim = "paged_adamw_8bit"
```

**配置三：性能优先（多卡 80GB）**
```python
per_device_train_batch_size = 8
gradient_accumulation_steps = 2
gradient_checkpointing = False
max_total_length = 4096
optim = "adamw_torch"
```

### 6.3 训练过程监控

```bash
# 启动 TensorBoard
tensorboard --logdir output/logs --port 6006

# 访问 http://localhost:6006 查看训练曲线
```

**关注指标：**

- **Loss**：训练损失，应持续下降
- **Learning Rate**：学习率变化曲线
- **Eval Loss**：验证损失，用于判断过拟合

**判断训练状态：**

| 现象 | 可能原因 | 解决方案 |
|------|----------|----------|
| Loss 持续下降，Eval Loss 上升 | 过拟合 | 减少训练轮数、增加数据、降低 LoRA rank |
| Loss 不下降 | 学习率太小或数据问题 | 增大学习率、检查数据格式 |
| Loss 震荡剧烈 | 学习率太大 | 降低学习率、增加 batch size |
| Loss 变 NaN | 梯度爆炸 | 降低学习率、启用梯度裁剪 |

### 6.4 模型评估

```bash
# 运行评估脚本
python scripts/evaluate.py --lora-path output/checkpoints/final

# 查看评估结果
# - BLEU 分数: 衡量生成文本与参考文本的相似度
# - ROUGE 分数: 衡量召回率
# - 人工评估: 回复的专业性和共情度
```

### 6.5 模型对比评估

训练完成后，需要对比基座模型和微调后模型的效果，以验证微调的价值。

#### 6.5.1 使用对比评估脚本

```bash
# 基本用法（推荐）
python scripts/compare_models.py --lora-path output/checkpoints/final

# 快速测试（限制样本数）
python scripts/compare_models.py --lora-path output/checkpoints/final --max-samples 50

# 保存详细对比结果到文件
python scripts/compare_models.py --lora-path output/checkpoints/final --save-details
```

#### 6.5.2 评估维度说明

| 指标 | 说明 | 期望变化 |
|------|------|----------|
| **ROUGE-1/2/L** | 文本相似度指标 | 微调后略有提升（+20%~+50%） |
| **平均回复长度** | 模型回复的字数 | 微调后更接近参考长度 |
| **专业术语覆盖率** | 心理咨询关键词使用比例 | 微调后显著提升（+10%~+30%） |
| **不推荐词使用率** | 机械性AI表达的比例 | 微调后显著降低 |

#### 6.5.3 对比报告示例

```
============================================================
模型对比评估报告
============================================================

模型配置:
  娡型类型      平均回复长度    专业术语覆盖率
  基座模型       42.3字          12.5%
  微调模型       78.5字          56.8%

ROUGE 指标对比:
  指标        基座模型    微调模型    提升
  ROUGE-1    0.234      0.312      +33.4%
  ROUGE-L    0.189      0.267      +41.3%

回复样例对比 (#1):
------------------------------------------------------------
用户输入: 我最近总是感觉很焦虑，不知道该怎么办...

【基座模型】 作为一个AI，我不能给出医疗建议...建议您咨询专业心理医生...
【微调模型】 我理解你现在的焦虑感受...你愿意多和我聊聊是什么让你感到焦虑的吗？我们可以一起找到应对的方法。
```

#### 6.5.4 埥看详细对比结果

如果使用了 `--save-details` 参数，结果会保存到 `output/model_comparison.json`，包含：
- 宯整的指标数据
- 所有样本的回复对比
- 可用于后续分析

---

## 七、数据增强说明

### 7.1 增强策略

本项目的数据增强模块（`scripts/augmentation.py`）提供以下策略：

| 策略 | 说明 | 效果 |
|------|------|------|
| **paraphrase** | 同义改写用户输入 | 增加输入多样性 |
| **enhance** | 优化咨询师回复 | 提升回复质量 |
| **scenario** | 场景扩展（如青少年、职场等） | 增加场景覆盖 |
| **clean** | 清理和标准化回复 | 提升数据质量 |

### 7.2 支持的 API 提供商

数据增强模块支持多种大语言模型 API：

| API 类型 | 提供商 | 推荐模型 | 特点 |
|----------|--------|----------|------|
| **GLM** | 智谱 AI | glm-4.7 | 中文理解能力强，推荐使用 |
| **Qwen** | 阿里云 DashScope | qwen-plus | 性价比高，响应快速 |

### 7.3 使用方法

#### 7.3.1 使用 GLM API（推荐）

```python
from scripts.augmentation import DataAugmenter, GLMAPI, create_augmenter

# 方式一：使用便捷函数（推荐）
augmenter = create_augmenter(
    api_type="glm",  # 使用 GLM API
    augment_ratio=0.3,
    strategies=["paraphrase", "enhance"]
)

# 方式二：直接创建 GLMAPI 实例
api = GLMAPI(api_key="your-api-key", model="glm-4.7")
augmenter = DataAugmenter(api_type="glm", api_key="your-api-key")

# 方式三：使用 DataAugmenter 类
augmenter = DataAugmenter(
    api_type="glm",
    model="glm-4.7",  # 可选：glm-4, glm-4-flash, glm-4-plus, glm-4.7
    augment_ratio=0.3,
    strategies=["paraphrase", "enhance"]
)

# 执行增强
augmented_data = augmenter.augment(data, verbose=True)

# 查看统计信息
print(augmenter.get_stats())
```

#### 7.3.2 使用 Qwen API

```python
from scripts.augmentation import DataAugmenter, QwenAPI, create_augmenter

# 方式一：使用便捷函数
augmenter = create_augmenter(
    api_type="qwen",
    augment_ratio=0.3,
    strategies=["paraphrase", "enhance"]
)

# 方式二：直接创建
augmenter = DataAugmenter(
    api_type="qwen",
    model="qwen-plus",  # 可选：qwen-turbo, qwen-plus, qwen-max
    api_key="your-dashscope-api-key",
    augment_ratio=0.3
)

# 执行增强
augmented_data = augmenter.augment(data, verbose=True)
```

### 7.4 命令行使用

```bash
# 在数据处理时启用增强
python scripts/process_data.py --mode multi --augment --augment-ratio 0.3

# 指定增强策略
python scripts/process_data.py --mode multi --augment --strategies paraphrase,enhance
```

### 7.5 API 配置

#### 7.5.1 GLM API（智谱 AI）配置

```bash
# 设置环境变量
export ZHIPUAI_API_KEY="your-api-key"

# 或在代码中传入
augmenter = DataAugmenter(api_type="glm", api_key="your-api-key")
```

**获取 GLM API Key：**
1. 访问 [智谱 AI 开放平台](https://open.bigmodel.cn/)
2. 注册/登录账号
3. 进入控制台创建 API Key

**GLM 模型选择建议：**

| 模型 | 特点 | 适用场景 |
|------|------|----------|
| **glm-4.7** | 最新版本，性能最佳 | 推荐默认使用 |
| glm-4-plus | 高性能版本 | 复杂任务 |
| glm-4-flash | 快速响应 | 简单任务、批量处理 |
| glm-4 | 标准版本 | 通用场景 |

#### 7.5.2 Qwen API（阿里云 DashScope）配置

```bash
# 设置环境变量
export DASHSCOPE_API_KEY="your-api-key"

# 或在代码中传入
augmenter = DataAugmenter(api_type="qwen", api_key="your-api-key")
```

**获取 Qwen API Key：**
1. 访问 [阿里云 DashScope 控制台](https://dashscope.console.aliyun.com/)
2. 注册/登录阿里云账号
3. 创建 API Key

### 7.6 增强效果示例

```python
# 原始数据
original = {
    "input": "最近工作压力很大，经常失眠",
    "response": "我理解你的感受，建议你尝试放松一下。"
}

# 同义改写后
paraphrased = {
    "input": "这段时间工作上的事情让我喘不过气，晚上总是睡不着",
    "response": "我理解你的感受，建议你尝试放松一下。"
}

# 回复增强后
enhanced = {
    "input": "最近工作压力很大，经常失眠",
    "response": "我能感受到你现在承受着很大的压力。失眠确实会让人感到疲惫和焦虑。建议你可以尝试一些放松的方法，比如睡前做深呼吸练习，或者听一些轻柔的音乐。如果情况持续，也可以考虑寻求专业帮助。"
}
```

---

## 八、常见问题

### 8.1 显存不足 (OOM)

**问题**：训练时出现 `CUDA out of memory` 错误

**解决方案**：

1. **减小批次大小**：
   ```bash
   python scripts/train.py --batch-size 1
   ```

2. **减小序列长度**：在 `config/config.py` 中设置：
   ```python
   max_total_length: int = 1024
   ```

3. **启用梯度检查点**：在训练脚本中添加：
   ```python
   training_args.gradient_checkpointing = True
   ```

4. **使用 CPU offload**：
   ```python
   training_args.optim = "paged_adamw_8bit"
   ```

### 8.2 训练速度慢

**问题**：训练速度太慢，一个 epoch 需要很长时间

**解决方案**：

1. **确保使用 4-bit 量化**（默认启用）
2. **增大梯度累积**：
   ```python
   gradient_accumulation_steps: int = 8
   ```
3. **使用更快的优化器**：
   ```python
   optim: str = "paged_adamw_8bit"
   ```
4. **减少日志频率**：
   ```python
   logging_steps: int = 50
   save_steps: int = 1000
   ```

### 8.3 模型效果不佳

**问题**：微调后模型效果没有明显提升

**解决方案**：

1. **检查数据质量**：
   ```bash
   python scripts/process_data.py --explore-only
   ```

2. **增加训练数据量**：使用数据增强扩充数据

3. **调整超参数**：
   - 增加训练轮次：`--epochs 5`
   - 调整学习率：`--learning-rate 1e-4`
   - 增大 LoRA rank：`r = 32`

4. **检查 System Prompt**：确保系统提示词清晰明确

### 8.4 模型加载失败

**问题**：加载模型时报错

**解决方案**：

1. **检查模型路径**：确保 `base_model_path` 正确
2. **检查网络连接**：如使用远程模型，确保能访问 Hugging Face
3. **使用镜像**：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

### 8.5 数据增强 API 调用失败

**问题**：数据增强时 API 调用失败

**解决方案**：

#### GLM API 问题排查

1. **检查 API Key**：确保 `ZHIPUAI_API_KEY` 正确设置
   ```bash
   # 检查环境变量
   echo $ZHIPUAI_API_KEY

   # 或在代码中验证
   from scripts.augmentation import GLMAPI
   api = GLMAPI()
   print(f"API 可用: {api.is_available()}")
   ```

2. **检查账户余额**：登录 [智谱 AI 控制台](https://open.bigmodel.cn/) 查看余额

3. **检查网络**：确保能访问 `open.bigmodel.cn`

4. **常见错误码**：
   - `401 Unauthorized`：API Key 无效或过期
   - `429 Too Many Requests`：请求频率超限，稍后重试
   - `500/502/503`：服务器错误，自动重试机制会处理

#### Qwen API 问题排查

1. **检查 API Key**：确保 `DASHSCOPE_API_KEY` 正确设置
2. **检查账户余额**：确保 API 有足够的调用额度
3. **检查网络**：确保能访问阿里云 API
4. **降低并发**：减少同时请求的数量

### 8.6 如何选择 API 提供商

| 对比项 | GLM (智谱 AI) | Qwen (阿里云) |
|--------|---------------|---------------|
| 中文理解 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 响应速度 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 价格 | 相近 | 相近 |
| 免费额度 | 有 | 有 |
| 推荐场景 | 心理咨询等中文任务 | 通用场景 |

**建议**：心理咨询对话增强推荐使用 GLM API (GLM-4.7)，中文理解和生成能力更强。

---

## 附录

### A. 依赖清单

```txt
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

# 可视化
gradio>=4.0.0
tensorboard>=2.15.0

# 工具库
tqdm>=4.66.0
requests>=2.31.0
```

### B. 参考资料

- [QLoRA 论文](https://arxiv.org/abs/2305.14314)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [Qwen2.5 文档](https://github.com/QwenLM/Qwen2.5)
- [智谱 AI GLM 文档](https://open.bigmodel.cn/dev/api)
- [PEFT 文档](https://huggingface.co/docs/peft)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [BitsAndBytes 文档](https://github.com/TimDettmers/bitsandbytes)

---

> 项目持续更新中，如有问题欢迎提 Issue 或 PR。
