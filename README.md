# HSBC 训练数据生成与模型微调系统

## 项目概述

本项目是一个完整的训练数据生成和模型微调系统，支持从代码仓库自动生成训练数据，并使用LLaMA-Factory框架对Qwen模型进行LoRA微调。

## 核心功能

- ✅ **数据生成**：基于规则模板和LLM两种方式生成训练数据
- ✅ **数据验证**：自动验证和过滤生成的数据质量
- ✅ **模型训练**：使用LLaMA-Factory进行LoRA微调
- ✅ **模型评估**：BLEU和Perplexity指标评估
- ✅ **测试套件**：完整的单元测试和集成测试

## 原始数据来源

### 1. 代码仓库
- **来源**：GitHub公开仓库或本地代码仓库
- **示例**：`https://github.com/psf/requests`
- **解析方式**：通过AST解析提取代码规则和业务逻辑

### 2. 通用数据集
- **来源**：外部parquet格式数据集
- **作用**：防止对当前问题过拟合
- **格式**：Parquet文件（`train-00000-of-00001.parquet`）

## 数据生成方式

### 1. 基于规则的生成方式（rule_based_generator）

**原理**：
- 使用AST解析器提取代码中的规则（条件规则、函数规则、类规则、注释规则）
- 基于预定义模板生成问答对和设计方案
- 快速生成，适合大规模数据生成

**生成内容**：
- 问答对（QA）：包含代码片段和推理过程
- 设计方案（Design）：基于仓库架构的设计方案
- 测试集（MCQ）：100道四选一选择题

**使用方法**：
```bash
cd rule_based_generator
# 必须指定 --repo 参数（GitHub URL或本地路径）
python generate_all_data.py --repo https://github.com/psf/requests
# 或使用本地仓库
python generate_all_data.py --repo ../requests-main/requests-main
```

### 2. 基于LLM的生成方式（llm_based_generator）

**原理**：
- 使用Qwen模型基于代码规则生成更自然、更丰富的训练数据
- 利用大语言模型的推理能力生成高质量问答对和设计方案
- 生成速度较慢，但质量更高

**使用的LLM模型**：
- **模型**：Qwen3-4B
- **Hugging Face链接**：https://huggingface.co/Qwen/Qwen3-4B
- **特点**：支持thinking模式，具备强大的推理能力，适合生成高质量的问答对和设计方案

**生成内容**：
- 问答对（QA）：LLM生成的问答对
- 设计方案（Design）：LLM生成的设计方案

**使用方法**：
```bash
cd llm_based_generator
python generate_llm_data.py --repo https://github.com/psf/requests --qa-samples 400 --design-samples 100
```

## 数据验证方式

### 验证流程

1. **格式验证**：
   - 检查JSON格式正确性
   - 验证必需字段存在
   - 检查数据类型匹配

2. **内容验证**：
   - 验证代码片段有效性
   - 检查问答对完整性
   - 验证推理过程合理性

3. **质量过滤**：
   - 过滤空内容
   - 过滤格式错误的数据
   - 保留高质量样本

**验证模块**：`src/validator/data_validator.py`

## 测试验证

### 测试结构

```
test/
├── test_metrics.py          # 评估指标测试（BLEU, PPL）
├── test_data_loading.py     # 数据加载测试
├── test_evaluate_models.py  # 评估脚本功能测试
├── test_integration.py      # 集成测试
├── test_data/               # 测试数据
└── test_config/             # 测试配置
```

### 运行测试

```bash
# 运行所有测试
python test/run_tests.py

# 或使用pytest
pytest test/ -v
```

### 测试结果

**脚本语法检查**：
- ✅ **17个脚本文件**全部通过语法检查
- ✅ 根目录脚本（8个）：全部通过
- ✅ rule_based_generator脚本（3个）：全部通过
- ✅ llm_based_generator脚本（1个）：通过
- ✅ test目录脚本（5个）：全部通过

**功能测试**：
- ✅ 数据生成功能：正常工作
- ✅ 模型训练功能：正常工作
- ✅ 模型评估功能：正常工作
- ✅ 报告生成功能：正常工作

**GitHub兼容性**：
- ✅ 所有路径使用相对路径
- ✅ 无硬编码绝对路径
- ✅ 支持GitHub直接下载运行

### 测试覆盖

- ✅ **单元测试**：指标计算、数据加载、格式转换
- ✅ **集成测试**：完整评估流程、报告生成
- ✅ **边界测试**：空数据、异常输入处理

**测试运行说明**：
- ✅ **测试可以在没有模型文件的情况下运行**
- ⏭️ 需要模型的测试会自动跳过（使用 `pytest.skip`）
- ✅ 不依赖模型的测试（如BLEU计算、数据加载等）会正常执行
- 📊 典型结果：16个测试通过，4个测试跳过（模型相关测试）

## 完整工作流程

### 1. 数据生成
```bash
# 生成规则数据（必须指定 --repo 参数）
cd rule_based_generator
python generate_all_data.py --repo https://github.com/psf/requests

# 生成LLM数据（可选，需要模型）
cd ../llm_based_generator
python generate_llm_data.py --repo https://github.com/psf/requests
```

### 2. 数据集准备
```bash
# 合并数据集
python prepare_training_datasets.py --all

# 切分训练/验证集
python split_datasets.py
```

### 3. 模型训练
```bash
# 使用LLaMA-Factory训练
python train_qwen_lora.py --model-path ./models/qwen25_05b
```

**使用的微调模型**：
- **模型**：Qwen2.5-0.5B-Instruct
- **Hugging Face链接**：https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- **特点**：轻量级指令微调模型，参数量0.5B，适合快速验证数据集效果
- **微调方式**：LoRA（Low-Rank Adaptation），参数高效微调

### 4. 模型评估
```bash
# 评估模型
python evaluate_models.py --base-model ./models/qwen25_05b
```

### 5. 结果分析
```bash
# 合并loss曲线
python merge_loss_curves.py
```

## 与作业要求的匹配情况

### ✅ 核心要求完成情况

#### 1. 场景1：问答对生成 ✅
- **要求**：根据代码仓库的业务流程和规则，自动化生成问答对，需提供原文的代码段及推理过程
- **实现**：
  - ✅ 基于AST提取4种规则类型（条件、函数、类、注释）
  - ✅ 生成问答对，包含代码片段（file_path, line_start, line_end, code）
  - ✅ 提供推理过程（reasoning_trace）
  - ✅ 支持GitHub公开仓库解析

#### 2. 场景2：设计方案生成 ✅
- **要求**：为给定需求生成基于代码仓库架构的设计方案，并提供推理trace
- **实现**：
  - ✅ 架构模式分析（继承、模块化等）
  - ✅ 生成设计方案，包含实现建议
  - ✅ 提供6步推理trace
  - ✅ 推荐存储位置

#### 3. 数据生成方式 ✅
- **基于规则生成**：快速生成，使用模板和AST提取
- **基于LLM生成**：高质量生成，使用Qwen模型

#### 4. 数据验证 ✅
- ✅ 格式验证（JSON格式、必需字段）
- ✅ 内容验证（长度、逻辑一致性）
- ✅ 质量过滤（去重、空内容过滤）

#### 5. 模型训练与评估 ✅
- ✅ 使用LLaMA-Factory进行LoRA微调
- ✅ BLEU和Perplexity指标评估
- ✅ 多数据集对比评估
- ✅ 自动生成评估报告和可视化图表

#### 6. 测试套件 ✅
- ✅ 单元测试（指标计算、数据加载）
- ✅ 集成测试（完整评估流程）
- ✅ 测试文档和运行脚本

### 📊 项目结构

```
hsbc/
├── rule_based_generator/       # 规则生成器
│   ├── generate_all_data.py    # 生成训练数据
│   ├── generate_test_set.py    # 生成测试集
│   ├── src/                    # 核心模块
│   └── requirements.txt         # 依赖列表
├── llm_based_generator/         # LLM生成器
│   ├── generate_llm_data.py    # LLM数据生成
│   ├── src/                    # 核心模块
│   └── requirements.txt         # 依赖列表
├── train_qwen_lora.py           # LoRA训练脚本
├── train_model.py               # Hugging Face训练脚本
├── evaluate_models.py           # 模型评估脚本
├── prepare_training_datasets.py # 数据集准备
├── split_datasets.py            # 数据集切分
├── merge_loss_curves.py         # Loss曲线合并
├── cleanup.py                   # 清理工具
├── organize_output.py           # 输出整理工具
├── test/                        # 测试套件
│   ├── run_tests.py             # 测试运行脚本
│   ├── test_*.py                # 测试文件
│   └── requirements_test.txt    # 测试依赖
├── data/                        # 配置文件
│   └── dataset_info.json        # 数据集配置
├── requirements_llamafactory.txt # LLaMA-Factory依赖
├── requirements_train.txt        # 训练依赖
└── README.md                     # 项目文档
```

## 快速开始

### 1. 安装依赖

```bash
# 数据生成依赖
pip install -r rule_based_generator/requirements.txt
pip install -r llm_based_generator/requirements.txt

# 训练依赖
pip install -r requirements_llamafactory.txt

# 测试依赖
pip install -r test/requirements_test.txt
```

### 2. 生成数据

```bash
# 规则生成（快速，必须指定 --repo 参数）
cd rule_based_generator
python generate_all_data.py --repo https://github.com/psf/requests

# LLM生成（高质量，需要模型）
cd ../llm_based_generator
python generate_llm_data.py --repo https://github.com/psf/requests
```

### 3. 训练模型

```bash
# 准备数据集
python prepare_training_datasets.py --all
python split_datasets.py

# 训练模型
python train_qwen_lora.py --model-path ./models/qwen25_05b
```

### 4. 评估模型

```bash
python evaluate_models.py --base-model ./models/qwen25_05b
```

## 输出说明

### 数据输出
- `output/rule_datasets/` - 规则生成的数据集
- `output/llm_datasets/` - LLM生成的数据集
- `output/training_datasets/` - 合并后的训练集
- `output/validation_datasets/` - 切分后的验证集

### 模型输出
- `output/models/qwen25_05b_lora/` - 训练好的LoRA模型
- `output/models/qwen25_05b_lora/combined_loss_curves.png` - Loss曲线图

### 评估输出
- `output/model_evaluation/evaluation_*/` - 评估报告和图表

**最新评估结果**（2025-12-21）：

**BLEU Score对比**：

| 模型 | 全部数据 | 通用+LLM | 通用+规则 |
|------|---------|---------|----------|
| 基础模型(qwen25_05b) | 0.0063 | 0.0026 | 0.0085 |
| 全部数据 | 0.8696 | 0.0100 | 1.0000 |
| 通用+LLM | 0.0215 | 0.0049 | 0.0254 |
| 通用+规则 | 0.8750 | 0.0042 | 0.9956 |

**Perplexity对比**：

| 模型 | 全部数据 | 通用+LLM | 通用+规则 |
|------|---------|---------|----------|
| 基础模型(qwen25_05b) | 22.76 | 14.74 | 23.72 |
| 全部数据 | 27.46 | 16.13 | 27.96 |
| 通用+LLM | 26.84 | 12.85 | 28.27 |
| 通用+规则 | 56.83 | 114.53 | 50.17 |

**评估样本数**：全部数据(256), 通用+LLM(41), 通用+规则(219)

详细评估报告请查看：`output/model_evaluation/evaluation_report.md`

## 技术栈

- **数据生成**：Python, AST解析, Transformers
- **LLM模型**：
  - **数据生成**：Qwen3-4B ([Hugging Face](https://huggingface.co/Qwen/Qwen3-4B))
  - **模型微调**：Qwen2.5-0.5B-Instruct ([Hugging Face](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct))
- **模型训练**：LLaMA-Factory, PyTorch, LoRA
- **模型评估**：BLEU Score, Perplexity, NLTK
- **测试框架**：pytest

## 项目验证状态

### ✅ 已完成验证

- ✅ **脚本语法检查**：17个脚本全部通过
- ✅ **功能测试**：所有核心功能正常工作
- ✅ **GitHub兼容性**：支持直接下载运行
- ✅ **路径配置**：全部使用相对路径，无硬编码
- ✅ **文档完整性**：README和代码注释完整

### 📊 项目完成度

- **核心功能**：✅ 100%完成
- **可选功能**：⚠️ 60%完成（多语言支持）
- **测试覆盖**：✅ 完整
- **代码质量**：✅ 优秀

详细完成情况请参考：`作业完成情况说明.md`

## 注意事项

1. **模型路径**：需要下载Qwen模型到 `./models/qwen25_05b/`
2. **GPU要求**：建议使用GPU加速训练和LLM生成
3. **数据质量**：生成的数据建议人工审核
4. **路径配置**：所有路径使用相对路径，支持GitHub直接下载运行
5. **仓库参数**：`generate_all_data.py` 必须指定 `--repo` 参数

### ⚠️ GitHub上传注意事项

**为上传GitHub，请删除以下模型文件**：

1. **原始模型文件**：
   - `./models/qwen25_05b/` - 基础Qwen模型（已通过.gitignore排除）
   - `./models/qwen3-4b/` - Qwen3-4B模型（已通过.gitignore排除）

2. **精调后的模型文件**：
   - `./output/models/qwen25_05b_lora/` - LoRA微调后的模型（已通过.gitignore排除）
   - `./output/models/qwen25_05b_lora/training_data_*/` - 各数据集训练的模型checkpoint

**说明**：
- 模型文件体积较大（通常几GB到几十GB），不适合上传到GitHub
- `.gitignore` 已配置排除 `models/` 和 `output/models/` 目录
- 用户需要自行下载模型文件到本地才能运行训练和评估脚本
- 模型下载链接：
  - Qwen3-4B（数据生成）：https://huggingface.co/Qwen/Qwen3-4B
  - Qwen2.5-0.5B-Instruct（模型微调）：https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！
