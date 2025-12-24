# 基于LLM的训练数据生成系统

## 项目简介

本系统使用Qwen模型从代码仓库自动生成高质量训练数据，支持Qwen 2.5/3系列模型的微调。

**核心特点**：基于LLM生成，利用大语言模型的推理能力生成更自然、更丰富的训练数据。

## 功能特性

- ✅ 支持本地代码仓库和GitHub仓库解析
- ✅ 使用Qwen模型生成问答对和设计方案
- ✅ 自动提取代码规则（条件规则、函数规则、类规则）
- ✅ 生成训练数据：问答对、设计方案
- ✅ 数据质量验证和过滤
- ✅ JSONL格式输出，兼容Hugging Face

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 生成训练数据

```bash
# 使用默认参数
python generate_llm_data.py

# 自定义参数
python generate_llm_data.py --qa-samples 100 --design-samples 50
```

## 项目结构

```
llm_based_generator/
├── src/                    # 核心代码模块
│   ├── parser/            # 模块1：仓库解析器
│   │   ├── repository_parser.py  # 解析本地/GitHub仓库
│   │   └── ast_extractor.py      # AST提取规则
│   │
│   ├── generator/         # 模块2：LLM数据生成器
│   │   └── llm_generator.py      # LLM生成器
│   │
│   ├── validator/         # 模块3：数据验证器
│   │   └── data_validator.py     # 数据质量验证
│   │
│   └── output/            # 模块4：输出处理器
│       └── output_processor.py   # JSONL格式输出
│
├── generate_llm_data.py   # 生成训练数据脚本
├── requirements.txt        # 依赖列表
└── README.md              # 本文档
```

## 使用方法

### 生成训练数据

```bash
# 使用默认参数（默认使用GitHub仓库: https://github.com/psf/requests）
python generate_llm_data.py

# 使用自定义GitHub仓库
python generate_llm_data.py --repo https://github.com/owner/repo

# 使用本地仓库
python generate_llm_data.py --repo ./your_local_repo

# 自定义参数
python generate_llm_data.py --qa-samples 100 --design-samples 50 --max-files 100

# 只生成问答对
python generate_llm_data.py --no-design

# 只生成设计方案
python generate_llm_data.py --no-qa
```

## 输出文件

### 训练数据（output/llm_datasets/）

- `llm_qa.jsonl` - LLM生成的问答对数据集
- `llm_design.jsonl` - LLM生成的设计方案数据集
- `llm_all_data.jsonl` - 合并所有数据的文件

## 参数说明

### generate_llm_data.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--repo` | `./test_repo` | 仓库路径（本地路径或GitHub URL） |
| `--model-path` | 从环境变量获取 | LLM模型路径（默认从`QWEN_MODEL_PATH`环境变量获取，或使用`models/qwen3-4b`） |
| `--output-dir` | `output/llm_datasets` | 输出目录 |
| `--qa-samples` | 400 | 问答对样本数量 |
| `--design-samples` | 100 | 设计方案样本数量 |
| `--max-files` | 100 | 最大解析文件数 |
| `--device` | 自动选择 | 设备（cuda/cpu） |

## 环境配置

### 1. 设置模型路径（可选）

如果模型不在默认位置，可以通过环境变量设置：

```bash
# Linux/Mac
export QWEN_MODEL_PATH=/path/to/your/qwen-model

# Windows
set QWEN_MODEL_PATH=D:\path\to\your\qwen-model
```

或者在运行时指定：

```bash
python generate_llm_data.py --model-path /path/to/your/qwen-model
```

### 2. 使用GitHub仓库

```bash
python generate_llm_data.py --repo https://github.com/psf/requests
```

## 注意事项

1. **首次使用需要安装依赖**：
   ```bash
   pip install transformers torch accelerate pandas pyarrow
   ```

2. **模型路径**：
   - 如果未指定`--model-path`，会从环境变量`QWEN_MODEL_PATH`获取
   - 如果环境变量也未设置，会尝试使用`models/qwen3-4b`相对路径
   - 建议通过环境变量或命令行参数明确指定模型路径

3. **Qwen3-4b模型**：
   - 需要较大内存（建议16GB+）
   - 建议使用GPU加速（CUDA）
   - 如果没有GPU，可以使用较小的模型（如qwen3-0.6b）

4. **生成速度**：
   - LLM生成比基于规则的方法慢
   - 400个问答对大约需要30-60分钟（取决于硬件）
   - 请耐心等待

5. **JSON格式问题**：
   - 系统已改进JSON解析逻辑，能自动处理常见的格式问题
   - 如果遇到解析失败，会自动回退到文本提取模式

## 许可证

MIT License
