# 基于规则的训练数据生成系统

## 项目简介

本系统用于从代码仓库（本地或GitHub）自动生成高质量训练数据，支持Qwen 2.5/3系列模型的微调。

**核心特点**：基于代码规则提取，生成结构化的训练数据和测试集。

## 功能特性

- ✅ 支持本地代码仓库和GitHub仓库解析
- ✅ 自动提取代码规则（条件规则、函数规则、类规则）
- ✅ 生成训练数据：问答对、设计方案
- ✅ 生成测试集：100道四选一选择题
- ✅ 数据质量验证和过滤
- ✅ JSONL格式输出，兼容Hugging Face

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 生成训练数据

```bash
# 一键生成所有训练数据（每类1000条）
# 注意：必须指定 --repo 参数
python generate_all_data.py --repo https://github.com/psf/requests
# 或使用本地仓库
python generate_all_data.py --repo ./requests-main/requests-main
```

### 3. 生成测试集

```bash
# 生成100道四选一选择题（使用默认仓库路径）
python generate_test_set.py

# 或指定仓库路径
python generate_test_set.py --repo https://github.com/psf/requests
```

## 项目结构

```
rule_based_generator/
├── src/                    # 核心代码模块
│   ├── parser/            # 模块1：仓库解析器
│   │   ├── repository_parser.py  # 解析本地/GitHub仓库
│   │   └── ast_extractor.py      # AST提取规则
│   │
│   ├── generator/         # 模块2：数据生成器
│   │   ├── qa_generator.py           # 问答对生成器
│   │   ├── design_generator.py       # 设计方案生成器
│   │   ├── rule_type_generator.py    # 规则类型生成器
│   │   └── mcq_generator.py          # 选择题生成器
│   │
│   ├── validator/         # 模块3：数据验证器
│   │   └── data_validator.py         # 数据质量验证
│   │
│   └── output/            # 模块4：输出处理器
│       └── output_processor.py       # JSONL格式输出
│
├── generate_all_data.py   # 一键生成训练数据脚本
├── generate_test_set.py    # 生成测试集脚本
├── check_data_types.py     # 检查数据类型工具
├── view_data.py            # 查看训练数据工具
├── view_test_set.py        # 查看测试集工具
├── requirements.txt        # 依赖列表
└── README.md              # 本文档
```

## 使用方法

### 生成训练数据

```bash
# 使用默认参数（每类1000条）
python generate_all_data.py

# 自定义参数
python generate_all_data.py --samples-per-type 1000 --design-samples 1000
```

### 生成测试集

```bash
# 生成100道选择题
python generate_test_set.py

# 自定义题目数量
python generate_test_set.py --num-questions 100
```

### 查看数据

```bash
# 查看数据类型统计
python check_data_types.py

# 查看训练数据
python view_data.py output/datasets/conditional_rule.jsonl

# 查看测试集
python view_test_set.py output/test_set.jsonl
```

## 输出文件

### 训练数据（output/datasets/）

- `conditional_rule.jsonl` - 条件规则数据集
- `function_rule.jsonl` - 函数规则数据集
- `class_rule.jsonl` - 类规则数据集
- `comment_rule.jsonl` - 注释规则数据集
- `design.jsonl` - 设计方案数据集
- `all_data.jsonl` - 合并所有数据的文件

### 测试集（output/）

- `test_set.jsonl` - 选择题测试集（100道四选一）

## 数据格式

### 训练数据格式

```json
{
  "type": "qa",
  "question": "What is the rule for timeout handling?",
  "answer": "Requests allows setting timeouts...",
  "code_snippets": [...],
  "reasoning_trace": "Step 1: Locate timeout...",
  "metadata": {
    "rule_type": "conditional_rule",
    "language": "python"
  }
}
```

### 测试集格式

```json
{
  "type": "mcq",
  "question": "What is the business rule for handling timeout?",
  "options": {
    "A": "The system enforces timeouts...",
    "B": "The condition is always false...",
    "C": "The system ignores the condition...",
    "D": "The condition is only checked..."
  },
  "correct_answer": "A",
  "explanation": "The correct answer explains...",
  "code_snippets": [...],
  "metadata": {
    "rule_type": "conditional_rule",
    "language": "python"
  }
}
```

## 规则类型说明

### 训练数据规则类型

1. **conditional_rule** - 条件规则（从if语句提取）
2. **function_rule** - 函数规则（从函数定义提取）
3. **class_rule** - 类规则（从类定义提取）
4. **comment_rule** - 注释规则（从注释提取）
5. **design** - 设计方案（基于仓库架构生成）

### 测试集题目类型

- 基于上述规则类型生成选择题
- 每道题包含4个选项，1个正确答案，3个干扰项

## 参数说明

### generate_all_data.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--repo` | 无（必须指定） | 仓库路径（GitHub URL或本地路径） |
| `--output-dir` | `output/datasets` | 输出目录 |
| `--samples-per-type` | 1000 | 每种规则类型的样本数量 |
| `--design-samples` | 1000 | 设计方案样本数量 |
| `--max-files` | 100 | 最大解析文件数 |

### generate_test_set.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--repo` | `./requests-main/requests-main` | 仓库路径（GitHub URL或本地路径） |
| `--output` | `output/test_set.jsonl` | 输出文件路径 |
| `--num-questions` | 100 | 题目数量 |
| `--max-files` | 100 | 最大解析文件数 |

## 使用示例

### 示例1：生成完整数据集

```bash
# 生成训练数据
python generate_all_data.py

# 生成测试集
python generate_test_set.py
```

### 示例2：快速测试

```bash
python generate_all_data.py --samples-per-type 10 --design-samples 5
python generate_test_set.py --num-questions 10
```

### 示例3：使用GitHub仓库

```bash
python generate_all_data.py --repo https://github.com/psf/requests --max-files 50
```

## 性能

- **解析时间**: ~5-10秒（100个文件）
- **生成训练数据**: ~1-2分钟（每类1000条）
- **生成测试集**: ~10-20秒（100道题）
- **总时间**: ~2-3分钟（完整生成）

## 注意事项

1. 首次使用建议先用小数量测试
2. GitHub API可能有速率限制，建议使用本地仓库
3. 生成的数据需要人工审核以确保质量
4. 可以根据需要调整生成参数

## 许可证

MIT License
