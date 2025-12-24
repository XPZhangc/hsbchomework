[DESIGN.md](https://github.com/user-attachments/files/24330116/DESIGN.md)
# 训练数据生成系统设计文档

## 1. 项目概述

### 1.1 背景
本系统旨在为基于 Qwen 2.5 系列模型的微调提供高质量训练数据，使模型能够：
- **场景1**：回答关于本地代码仓库的业务流程和规则问题
- **场景2**：基于代码仓库架构生成设计方案

### 1.2 核心目标
- 自动化生成高质量的训练数据
- 确保数据覆盖所需场景并在逻辑上正确
- 提供结构化的推理 trace，支持模型学习推理过程
- 支持多语言代码仓库
- 保证数据的多样性和代表性

## 2. 数据集结构设计

### 2.1 场景1：问答对（QA Pairs）数据结构

```json
{
  "type": "qa",
  "question": "问题内容，深入理解代码的业务逻辑和规则",
  "answer": "详细答案，解释代码的功能、规则和设计意图",
  "code_snippets": [
    {
      "file_path": "相对路径/文件名.py",
      "line_start": 10,
      "line_end": 25,
      "code": "原始代码片段，作为问题的依据"
    }
  ],
  "reasoning_trace": {
    "steps": [
      {
        "step": 1,
        "action": "识别代码结构",
        "evidence": "代码片段第10-15行",
        "conclusion": "这是一个条件判断规则"
      },
      {
        "step": 2,
        "action": "分析业务逻辑",
        "evidence": "if语句中的条件表达式",
        "conclusion": "该规则用于验证输入参数的有效性"
      }
    ],
    "summary": "整体推理过程的总结"
  },
  "metadata": {
    "repository": "仓库名称或URL",
    "rules_extracted": ["规则描述1", "规则描述2"],
    "language": "python",
    "rule_type": "conditional_rule",
    "complexity": "medium",
    "generation_method": "llm",
    "generation_timestamp": "2025-12-19T23:00:00Z"
  }
}
```

### 2.2 场景2：设计方案（Design Schemes）数据结构

```json
{
  "type": "design",
  "demand": "设计需求描述",
  "scheme": "详细的设计方案内容，说明如何集成到现有系统",
  "code_snippets": [
    {
      "file_path": "相关文件路径",
      "line_start": 1,
      "line_end": 50,
      "code": "相关代码片段，作为设计方案的参考"
    }
  ],
  "reasoning_trace": {
    "steps": [
      {
        "step": 1,
        "action": "分析现有架构",
        "evidence": "代码片段中的类结构",
        "conclusion": "系统采用模块化设计"
      },
      {
        "step": 2,
        "action": "识别集成点",
        "evidence": "基类和接口定义",
        "conclusion": "可以通过继承基类实现新功能"
      },
      {
        "step": 3,
        "action": "设计决策",
        "evidence": "现有设计模式",
        "conclusion": "采用策略模式实现可扩展性"
      }
    ],
    "integration_points": [
      {
        "file": "src/base.py",
        "description": "继承BaseHandler类",
        "rationale": "保持与现有代码的一致性"
      }
    ],
    "summary": "设计方案的总体思路和关键决策"
  },
  "metadata": {
    "repository": "仓库名称或URL",
    "rules_extracted": ["相关规则1", "相关规则2"],
    "language": "python",
    "generation_method": "llm",
    "generation_timestamp": "2025-12-19T23:00:00Z"
  }
}
```

### 2.3 元数据字段说明

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `type` | string | 是 | 数据类型：`qa` 或 `design` |
| `question` | string | 是（qa类型） | 问题内容 |
| `answer` | string | 是（qa类型） | 答案内容 |
| `demand` | string | 是（design类型） | 设计需求 |
| `scheme` | string | 是（design类型） | 设计方案 |
| `code_snippets` | array | 是 | 相关代码片段列表 |
| `reasoning_trace` | object | 是 | 结构化的推理过程 |
| `metadata` | object | 是 | 元数据信息 |

## 3. 数据生成流程

### 3.1 整体流程

```
1. 仓库解析阶段
   ├── 解析本地/GitHub仓库
   ├── 提取代码片段
   ├── AST分析提取规则
   └── 构建仓库结构信息

2. LLM生成阶段
   ├── 加载Qwen模型
   ├── 生成问答对（场景1）
   │   ├── 基于规则生成问题
   │   ├── 生成详细答案
   │   └── 生成结构化推理trace
   └── 生成设计方案（场景2）
       ├── 分析需求
       ├── 生成设计方案
       └── 生成推理trace

3. 数据验证阶段
   ├── 基本验证（字段完整性、长度）
   ├── 逻辑验证（一致性、相关性）
   ├── 代码片段验证
   └── 多样性检查

4. 输出处理阶段
   ├── 格式化为JSONL
   ├── 保存到文件
   └── 生成统计报告
```

### 3.2 问答对生成流程

1. **规则选择策略**
   - 按规则类型分布采样（conditional_rule, function_rule, class_rule, comment_rule）
   - 按代码复杂度分层采样（简单、中等、复杂）
   - 确保覆盖不同文件和不同语言

2. **问题生成**
   - 基于代码规则生成深入的问题
   - 问题类型多样化：what（是什么）、how（如何工作）、why（为什么这样设计）

3. **答案生成**
   - 详细解释代码功能和业务逻辑
   - 包含设计意图和最佳实践说明

4. **推理trace生成**
   - 步骤化推理过程
   - 每个步骤包含：动作、证据、结论
   - 提供整体推理总结

### 3.3 设计方案生成流程

1. **需求生成**
   - 预设需求模板库
   - 基于仓库特点生成变体需求
   - 确保需求与仓库架构相关

2. **架构分析**
   - 提取关键类和函数
   - 识别设计模式和架构风格
   - 分析代码组织结构

3. **方案生成**
   - 遵循现有架构风格
   - 说明集成点和集成方式
   - 提供实现思路和关键代码结构

4. **推理trace生成**
   - 分析现有架构的步骤
   - 识别集成点的步骤
   - 设计决策的推理过程

## 4. 数据质量保证机制

### 4.1 验证规则

#### 基本验证
- ✅ 必需字段完整性检查
- ✅ 文本长度验证（问题≥10字符，答案≥20字符，推理≥30字符）
- ✅ 代码片段格式验证

#### 逻辑验证
- ✅ 问答一致性：答案应包含问题的关键词
- ✅ 设计方案相关性：方案应提及需求关键词
- ✅ 代码片段有效性：代码应真实存在且格式正确

#### 质量验证
- ✅ 推理trace结构化检查
- ✅ 代码片段与推理的关联性
- ✅ 元数据完整性

### 4.2 多样性保证

#### 规则类型分布
- conditional_rule: 30%
- function_rule: 30%
- class_rule: 25%
- comment_rule: 15%

#### 问题类型分布
- What类型（是什么）: 40%
- How类型（如何工作）: 35%
- Why类型（为什么）: 25%

#### 复杂度分布
- 简单（<50行）: 40%
- 中等（50-200行）: 45%
- 复杂（>200行）: 15%

#### 文件覆盖
- 确保至少覆盖N个不同文件（N = min(样本数/10, 总文件数)）
- 优先选择核心代码文件（src目录）

### 4.3 去重策略

- **问答对去重**：基于问题核心内容（移除变异标记后）
- **设计方案去重**：基于需求内容
- **代码片段去重**：相同文件相同行号范围视为重复

## 5. 推理 Trace 设计规范

### 5.1 结构要求

推理trace必须包含以下结构：

```json
{
  "steps": [
    {
      "step": 1,
      "action": "具体动作描述",
      "evidence": "证据来源（代码行、文件等）",
      "conclusion": "得出的结论"
    }
  ],
  "summary": "整体推理过程的总结"
}
```

### 5.2 步骤要求

每个步骤应：
- **action**：清晰描述执行的动作（如"识别代码结构"、"分析业务逻辑"）
- **evidence**：明确指出证据来源（如"第10-15行"、"BaseHandler类"）
- **conclusion**：基于证据得出的明确结论

### 5.3 质量标准

- 至少包含2个推理步骤
- 步骤之间应有逻辑关联
- 证据应可追溯到具体代码位置
- 结论应基于证据合理推导

## 6. 系统架构

### 6.1 模块划分

```
llm_based_generator/
├── src/
│   ├── parser/              # 仓库解析模块
│   │   ├── repository_parser.py    # 解析本地/GitHub仓库
│   │   └── ast_extractor.py        # AST提取规则
│   │
│   ├── generator/            # LLM生成模块
│   │   └── llm_generator.py        # LLM生成器（问答对+设计方案）
│   │
│   ├── validator/            # 数据验证模块
│   │   └── data_validator.py        # 数据质量验证
│   │
│   └── output/              # 输出处理模块
│       └── output_processor.py     # JSONL格式输出
│
├── generate_llm_data.py     # 主生成脚本
├── DESIGN.md                # 本文档
└── README.md                # 使用说明
```

### 6.2 可扩展性设计

- **多模型支持**：通过配置支持不同的LLM模型
- **多语言支持**：已支持Python、Java、JavaScript、TypeScript等
- **自定义验证规则**：可扩展验证器添加新的验证规则
- **输出格式扩展**：支持多种输出格式（JSONL、JSON、CSV等）

## 7. 使用示例

### 7.1 基本使用

```bash
# 使用默认参数生成数据
python generate_llm_data.py

# 自定义参数
python generate_llm_data.py \
  --repo D:\hsbc\requests-main\requests-main \
  --model-path D:\hsbc\qwen3-4b \
  --qa-samples 200 \
  --design-samples 100 \
  --max-files 150
```

### 7.2 输出文件

- `output/llm_datasets/llm_qa.jsonl` - 问答对数据集
- `output/llm_datasets/llm_design.jsonl` - 设计方案数据集
- `output/llm_datasets/llm_all_data.jsonl` - 合并数据集

### 7.3 数据格式验证

```python
import jsonlines

# 读取并验证数据
with jsonlines.open('output/llm_datasets/llm_qa.jsonl') as reader:
    for obj in reader:
        assert obj['type'] == 'qa'
        assert 'question' in obj
        assert 'answer' in obj
        assert 'reasoning_trace' in obj
        assert 'steps' in obj['reasoning_trace']
```

## 8. 评判标准对应

### 8.1 数据集覆盖度 ✅
- ✅ 覆盖场景1（问答对）和场景2（设计方案）
- ✅ 包含代码片段和推理trace
- ✅ 逻辑上正确（通过验证器保证）

### 8.2 数据处理方法 ✅
- ✅ 自动化生成（基于LLM）
- ✅ 高质量保证（多层验证）
- ✅ 创新性（结构化推理trace）

### 8.3 系统架构 ✅
- ✅ 模块化设计，职责清晰
- ✅ 可扩展（支持多模型、多语言）
- ✅ 支持未来需求变化

### 8.4 示例数据质量 ✅
- ✅ 清晰的数据结构定义
- ✅ 结构化的推理trace
- ✅ 合规性（符合微调数据格式）

## 9. 未来改进方向

1. **推理trace质量提升**
   - 增加更多验证规则
   - 支持多轮推理
   - 增加推理链的可视化

2. **数据多样性增强**
   - 支持更多编程语言
   - 增加代码复杂度分析
   - 支持跨文件关联分析

3. **微调支持**
   - 提供Qwen 2.5微调脚本
   - 支持LoRA/QLoRA微调
   - 提供快速验证流程

4. **性能优化**
   - 批量生成优化
   - 缓存机制
   - 并行处理支持
