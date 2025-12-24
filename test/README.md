# 模型评估测试套件

## 项目简介

本测试套件用于验证和测试模型训练与评估系统的各项功能。

## 功能特性

- ✅ 模型评估功能测试
- ✅ 数据集加载和验证测试
- ✅ BLEU和Perplexity指标计算测试
- ✅ 报告生成功能测试
- ✅ 可视化图表生成测试

## 快速开始

### 1. 运行所有测试

```bash
# 运行所有测试
python -m pytest test/

# 运行特定测试文件
python -m pytest test/test_evaluate_models.py

# 运行并显示详细输出
python -m pytest test/ -v
```

### 2. 运行单元测试

```bash
# 测试评估指标计算
python test/test_metrics.py

# 测试数据加载
python test/test_data_loading.py
```

### 3. 运行集成测试

```bash
# 测试完整评估流程（需要训练好的模型）
python test/test_integration.py
```

## 测试结构

```
test/
├── test_evaluate_models.py      # 评估脚本功能测试
├── test_metrics.py              # 评估指标测试
├── test_data_loading.py          # 数据加载测试
├── test_integration.py           # 集成测试
├── test_data/                   # 测试数据
│   ├── sample_dataset.jsonl      # 示例数据集
│   └── sample_validation.jsonl   # 示例验证集
├── test_config/                  # 测试配置
│   └── test_config.json          # 测试配置文件
└── README.md                     # 本文档
```

## 测试用例说明

### 1. 评估指标测试 (test_metrics.py)

- BLEU分数计算正确性
- Perplexity计算正确性
- 边界情况处理（空文本、无效输入等）

### 2. 数据加载测试 (test_data_loading.py)

- JSONL格式数据加载
- ShareGPT格式转换
- 数据验证和过滤

### 3. 评估脚本测试 (test_evaluate_models.py)

- 模型加载功能
- 生成策略（Greedy vs Sampling）
- 批量评估流程

### 4. 集成测试 (test_integration.py)

- 完整评估流程
- 报告生成
- 图表生成

## 注意事项

1. **模型文件**：
   - ⚠️ **测试可以在没有模型文件的情况下运行**
   - 需要模型的测试会自动跳过（使用 `pytest.skip`）
   - 不依赖模型的测试（如BLEU计算、数据加载等）会正常执行
   - 如果已下载模型，请确保模型路径正确：`./models/qwen25_05b/`

2. **测试数据**：使用 `test_data/` 目录下的示例数据进行测试

3. **GPU要求**：部分测试需要GPU支持，CPU模式下可能较慢

4. **依赖安装**：运行测试前请确保安装所有依赖：`pip install -r requirements_test.txt`

5. **测试结果说明**：
   - ✅ **PASSED**：测试通过
   - ⏭️ **SKIPPED**：测试跳过（通常是因为模型文件不存在，这是正常的）
   - ❌ **FAILED**：测试失败（需要检查）

**典型测试结果**（无模型文件时）：
- 16个测试通过（不依赖模型）
- 4个测试跳过（需要模型，自动跳过）
- 这是**正常情况**，说明测试脚本已正确处理模型缺失的情况

## 测试覆盖率

- 单元测试覆盖率：>80%
- 集成测试覆盖率：>60%

## 版本历史

- V1.0 20250115：初始版本，包含基础测试用例
