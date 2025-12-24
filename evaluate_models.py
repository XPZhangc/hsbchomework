#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""评估三个训练好的模型在测试集上的BLEU Score和Perplexity"""

import json
import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from datetime import datetime

# 抑制transformers库的警告信息（如果不需要详细日志）
# 可以通过环境变量 TRANSFORMERS_VERBOSITY=error 来完全抑制警告
if 'TRANSFORMERS_VERBOSITY' not in os.environ:
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# 导入必要的库
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    nltk.download('punkt', quiet=True)
    try:
        from tqdm import tqdm
        HAS_TQDM = True
    except ImportError:
        HAS_TQDM = False
        print("提示: 安装tqdm可以获得更好的进度显示: pip install tqdm")
except ImportError as e:
    print(f"错误: 缺少必要的库: {e}")
    print("请安装: pip install torch transformers peft nltk")
    sys.exit(1)


def load_jsonl_dataset(file_path: Path) -> List[Dict[str, Any]]:
    """加载JSONL格式的数据集"""
    data = []
    if not file_path.exists():
        raise FileNotFoundError(f"数据集文件不存在: {file_path}")
    
    print(f"正在加载数据集: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if item:
                    data.append(item)
            except json.JSONDecodeError:
                continue
    
    print(f"已加载 {len(data)} 条有效数据")
    return data


def convert_to_sharegpt_format(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将原始数据转换为ShareGPT格式（用于提取问题和答案）"""
    sharegpt_data = []
    
    for item in data:
        data_type = item.get('type', '')
        question = item.get('question', '')
        answer = item.get('answer', '')
        
        if not question or not answer:
            continue
        
        conversation = []
        
        if data_type in ['qa', 'design']:
            user_content = question
            code_snippets = item.get('code_snippets', [])
            if code_snippets and len(code_snippets) > 0:
                code_text = "\n相关代码:\n"
                for snippet in code_snippets[:3]:
                    file_path = snippet.get('file_path', '')
                    code = snippet.get('code', '')
                    if code:
                        code_text += f"文件: {file_path}\n```python\n{code}\n```\n\n"
                user_content = code_text + user_content
            
            conversation.append({"role": "user", "content": user_content})
            conversation.append({"role": "assistant", "content": answer})
            sharegpt_data.append({"conversation": conversation})
    
    return sharegpt_data


def format_prompt_for_qwen(conversation: List[Dict[str, str]], tokenizer) -> str:
    """将对话格式化为Qwen模型的输入格式"""
    messages = []
    for msg in conversation:
        role = msg['role']
        content = msg['content']
        if role == 'user':
            messages.append({"role": "user", "content": content})
        elif role == 'assistant':
            messages.append({"role": "assistant", "content": content})
    
    # 使用tokenizer的apply_chat_template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text


def calculate_bleu(reference: str, candidate: str) -> float:
    """计算BLEU分数"""
    try:
        # 分词
        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        
        if len(ref_tokens) == 0 or len(cand_tokens) == 0:
            return 0.0
        
        # 使用smoothing function避免0分
        smoothing = SmoothingFunction().method1
        score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing)
        return score
    except Exception as e:
        print(f"计算BLEU时出错: {e}")
        return 0.0


def calculate_perplexity(model, tokenizer, text: str, device: str) -> float:
    """计算文本的困惑度（Perplexity）"""
    try:
        # 编码文本，限制长度以提高速度
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
        
        if inputs['input_ids'].shape[1] < 2:
            return float('inf')
        
        # 计算loss
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
        
        # PPL = exp(loss)
        ppl = np.exp(loss)
        return ppl
    except Exception as e:
        print(f"  计算PPL时出错: {e}")
        return float('inf')


def evaluate_model(model_path: Path, base_model_path: str, test_data: List[Dict[str, Any]], 
                  device: str, max_samples: int = None, use_sampling: bool = False, 
                  temperature: float = 0.2) -> Dict[str, float]:
    """
    评估单个模型
    
    Args:
        model_path: 模型路径（LoRA adapter路径或基础模型路径）
        base_model_path: 基础模型路径
        test_data: 测试数据
        device: 设备（cuda/cpu）
        max_samples: 最大样本数
        use_sampling: 是否使用采样生成（True）或贪婪解码（False）
        temperature: 采样温度（仅在use_sampling=True时有效）
    """
    """评估单个模型"""
    print(f"\n正在加载模型: {model_path}")
    
    # 加载基础模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 加载LoRA adapter
    if (model_path / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base_model, str(model_path))
        print("已加载LoRA adapter")
    else:
        model = base_model
        print("未找到LoRA adapter，使用基础模型")
    
    model.eval()
    model.to(device)
    
    # 转换数据格式
    sharegpt_data = convert_to_sharegpt_format(test_data)
    
    if max_samples:
        sharegpt_data = sharegpt_data[:max_samples]
    
    print(f"开始评估 {len(sharegpt_data)} 个样本...")
    if use_sampling:
        print(f"  生成策略: 采样生成 (temperature={temperature}, top_p=0.8, top_k=20)")
    else:
        print(f"  生成策略: Greedy解码 (不使用temperature，直接选择概率最高的token)")
    
    bleu_scores = []
    ppl_scores = []
    
    # 使用tqdm显示进度条，如果没有安装则使用简单的进度打印
    if HAS_TQDM:
        iterator = tqdm(sharegpt_data, desc="评估进度", unit="样本")
    else:
        iterator = sharegpt_data
    
    for idx, item in enumerate(iterator):
        if not HAS_TQDM and (idx + 1) % 10 == 0:
            print(f"  处理进度: {idx + 1}/{len(sharegpt_data)}")
        
        conversation = item['conversation']
        
        # 提取问题和答案
        user_msg = None
        assistant_msg = None
        for msg in conversation:
            if msg['role'] == 'user':
                user_msg = msg['content']
            elif msg['role'] == 'assistant':
                assistant_msg = msg['content']
        
        if not user_msg or not assistant_msg:
            continue
        
        # 构建输入（只包含用户消息）
        user_conversation = [{"role": "user", "content": user_msg}]
        input_text = format_prompt_for_qwen(user_conversation, tokenizer)
        
        # 生成回答
        try:
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(device)
            
            with torch.no_grad():
                # 根据use_sampling参数决定生成策略
                if use_sampling:
                    # 采样生成：使用temperature, top_p, top_k
                    generate_kwargs = {
                        'max_new_tokens': 256,
                        'do_sample': True,
                        'temperature': temperature,
                        'top_p': 0.8,
                        'top_k': 20,
                        'pad_token_id': tokenizer.eos_token_id
                    }
                    # 将inputs的键值对添加到generate_kwargs中
                    generate_kwargs.update(inputs)
                    outputs = model.generate(**generate_kwargs)
                else:
                    # Greedy decoding（更快）：明确不传递任何采样参数
                    generate_kwargs = {
                        'max_new_tokens': 256,
                        'do_sample': False,
                        'pad_token_id': tokenizer.eos_token_id,
                        'num_beams': 1
                    }
                    # 将inputs的键值对添加到generate_kwargs中
                    generate_kwargs.update(inputs)
                    outputs = model.generate(**generate_kwargs)
            
            generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # 计算BLEU
            bleu = calculate_bleu(assistant_msg, generated_text)
            bleu_scores.append(bleu)
            
            # 计算PPL（在完整对话上，限制长度以提高速度）
            full_conversation_text = format_prompt_for_qwen(conversation, tokenizer)
            # 限制文本长度以提高PPL计算速度
            if len(full_conversation_text) > 2000:
                full_conversation_text = full_conversation_text[:2000]
            ppl = calculate_perplexity(model, tokenizer, full_conversation_text, device)
            if ppl != float('inf') and ppl > 0:
                ppl_scores.append(ppl)
        
        except Exception as e:
            print(f"  样本 {idx} 处理失败: {e}")
            continue
    
    # 清理GPU内存
    del model
    del base_model
    torch.cuda.empty_cache()
    
    # 计算平均值
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    avg_ppl = np.mean(ppl_scores) if ppl_scores else float('inf')
    
    return {
        'bleu': avg_bleu,
        'ppl': avg_ppl,
        'num_samples': len(bleu_scores)
    }


def plot_comparison(results: Dict[str, Dict[str, Dict[str, float]]], output_file: Path):
    """绘制对比图表"""
    # 分离基础模型和LoRA模型
    base_model_name = '基础模型(qwen25_05b)'
    lora_models = [m for m in results.keys() if m != base_model_name]
    
    # 按顺序排列：基础模型在前，然后是LoRA模型
    if base_model_name in results:
        models = [base_model_name] + lora_models
    else:
        models = lora_models
    
    test_sets = ['training_data_all', 'training_data_general_llm', 'training_data_general_rule']
    test_set_names = ['全部数据', '通用+LLM', '通用+规则']
    
    # 准备数据
    bleu_data = {name: [] for name in test_set_names}
    ppl_data = {name: [] for name in test_set_names}
    
    for test_set, test_name in zip(test_sets, test_set_names):
        for model_name in models:
            if test_set in results[model_name]:
                bleu_data[test_name].append(results[model_name][test_set]['bleu'])
                ppl_data[test_name].append(results[model_name][test_set]['ppl'])
            else:
                bleu_data[test_name].append(0)
                ppl_data[test_name].append(float('inf'))
    
    # 创建图表（3个测试集，需要更大的图表）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    x = np.arange(len(models))
    width = 0.25  # 3个测试集，需要更窄的柱子
    
    # BLEU Score对比
    for idx, (test_name, bleu_values) in enumerate(bleu_data.items()):
        # 处理inf值
        bleu_values_clean = [v if v != float('inf') else 0 for v in bleu_values]
        ax1.bar(x + idx * width, bleu_values_clean, width, label=test_name, alpha=0.8)
    
    ax1.set_xlabel('模型', fontsize=12)
    ax1.set_ylabel('BLEU Score', fontsize=12)
    ax1.set_title('BLEU Score对比', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width)  # 3个测试集，调整位置
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Perplexity对比
    for idx, (test_name, ppl_values) in enumerate(ppl_data.items()):
        # 处理inf值，使用log scale
        ppl_values_clean = [v if v != float('inf') and v > 0 else 1e-10 for v in ppl_values]
        ax2.bar(x + idx * width, ppl_values_clean, width, label=test_name, alpha=0.8)
    
    ax2.set_xlabel('模型', fontsize=12)
    ax2.set_ylabel('Perplexity (PPL)', fontsize=12)
    ax2.set_title('Perplexity对比', fontsize=14, fontweight='bold')
    ax2.set_xticks(x + width)  # 3个测试集，调整位置
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n对比图表已保存到: {output_file}")


def generate_markdown_report(results: Dict[str, Dict[str, Dict[str, float]]], output_file: Path):
    """生成Markdown格式的评估报告"""
    models = list(results.keys())
    # 所有模型都在三个测试集上评估
    test_sets = ['training_data_all', 'training_data_general_llm', 'training_data_general_rule']
    test_set_names = ['全部数据', '通用+LLM', '通用+规则']
    
    # 基础模型的所有测试集（与test_sets相同）
    all_test_sets = test_sets
    all_test_set_names = test_set_names
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 模型评估对比报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 评估指标说明\n\n")
        f.write("- **BLEU Score**: 衡量生成文本与参考答案的相似度，范围0-1，越高越好\n")
        f.write("- **Perplexity (PPL)**: 衡量模型对测试数据的困惑度，越低越好\n\n")
        
        f.write("## 评估结果\n\n")
        
        # 所有模型在所有测试集上的对比
        f.write("### 所有模型在测试集上的表现（BLEU Score）\n\n")
        f.write("| 模型 | " + " | ".join(test_set_names) + " |\n")
        f.write("|" + "---|" * (len(test_set_names) + 1) + "\n")
        for model_name in models:
            row = [model_name]
            for test_set in test_sets:
                if test_set in results[model_name]:
                    bleu = results[model_name][test_set]['bleu']
                    row.append(f"{bleu:.4f}")
                else:
                    row.append("N/A")
            f.write("| " + " | ".join(row) + " |\n")
        
        f.write("\n### 所有模型在测试集上的表现（Perplexity）\n\n")
        f.write("| 模型 | " + " | ".join(test_set_names) + " |\n")
        f.write("|" + "---|" * (len(test_set_names) + 1) + "\n")
        for model_name in models:
            row = [model_name]
            for test_set in test_sets:
                if test_set in results[model_name]:
                    ppl = results[model_name][test_set]['ppl']
                    if ppl != float('inf'):
                        row.append(f"{ppl:.2f}")
                    else:
                        row.append("N/A")
                else:
                    row.append("N/A")
            f.write("| " + " | ".join(row) + " |\n")
        
        # 完整对比表（所有模型在所有测试集上）
        f.write("\n### 完整对比\n\n")
        f.write("#### BLEU Score对比\n\n")
        f.write("| 模型 | " + " | ".join(test_set_names) + " |\n")
        f.write("|" + "---|" * (len(test_set_names) + 1) + "\n")
        # 先显示基础模型
        if '基础模型(qwen25_05b)' in results:
            row = ['基础模型(qwen25_05b)']
            for test_set in test_sets:
                if test_set in results['基础模型(qwen25_05b)']:
                    bleu = results['基础模型(qwen25_05b)'][test_set]['bleu']
                    row.append(f"{bleu:.4f}")
                else:
                    row.append("N/A")
            f.write("| " + " | ".join(row) + " |\n")
        # 再显示LoRA模型
        for model_name in models:
            if model_name == '基础模型(qwen25_05b)':
                continue
            row = [model_name]
            for test_set in test_sets:
                if test_set in results[model_name]:
                    bleu = results[model_name][test_set]['bleu']
                    row.append(f"{bleu:.4f}")
                else:
                    row.append("N/A")
            f.write("| " + " | ".join(row) + " |\n")
        
        f.write("\n#### Perplexity对比\n\n")
        f.write("| 模型 | " + " | ".join(test_set_names) + " |\n")
        f.write("|" + "---|" * (len(test_set_names) + 1) + "\n")
        # 先显示基础模型
        if '基础模型(qwen25_05b)' in results:
            row = ['基础模型(qwen25_05b)']
            for test_set in test_sets:
                if test_set in results['基础模型(qwen25_05b)']:
                    ppl = results['基础模型(qwen25_05b)'][test_set]['ppl']
                    if ppl != float('inf'):
                        row.append(f"{ppl:.2f}")
                    else:
                        row.append("N/A")
                else:
                    row.append("N/A")
            f.write("| " + " | ".join(row) + " |\n")
        # 再显示LoRA模型
        for model_name in models:
            if model_name == '基础模型(qwen25_05b)':
                continue
            row = [model_name]
            for test_set in test_sets:
                if test_set in results[model_name]:
                    ppl = results[model_name][test_set]['ppl']
                    if ppl != float('inf'):
                        row.append(f"{ppl:.2f}")
                    else:
                        row.append("N/A")
                else:
                    row.append("N/A")
            f.write("| " + " | ".join(row) + " |\n")
        
        f.write("\n## 详细结果\n\n")
        for model_name in models:
            f.write(f"### {model_name}\n\n")
            # 所有模型都在相同的测试集上评估
            for test_set, test_name in zip(test_sets, test_set_names):
                if test_set in results[model_name]:
                    result = results[model_name][test_set]
                    f.write(f"**{test_name}测试集**:\n")
                    f.write(f"- BLEU Score: {result['bleu']:.4f}\n")
                    if result['ppl'] != float('inf'):
                        f.write(f"- Perplexity: {result['ppl']:.2f}\n")
                    else:
                        f.write(f"- Perplexity: N/A\n")
                    f.write(f"- 评估样本数: {result['num_samples']}\n\n")
    
    print(f"\n评估报告已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='评估训练好的模型')
    parser.add_argument('--base-model', type=str, default='./models/qwen25_05b',
                       help='基础模型路径（默认: ./models/qwen25_05b）')
    parser.add_argument('--models-dir', type=str, default='./output/models/qwen25_05b_lora',
                       help='模型目录（默认: ./output/models/qwen25_05b_lora）')
    parser.add_argument('--test-dir', type=str, default='./output/validation_datasets',
                       help='测试集目录（默认: ./output/validation_datasets）')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='每个测试集的最大样本数（默认: 全部）')
    parser.add_argument('--output', type=str, default='./output/model_evaluation',
                       help='输出目录（默认: ./output/model_evaluation）')
    parser.add_argument('--use-sampling', action='store_true',
                       help='使用sampling生成（默认: 使用greedy decoding，不使用temperature）')
    parser.add_argument('--temperature', type=float, default=0.2,
                       help='Sampling temperature（默认: 0.2，仅在--use-sampling时有效；greedy模式下不使用temperature）')
    
    args = parser.parse_args()
    
    # 验证参数：如果使用sampling，确保temperature > 0
    if args.use_sampling and args.temperature <= 0:
        print("警告: temperature必须大于0，已重置为0.2", file=sys.stderr)
        args.temperature = 0.2
    
    # 打印生成策略信息
    if args.use_sampling:
        print(f"\n生成策略: 采样生成 (temperature={args.temperature})")
    else:
        print(f"\n生成策略: Greedy解码 (不使用temperature)")
    
    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 模型目录
    models_dir = Path(args.models_dir)
    base_model_path = args.base_model
    
    # 测试集
    test_dir = Path(args.test_dir)
    # LoRA模型和基础模型都在所有三个测试集上评估
    test_sets = {
        'training_data_all': test_dir / 'training_data_all_val.jsonl',
        'training_data_general_llm': test_dir / 'training_data_general_llm_val.jsonl',
        'training_data_general_rule': test_dir / 'training_data_general_rule_val.jsonl'
    }
    
    # 所有测试集（与test_sets相同，保留用于兼容性）
    all_test_sets = test_sets
    
    # 模型列表（训练好的LoRA模型）
    model_dirs = {
        '全部数据': models_dir / 'training_data_all',
        '通用+LLM': models_dir / 'training_data_general_llm',
        '通用+规则': models_dir / 'training_data_general_rule'
    }
    
    # 评估结果
    results = {}
    total_evaluations = 0
    # LoRA模型：3个模型 × 3个测试集 = 9个评估任务
    # 基础模型：1个模型 × 3个测试集 = 3个评估任务
    # 总共：12个评估任务
    expected_evaluations = len(model_dirs) * len(test_sets) + len(all_test_sets)  # 9 + 3 = 12
    
    # 计算总体统计
    # 基础模型：1个模型在3个数据集上 = 3个结果
    # LoRA模型：3个模型在3个数据集上 = 9个结果
    # 总模型数：1（基础）+ 3（LoRA）= 4个模型
    # 总数据集数：3个（全部数据、通用+LLM、通用+规则）
    total_models = 1 + len(model_dirs)  # 基础模型 + LoRA模型
    total_datasets = len(all_test_sets)  # 3个数据集
    
    print(f"\n{'='*60}")
    print(f"评估任务统计：")
    print(f"  在 {total_datasets} 个数据集上评估 {total_models} 个模型，估计 {expected_evaluations} 个结果")
    print(f"{'='*60}")
    print(f"详细说明：")
    print(f"  - 基础模型(qwen25_05b)：1 个模型 × {len(all_test_sets)} 个数据集 = {len(all_test_sets)} 个结果")
    print(f"    （评估数据集：全部数据、通用+LLM、通用+规则）")
    print(f"  - LoRA微调模型：{len(model_dirs)} 个模型 × {len(test_sets)} 个数据集 = {len(model_dirs) * len(test_sets)} 个结果")
    print(f"    （评估数据集：全部数据、通用+LLM、通用+规则）")
    print(f"  - 总计：{expected_evaluations} 个评估任务")
    print(f"    计算：基础模型 {len(all_test_sets)} 个结果 + LoRA模型 {len(model_dirs) * len(test_sets)} 个结果 = {expected_evaluations} 个结果")
    print(f"{'='*60}")
    
    # 首先评估基础模型（在所有三个测试集上）
    print(f"\n{'='*60}")
    print(f"评估基础模型: qwen25_05b")
    print(f"{'='*60}")
    
    results['基础模型(qwen25_05b)'] = {}
    
    for test_set_name, test_file in all_test_sets.items():
        if not test_file.exists():
            print(f"警告: 测试集不存在: {test_file}")
            continue
        
        print(f"\n在测试集 {test_set_name} 上评估基础模型...")
        test_data = load_jsonl_dataset(test_file)
        
        # 基础模型不需要LoRA adapter，直接使用基础模型路径
        eval_result = evaluate_model(
            Path(base_model_path),  # 使用基础模型路径作为"模型路径"
            base_model_path,
            test_data,
            device,
            max_samples=args.max_samples,
            use_sampling=args.use_sampling,
            temperature=args.temperature
        )
        
        results['基础模型(qwen25_05b)'][test_set_name] = eval_result
        total_evaluations += 1
        
        print(f"  ✓ BLEU Score: {eval_result['bleu']:.4f}")
        if eval_result['ppl'] != float('inf'):
            print(f"  ✓ Perplexity: {eval_result['ppl']:.2f}")
        else:
            print(f"  ✓ Perplexity: N/A")
        print(f"  ✓ 评估样本数: {eval_result['num_samples']}")
    
    # 对每个LoRA模型进行评估
    for model_name, model_dir in model_dirs.items():
        if not model_dir.exists():
            print(f"警告: 模型目录不存在: {model_dir}")
            continue
        
        print(f"\n{'='*60}")
        print(f"评估模型: {model_name}")
        print(f"{'='*60}")
        
        results[model_name] = {}
        
        # 在每个测试集上评估
        for test_set_name, test_file in test_sets.items():
            if not test_file.exists():
                print(f"警告: 测试集不存在: {test_file}")
                continue
            
            print(f"\n在测试集 {test_set_name} 上评估...")
            test_data = load_jsonl_dataset(test_file)
            
            eval_result = evaluate_model(
                model_dir,
                base_model_path,
                test_data,
                device,
                max_samples=args.max_samples,
                use_sampling=args.use_sampling,
                temperature=args.temperature
            )
            
            results[model_name][test_set_name] = eval_result
            total_evaluations += 1
            
            print(f"  ✓ BLEU Score: {eval_result['bleu']:.4f}")
            if eval_result['ppl'] != float('inf'):
                print(f"  ✓ Perplexity: {eval_result['ppl']:.2f}")
            else:
                print(f"  ✓ Perplexity: N/A")
            print(f"  ✓ 评估样本数: {eval_result['num_samples']}")
    
    # 验证评估结果数量
    print(f"\n{'='*60}")
    print(f"评估完成统计:")
    print(f"  预期评估数: {expected_evaluations}")
    print(f"  实际完成数: {total_evaluations}")
    
    if total_evaluations < expected_evaluations:
        print(f"  警告: 只完成了 {total_evaluations}/{expected_evaluations} 个评估任务")
    else:
        print(f"  ✓ 所有评估任务已完成")
    
    # 先保存结果，避免打印错误导致结果丢失
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建带时间戳的子文件夹用于本次评估结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    evaluation_dir = output_dir / f"evaluation_{timestamp}"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n评估结果将保存在: {evaluation_dir}")
    
    # 先保存评估结果JSON（避免后续出错导致结果丢失）
    json_file = evaluation_dir / "evaluation_results.json"
    import json
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"✓ 评估结果JSON已保存到: {json_file}")
    
    # 打印结果摘要
    print(f"\n结果摘要:")
    for model_name in results.keys():
        print(f"  {model_name}:")
        # 所有模型都在相同的测试集上评估
        for test_set_name in test_sets.keys():
            if test_set_name in results[model_name]:
                result = results[model_name][test_set_name]
                # 处理PPL格式化：如果是inf则显示N/A，否则格式化
                ppl_str = f"{result['ppl']:.2f}" if result['ppl'] != float('inf') else 'N/A'
                print(f"    - {test_set_name}: BLEU={result['bleu']:.4f}, PPL={ppl_str}")
    print(f"{'='*60}")
    
    # 生成报告和图表
    # 保存Markdown报告
    md_file = evaluation_dir / "evaluation_report.md"
    generate_markdown_report(results, md_file)
    
    # 保存对比图表
    chart_file = evaluation_dir / "evaluation_comparison.png"
    plot_comparison(results, chart_file)
    
    print(f"\n{'='*60}")
    print("评估完成！")
    print(f"  完成评估数: {total_evaluations}/{expected_evaluations}")
    print(f"  评估结果文件夹: {evaluation_dir}")
    print(f"  报告保存位置: {md_file}")
    print(f"  图表保存位置: {chart_file}")
    print(f"  结果JSON保存位置: {json_file}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
