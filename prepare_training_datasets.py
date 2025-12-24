"""
训练数据集准备脚本
支持合并多个数据集，生成不同组合的训练数据

数据集：
1. 通用数据集（parquet格式）- 防止过拟合
2. 基于规则的数据集（jsonl格式）
3. LLM生成的数据集（jsonl格式）

支持的组合：
- 1+2+3：全部数据集
- 1+2：通用+规则
- 1+3：通用+LLM
"""
import argparse
import json
import sys
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("警告: 未安装pandas，无法读取parquet文件。请运行: pip install pandas pyarrow")


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL文件"""
    data = []
    if not Path(file_path).exists():
        print(f"警告: 文件不存在: {file_path}")
        return data
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        print(f"  ✓ 已加载 {file_path}: {len(data)} 条")
    except Exception as e:
        print(f"  ✗ 加载失败 {file_path}: {str(e)}")
    
    return data


def load_parquet(file_path: str) -> List[Dict[str, Any]]:
    """加载Parquet文件并转换为JSONL格式"""
    if not HAS_PANDAS:
        raise ImportError("需要安装pandas和pyarrow来读取parquet文件")
    
    if not Path(file_path).exists():
        print(f"警告: 文件不存在: {file_path}")
        return []
    
    try:
        df = pd.read_parquet(file_path)
        # 转换为字典列表
        data = df.to_dict('records')
        print(f"  ✓ 已加载 {file_path}: {len(data)} 条")
        return data
    except Exception as e:
        print(f"  ✗ 加载失败 {file_path}: {str(e)}")
        return []


def normalize_data_format(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    标准化数据格式，确保所有数据集格式一致
    处理parquet和jsonl格式的差异
    """
    # 如果已经是标准格式，直接返回
    if 'type' in sample:
        return sample
    
    # 尝试从parquet格式转换
    # 常见的parquet格式字段：instruction, input, output, messages等
    normalized = {}
    
    # 检查是否是对话格式
    if 'messages' in sample:
        messages = sample['messages']
        if isinstance(messages, list) and len(messages) > 0:
            # 提取问题和答案
            question = ""
            answer = ""
            for msg in messages:
                role = msg.get('role', '')
                content = msg.get('content', '')
                if role == 'user':
                    question = content
                elif role == 'assistant':
                    answer = content
            
            normalized = {
                'type': 'qa',
                'question': question,
                'answer': answer,
                'code_snippets': sample.get('code_snippets', []),
                'reasoning_trace': sample.get('reasoning_trace', ''),
                'metadata': sample.get('metadata', {})
            }
    
    # 检查是否是instruction格式
    elif 'instruction' in sample or 'input' in sample:
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        output = sample.get('output', '')
        
        # 组合问题和答案
        question = instruction
        if input_text:
            question = f"{instruction}\n{input_text}" if instruction else input_text
        
        normalized = {
            'type': 'qa',
            'question': question,
            'answer': output,
            'code_snippets': sample.get('code_snippets', []),
            'reasoning_trace': sample.get('reasoning_trace', ''),
            'metadata': sample.get('metadata', {})
        }
    
    # 如果无法识别，尝试直接使用
    else:
        normalized = sample.copy()
        # 确保有type字段
        if 'type' not in normalized:
            normalized['type'] = 'qa'
    
    return normalized


def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """保存为JSONL格式"""
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"  ✓ 已保存 {file_path}: {len(data)} 条")


def sample_data(data: List[Dict[str, Any]], target_size: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    从数据集中采样指定数量的样本
    
    Args:
        data: 原始数据列表
        target_size: 目标采样数量
        seed: 随机种子
        
    Returns:
        采样后的数据列表
    """
    if len(data) <= target_size:
        return data
    
    random.seed(seed)
    return random.sample(data, target_size)


def merge_datasets(
    general_data: List[Dict[str, Any]],
    rule_data: List[Dict[str, Any]],
    llm_data: List[Dict[str, Any]],
    output_dir: str,
    combination: str
):
    """
    合并数据集，确保新数据集和通用数据集比例1:1
    
    Args:
        general_data: 通用数据集
        rule_data: 基于规则的数据集
        llm_data: LLM生成的数据集
        output_dir: 输出目录
        combination: 组合类型 ('all', 'general_rule', 'general_llm')
    """
    print(f"\n{'='*60}")
    print(f"合并数据集: {combination}")
    print(f"{'='*60}")
    
    # 标准化所有数据
    print("\n标准化数据格式...")
    general_normalized = [normalize_data_format(item) for item in general_data]
    rule_normalized = [normalize_data_format(item) for item in rule_data]
    llm_normalized = [normalize_data_format(item) for item in llm_data]
    
    print(f"\n原始数据统计:")
    print(f"  通用数据集: {len(general_normalized)} 条")
    print(f"  规则数据集: {len(rule_normalized)} 条")
    print(f"  LLM数据集: {len(llm_normalized)} 条")
    
    # 根据组合类型合并，确保比例1:1
    merged_data = []
    
    if combination == 'all':
        # 1+2+3：全部数据集，确保 通用:(规则+LLM) = 1:1
        if len(llm_normalized) == 0:
            print("  ⚠ 警告: LLM数据集为空，将只合并通用和规则数据集")
            # 规则+LLM = 规则
            new_data = rule_normalized.copy()
            rule_indices = list(range(len(rule_normalized)))
            llm_indices = []
        else:
            # 规则+LLM合并为新数据集，记录索引
            new_data = rule_normalized + llm_normalized
            rule_indices = list(range(len(rule_normalized)))
            llm_indices = list(range(len(rule_normalized), len(new_data)))
        
        # 计算目标数量：取通用数据集和新数据集的最小值
        general_count = len(general_normalized)
        new_count = len(new_data)
        target_count = min(general_count, new_count)
        
        print(f"\n比例调整:")
        print(f"  通用数据集: {general_count} 条")
        print(f"  新数据集(规则+LLM): {new_count} 条")
        print(f"  目标数量(1:1比例): {target_count} 条")
        
        # 采样到目标数量
        general_sampled = sample_data(general_normalized, target_count, seed=42)
        
        # 使用索引采样，以便跟踪来源
        random.seed(43)
        all_indices = list(range(len(new_data)))
        sampled_indices = random.sample(all_indices, target_count) if len(all_indices) > target_count else all_indices
        new_sampled = [new_data[i] for i in sampled_indices]
        
        # 统计来源
        rule_count_in_new = sum(1 for i in sampled_indices if i in rule_indices)
        llm_count_in_new = sum(1 for i in sampled_indices if i in llm_indices)
        
        merged_data.extend(general_sampled)
        merged_data.extend(new_sampled)
        output_file = Path(output_dir) / "training_data_all.jsonl"
        
        print(f"\n合并结果: 通用({len(general_sampled)}) + 新数据({len(new_sampled)}) = {len(merged_data)} 条")
        print(f"  其中新数据包含: 规则({rule_count_in_new}) + LLM({llm_count_in_new})")
    
    elif combination == 'general_rule':
        # 1+2：通用+规则，确保比例1:1
        general_count = len(general_normalized)
        rule_count = len(rule_normalized)
        target_count = min(general_count, rule_count)
        
        print(f"\n比例调整:")
        print(f"  通用数据集: {general_count} 条")
        print(f"  规则数据集: {rule_count} 条")
        print(f"  目标数量(1:1比例): {target_count} 条")
        
        # 采样到目标数量
        general_sampled = sample_data(general_normalized, target_count, seed=42)
        rule_sampled = sample_data(rule_normalized, target_count, seed=44)
        
        merged_data.extend(general_sampled)
        merged_data.extend(rule_sampled)
        output_file = Path(output_dir) / "training_data_general_rule.jsonl"
        print(f"\n合并结果: 通用({len(general_sampled)}) + 规则({len(rule_sampled)}) = {len(merged_data)} 条")
    
    elif combination == 'general_llm':
        # 1+3：通用+LLM，确保比例1:1
        if len(llm_normalized) == 0:
            raise ValueError("LLM数据集为空，无法生成 general_llm 组合。请先生成LLM数据集。")
        
        general_count = len(general_normalized)
        llm_count = len(llm_normalized)
        target_count = min(general_count, llm_count)
        
        print(f"\n比例调整:")
        print(f"  通用数据集: {general_count} 条")
        print(f"  LLM数据集: {llm_count} 条")
        print(f"  目标数量(1:1比例): {target_count} 条")
        
        # 采样到目标数量
        general_sampled = sample_data(general_normalized, target_count, seed=42)
        llm_sampled = sample_data(llm_normalized, target_count, seed=45)
        
        merged_data.extend(general_sampled)
        merged_data.extend(llm_sampled)
        output_file = Path(output_dir) / "training_data_general_llm.jsonl"
        print(f"\n合并结果: 通用({len(general_sampled)}) + LLM({len(llm_sampled)}) = {len(merged_data)} 条")
    
    else:
        raise ValueError(f"未知的组合类型: {combination}")
    
    # 打乱数据顺序
    random.seed(42)
    random.shuffle(merged_data)
    
    # 保存合并后的数据
    print(f"\n最终总样本数: {len(merged_data)} 条")
    save_jsonl(merged_data, str(output_file))
    
    # 统计信息
    print(f"\n数据统计:")
    type_counts = {}
    for item in merged_data:
        data_type = item.get('type', 'unknown')
        type_counts[data_type] = type_counts.get(data_type, 0) + 1
    
    for data_type, count in type_counts.items():
        print(f"  - {data_type}: {count} 条")
    
    # 验证比例（通过统计数量）
    if combination == 'all':
        general_in_merged = len(general_sampled)
        new_in_merged = len(new_sampled)
    elif combination == 'general_rule':
        general_in_merged = len(general_sampled)
        new_in_merged = len(rule_sampled)
    elif combination == 'general_llm':
        general_in_merged = len(general_sampled)
        new_in_merged = len(llm_sampled)
    else:
        general_in_merged = 0
        new_in_merged = 0
    
    print(f"\n比例验证:")
    print(f"  通用数据集: {general_in_merged} 条")
    print(f"  新数据集: {new_in_merged} 条")
    if general_in_merged > 0 and new_in_merged > 0:
        ratio = general_in_merged / new_in_merged
        print(f"  比例: {general_in_merged}:{new_in_merged} ≈ {ratio:.2f}:1")
        if abs(ratio - 1.0) < 0.01:
            print(f"  ✓ 比例正确 (1:1)")
        else:
            print(f"  ⚠ 比例略有偏差")
    else:
        print(f"  ⚠ 无法验证比例（数据集为空）")
    
    return len(merged_data), str(output_file)


def main():
    parser = argparse.ArgumentParser(
        description='准备训练数据集，支持多种数据集组合',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成所有组合
  python prepare_training_datasets.py --all
  
  # 只生成特定组合
  python prepare_training_datasets.py --combination all
  python prepare_training_datasets.py --combination general_rule
  python prepare_training_datasets.py --combination general_llm
  
  # 自定义数据集路径
  python prepare_training_datasets.py --all \\
    --general ./output/rule_datasets/train-00000-of-00001.parquet \\
    --rule ./output/rule_datasets/all_data.jsonl \\
    --llm ./output/llm_datasets/llm_all_data.jsonl
        """
    )
    
    parser.add_argument(
        '--general',
        type=str,
        default='./output/rule_datasets/train-00000-of-00001.parquet',
        help='通用数据集路径（parquet格式，默认: ./output/rule_datasets/train-00000-of-00001.parquet）'
    )
    parser.add_argument(
        '--rule',
        type=str,
        default='./output/rule_datasets/all_data.jsonl',
        help='基于规则的数据集路径（jsonl格式，默认: ./output/rule_datasets/all_data.jsonl）'
    )
    parser.add_argument(
        '--llm',
        type=str,
        default='./output/llm_datasets/llm_all_data.jsonl',
        help='LLM生成的数据集路径（jsonl格式，默认: ./output/llm_datasets/llm_all_data.jsonl）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output/training_datasets',
        help='输出目录（默认: ./output/training_datasets）'
    )
    parser.add_argument(
        '--combination',
        type=str,
        choices=['all', 'general_rule', 'general_llm'],
        help='要生成的组合类型（all/general_rule/general_llm）'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='生成所有组合（1+2+3, 1+2, 1+3）'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("训练数据集准备脚本")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"通用数据集: {args.general}")
    print(f"规则数据集: {args.rule}")
    print(f"LLM数据集: {args.llm}")
    print(f"输出目录: {args.output_dir}")
    print("="*60)
    
    # 确定要生成的组合
    if args.all:
        combinations = ['all', 'general_rule', 'general_llm']
    elif args.combination:
        combinations = [args.combination]
    else:
        print("\n错误: 请指定 --all 或 --combination")
        parser.print_help()
        sys.exit(1)
    
    # 加载数据集
    print("\n加载数据集...")
    general_data = []
    rule_data = []
    llm_data = []
    
    # 加载通用数据集（parquet）
    if Path(args.general).exists():
        general_data = load_parquet(args.general)
    else:
        print(f"  ⚠ 通用数据集不存在: {args.general}")
    
    # 加载规则数据集（jsonl）
    if Path(args.rule).exists():
        rule_data = load_jsonl(args.rule)
    else:
        print(f"  ⚠ 规则数据集不存在: {args.rule}")
    
    # 加载LLM数据集（jsonl）
    if Path(args.llm).exists():
        llm_data = load_jsonl(args.llm)
    else:
        print(f"  ⚠ LLM数据集不存在: {args.llm}")
        print(f"    提示: LLM数据集可能还未生成，可以稍后运行生成脚本")
    
    # 检查数据
    print(f"\n数据集统计:")
    print(f"  通用数据集: {len(general_data)} 条")
    print(f"  规则数据集: {len(rule_data)} 条")
    print(f"  LLM数据集: {len(llm_data)} 条")
    
    # 检查至少有一个数据集可用
    available_datasets = []
    if len(general_data) > 0:
        available_datasets.append("通用")
    if len(rule_data) > 0:
        available_datasets.append("规则")
    if len(llm_data) > 0:
        available_datasets.append("LLM")
    
    if len(available_datasets) == 0:
        print("\n错误: 所有数据集都为空或不存在！")
        sys.exit(1)
    
    print(f"  可用数据集: {', '.join(available_datasets)}")
    
    # 检查组合的可行性
    if 'general_llm' in combinations and len(llm_data) == 0:
        print("\n  ⚠ 警告: LLM数据集不存在，将跳过 general_llm 组合")
        combinations = [c for c in combinations if c != 'general_llm']
    
    if 'all' in combinations and len(llm_data) == 0:
        print("\n  ⚠ 警告: LLM数据集不存在，all组合将只包含通用和规则数据")
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 生成所有指定的组合
    results = []
    for combination in combinations:
        try:
            count, output_file = merge_datasets(
                general_data,
                rule_data,
                llm_data,
                args.output_dir,
                combination
            )
            results.append({
                'combination': combination,
                'count': count,
                'file': output_file
            })
        except Exception as e:
            print(f"\n错误: 生成组合 {combination} 失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 最终统计
    print("\n" + "="*60)
    print("生成完成统计")
    print("="*60)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n成功生成 {len(results)} 个数据集:")
    for result in results:
        print(f"  - {result['combination']}: {result['count']} 条 -> {result['file']}")
    
    print("\n" + "="*60)
    print("[OK] 数据集准备完成！")
    print("="*60)


if __name__ == '__main__':
    main()
