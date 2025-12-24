"""
数据集切分脚本
将training_datasets目录下的每个数据集切分为训练集和验证集

使用方法:
    python split_datasets.py
    python split_datasets.py --val-ratio 0.15
    python split_datasets.py --dataset training_data_general_rule.jsonl
"""
import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Any

def load_jsonl_dataset(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL格式的数据集"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if item:  # 确保不是空字典
                        data.append(item)
                except json.JSONDecodeError as e:
                    print(f"警告: 第 {line_num} 行JSON解析失败: {str(e)}")
                    continue
    except FileNotFoundError:
        print(f"错误: 文件不存在: {file_path}")
        return []
    except Exception as e:
        print(f"错误: 读取文件失败 {file_path}: {str(e)}")
        return []
    
    return data

def save_jsonl_dataset(data: List[Dict[str, Any]], file_path: str):
    """保存数据为JSONL格式"""
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def split_dataset(
    input_file: str,
    train_output_file: str,
    val_output_file: str,
    val_ratio: float = 0.1,
    random_seed: int = 42
) -> tuple:
    """
    切分数据集为训练集和验证集
    
    Args:
        input_file: 输入数据集文件路径（JSONL格式）
        train_output_file: 训练集输出文件路径
        val_output_file: 验证集输出文件路径
        val_ratio: 验证集比例（默认0.1，即10%）
        random_seed: 随机种子（默认42，确保可复现）
    
    Returns:
        (训练集数量, 验证集数量)
    """
    print(f"\n处理数据集: {Path(input_file).name}")
    print(f"验证集比例: {val_ratio*100:.1f}%")
    
    # 加载数据
    data = load_jsonl_dataset(input_file)
    
    if len(data) == 0:
        print(f"⚠ 警告: 数据集为空，跳过")
        return 0, 0
    
    print(f"总数据量: {len(data)} 条")
    
    # 随机打乱数据
    random.seed(random_seed)
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # 计算切分点
    total_count = len(shuffled_data)
    val_count = max(1, int(total_count * val_ratio))  # 至少1条
    train_count = total_count - val_count
    
    # 切分数据
    train_data = shuffled_data[:train_count]
    val_data = shuffled_data[train_count:]
    
    # 保存训练集
    save_jsonl_dataset(train_data, train_output_file)
    
    # 保存验证集
    save_jsonl_dataset(val_data, val_output_file)
    
    print(f"✓ 切分完成:")
    print(f"  - 训练集: {train_count} 条 ({train_count/total_count*100:.1f}%) -> {train_output_file}")
    print(f"  - 验证集: {val_count} 条 ({val_count/total_count*100:.1f}%) -> {val_output_file}")
    
    return train_count, val_count

def main():
    parser = argparse.ArgumentParser(
        description='数据集切分脚本 - 将训练数据集切分为训练集和验证集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 切分所有数据集（默认10%验证集）
  python split_datasets.py
  
  # 自定义验证集比例
  python split_datasets.py --val-ratio 0.15
  
  # 只切分指定数据集
  python split_datasets.py --dataset training_data_general_rule.jsonl
  
  # 指定输入和输出目录
  python split_datasets.py \\
    --input-dir ./output/training_datasets \\
    --output-dir ./output/validation_datasets
        """
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        default='./output/training_datasets',
        help='输入数据集目录（默认: ./output/training_datasets）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output/validation_datasets',
        help='输出目录（默认: ./output/validation_datasets，切分后的训练集和验证集都会保存到这里，之后需要运行organize_output.py整理）'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='指定要切分的数据集文件名（如training_data_general_rule.jsonl），如果不指定则处理所有数据集'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='验证集比例（默认: 0.1，即10%）'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='随机种子（默认: 42，确保可复现）'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='覆盖已存在的切分文件'
    )
    
    args = parser.parse_args()
    
    # 检查输入目录
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找数据集文件
    if args.dataset:
        dataset_files = [input_dir / args.dataset]
        if not dataset_files[0].exists():
            print(f"错误: 数据集文件不存在: {dataset_files[0]}")
            sys.exit(1)
    else:
        dataset_files = list(input_dir.glob("*.jsonl"))
        if len(dataset_files) == 0:
            print(f"错误: 在 {input_dir} 中未找到JSONL数据集文件")
            sys.exit(1)
    
    print("="*60)
    print("数据集切分工具")
    print("="*60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"验证集比例: {args.val_ratio*100:.1f}%")
    print(f"随机种子: {args.random_seed}")
    print(f"找到 {len(dataset_files)} 个数据集文件")
    print("="*60)
    
    # 统计信息
    total_train = 0
    total_val = 0
    success_count = 0
    skip_count = 0
    
    # 处理每个数据集
    for dataset_file in dataset_files:
        dataset_name = dataset_file.stem
        
        train_output_file = output_dir / f"{dataset_name}_train.jsonl"
        val_output_file = output_dir / f"{dataset_name}_val.jsonl"
        
        # 检查是否已存在
        if not args.overwrite and train_output_file.exists() and val_output_file.exists():
            print(f"\n跳过 {dataset_name}（已存在切分文件，使用 --overwrite 强制覆盖）")
            skip_count += 1
            continue
        
        try:
            train_count, val_count = split_dataset(
                str(dataset_file),
                str(train_output_file),
                str(val_output_file),
                val_ratio=args.val_ratio,
                random_seed=args.random_seed
            )
            
            if train_count > 0 or val_count > 0:
                total_train += train_count
                total_val += val_count
                success_count += 1
        except Exception as e:
            print(f"✗ 处理 {dataset_name} 失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 打印总结
    print("\n" + "="*60)
    print("切分完成！")
    print("="*60)
    print(f"成功处理: {success_count} 个数据集")
    if skip_count > 0:
        print(f"跳过: {skip_count} 个数据集（已存在）")
    print(f"总训练集数据: {total_train} 条")
    print(f"总验证集数据: {total_val} 条")
    print(f"总计: {total_train + total_val} 条")
    if total_train + total_val > 0:
        print(f"验证集比例: {total_val/(total_train+total_val)*100:.2f}%")
    print(f"\n输出目录: {output_dir}")
    print("="*60)
    
    # 列出生成的文件
    print("\n生成的文件:")
    train_files = sorted(output_dir.glob("*_train.jsonl"))
    val_files = sorted(output_dir.glob("*_val.jsonl"))
    
    for train_file, val_file in zip(train_files, val_files):
        train_size = train_file.stat().st_size / 1024  # KB
        val_size = val_file.stat().st_size / 1024  # KB
        print(f"  - {train_file.name} ({train_size:.1f} KB)")
        print(f"  - {val_file.name} ({val_size:.1f} KB)")

if __name__ == '__main__':
    main()
