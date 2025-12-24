"""
检查当前数据集包含的数据类型
"""
import json
from pathlib import Path
from collections import defaultdict

def check_datasets():
    """检查数据集类型"""
    datasets_dir = Path("output/datasets")
    
    if not datasets_dir.exists():
        print("数据集目录不存在")
        return
    
    print("=" * 60)
    print("数据集类型检查")
    print("=" * 60)
    
    # 检查已生成的文件
    dataset_files = {
        'conditional_rule': 'conditional_rule.jsonl',
        'function_rule': 'function_rule.jsonl',
        'class_rule': 'class_rule.jsonl',
        'comment_rule': 'comment_rule.jsonl',
        'all_rules': 'all_rules.jsonl'
    }
    
    print("\n已生成的数据集:")
    total_samples = 0
    for type_name, filename in dataset_files.items():
        file_path = datasets_dir / filename
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    samples = [json.loads(line) for line in f if line.strip()]
                count = len(samples)
                total_samples += count
                file_size = file_path.stat().st_size / 1024  # KB
                print(f"  [OK] {type_name}: {count} 条样本 ({file_size:.1f} KB)")
            except Exception as e:
                print(f"  [FAIL] {type_name}: 读取失败 - {e}")
        else:
            print(f"  - {type_name}: 未生成")
    
    print(f"\n总计: {total_samples} 条样本")
    
    # 检查all_rules.jsonl中的类型分布
    all_rules_file = datasets_dir / 'all_rules.jsonl'
    if all_rules_file.exists():
        print("\nall_rules.jsonl 中的类型分布:")
        type_counts = defaultdict(int)
        try:
            with open(all_rules_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        sample = json.loads(line)
                        rule_type = sample.get('metadata', {}).get('rule_type', 'unknown')
                        type_counts[rule_type] += 1
            
            for rule_type, count in sorted(type_counts.items()):
                print(f"  - {rule_type}: {count} 条")
        except Exception as e:
            print(f"  读取失败: {e}")
    
    # 检查可用的其他类型
    print("\n可用的其他数据类型:")
    print("  - comment_rule: 注释规则（已定义但未生成）")
    print("  - design: 设计方案（需要单独生成）")
    print("  - exception_rule: 异常处理规则（可扩展）")
    print("  - loop_rule: 循环规则（可扩展）")
    print("  - decorator_rule: 装饰器规则（可扩展）")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    check_datasets()

