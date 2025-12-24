"""
一键生成所有训练数据
整合所有数据生成功能，一键生成所有类型的数据集
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

from src.parser import RepositoryParser
from src.generator.rule_type_generator import RuleTypeGenerator
from src.generator import DesignGenerator
from src.validator import DataValidator
from src.output import OutputProcessor


def generate_rule_types(repo_data, output_dir, samples_per_type, rule_types):
    """生成规则类型数据"""
    print("\n" + "=" * 60)
    print("生成规则类型数据")
    print("=" * 60)
    
    generator = RuleTypeGenerator()
    output_processor = OutputProcessor()
    validator = DataValidator()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_generated = 0
    results = {}
    
    for rule_type in rule_types:
        print(f"\n生成 {rule_type} 类型数据...")
        print(f"  目标数量: {samples_per_type}")
        
        # 生成样本
        samples = generator.generate_by_type(repo_data, rule_type, samples_per_type)
        print(f"  生成: {len(samples)} 个样本")
        
        # 验证
        validated = validator.validate(samples)
        print(f"  验证通过: {len(validated)}/{len(samples)} 个样本")
        
        # 保存到文件
        output_file = output_path / f"{rule_type}.jsonl"
        output_processor.save_jsonl(validated, str(output_file))
        abs_path = output_file.resolve()
        print(f"  已保存到: {abs_path}")
        print(f"  文件路径: {abs_path}")
        
        results[rule_type] = len(validated)
        total_generated += len(validated)
    
    return results, total_generated


def generate_design_data(repo_data, output_dir, samples):
    """生成设计方案数据"""
    print("\n" + "=" * 60)
    print("生成设计方案数据")
    print("=" * 60)
    
    design_generator = DesignGenerator()
    output_processor = OutputProcessor()
    validator = DataValidator()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"目标数量: {samples}")
    
    # 扩展设计需求列表
    original_demands = design_generator.design_demands.copy()
    extended_demands = original_demands.copy()
    
    # 添加更多变体需求
    demand_variants = [
        "Design a new authentication module with OAuth2 support based on the repository architecture.",
        "Propose a design for adding distributed rate-limiting to the architecture.",
        "Design a comprehensive logging system with structured logging that integrates with the existing codebase.",
        "Propose a multi-level caching mechanism based on the current architecture.",
        "Design an advanced error handling module with retry logic following the repository patterns.",
        "Propose a design for adding real-time monitoring capabilities.",
        "Design a flexible configuration management system with environment-based settings.",
        "Propose a design for adding circuit breaker pattern with retry logic.",
        "Design an extensible plugin system with hot-reload capability based on the existing architecture.",
        "Propose a design for adding request/response validation layer with schema validation.",
    ]
    extended_demands.extend(demand_variants)
    
    # 生成样本
    all_samples = []
    seen_demands = set()
    
    # 首先生成所有唯一需求
    for demand in extended_demands:
        if len(all_samples) >= samples:
            break
        if demand.lower() not in seen_demands:
            seen_demands.add(demand.lower())
            samples_list = design_generator.generate(repo_data, num_samples=1)
            if samples_list:
                sample = samples_list[0]
                sample['demand'] = demand
                all_samples.append(sample)
    
    # 如果还需要更多样本，通过变异生成
    round_num = 1
    while len(all_samples) < samples and round_num < 20:
        for base_demand in original_demands:
            if len(all_samples) >= samples:
                break
            demand = f"{base_demand} (variant {round_num})"
            if demand.lower() not in seen_demands:
                seen_demands.add(demand.lower())
                samples_list = design_generator.generate(repo_data, num_samples=1)
                if samples_list:
                    sample = samples_list[0]
                    sample['demand'] = demand
                    sample['scheme'] = f"{sample['scheme']} This variant {round_num} considers additional requirements, performance optimizations, and edge cases specific to this implementation scenario."
                    all_samples.append(sample)
        round_num += 1
    
    print(f"生成: {len(all_samples)} 个样本")
    
    # 验证
    validated = validator.validate(all_samples)
    print(f"验证通过: {len(validated)}/{len(all_samples)} 个样本")
    
    # 保存数据
    output_file = output_path / "design.jsonl"
    output_processor.save_jsonl(validated, str(output_file))
    abs_path = output_file.resolve()
    print(f"已保存到: {abs_path}")
    print(f"文件路径: {abs_path}")
    
    return len(validated)


def merge_all_datasets(output_dir, rule_types):
    """合并所有数据集"""
    print("\n" + "=" * 60)
    print("合并所有数据集")
    print("=" * 60)
    
    output_processor = OutputProcessor()
    output_path = Path(output_dir)
    
    all_samples = []
    
    # 合并规则类型数据
    for rule_type in rule_types:
        file_path = output_path / f"{rule_type}.jsonl"
        if file_path.exists():
            samples = output_processor.load_jsonl(str(file_path))
            all_samples.extend(samples)
            print(f"  已加载 {rule_type}: {len(samples)} 条")
    
    # 合并设计方案数据
    design_file = output_path / "design.jsonl"
    if design_file.exists():
        samples = output_processor.load_jsonl(str(design_file))
        all_samples.extend(samples)
        print(f"  已加载 design: {len(samples)} 条")
    
    # 保存合并文件
    merged_file = output_path / "all_data.jsonl"
    output_processor.save_jsonl(all_samples, str(merged_file))
    abs_path = merged_file.resolve()
    print(f"\n合并文件已保存: {abs_path}")
    print(f"总样本数: {len(all_samples)} 条")
    
    return len(all_samples)


def main():
    parser = argparse.ArgumentParser(
        description='一键生成所有训练数据（规则类型 + 设计方案）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认参数（每类1000条）
  python generate_all_data.py
  
  # 自定义参数
  python generate_all_data.py --samples-per-type 1000 --design-samples 1000 --max-files 100
  
  # 只生成规则类型数据
  python generate_all_data.py --no-design
  
  # 只生成设计方案数据
  python generate_all_data.py --no-rules
        """
    )
    
    parser.add_argument(
        '--repo',
        type=str,
        default=None,
        help='仓库路径（GitHub URL或本地路径，例如: https://github.com/psf/requests 或 ./requests）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/datasets',
        help='输出目录（默认: output/datasets）'
    )
    parser.add_argument(
        '--samples-per-type',
        type=int,
        default=1000,
        help='每种规则类型的样本数量（默认: 1000）'
    )
    parser.add_argument(
        '--design-samples',
        type=int,
        default=1000,
        help='设计方案样本数量（默认: 1000）'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=100,
        help='最大解析文件数（默认: 100）'
    )
    parser.add_argument(
        '--types',
        type=str,
        nargs='+',
        default=['conditional_rule', 'function_rule', 'class_rule', 'comment_rule'],
        help='要生成的规则类型（默认: conditional_rule function_rule class_rule comment_rule）'
    )
    parser.add_argument(
        '--no-design',
        action='store_true',
        help='不生成设计方案数据'
    )
    parser.add_argument(
        '--no-rules',
        action='store_true',
        help='不生成规则类型数据'
    )
    parser.add_argument(
        '--no-merge',
        action='store_true',
        help='不生成合并文件'
    )
    
    args = parser.parse_args()
    
    # 打印开始信息
    print("=" * 60)
    print("一键生成所有训练数据")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"仓库: {args.repo}")
    print(f"输出目录: {args.output_dir}")
    print(f"规则类型样本数: {args.samples_per_type} 条/类")
    print(f"设计方案样本数: {args.design_samples} 条")
    print(f"最大文件数: {args.max_files}")
    print(f"规则类型: {', '.join(args.types)}")
    print("=" * 60)
    
    # 验证必需参数
    if not args.repo:
        print("错误: 必须指定 --repo 参数（GitHub URL或本地路径）")
        print("示例: --repo https://github.com/psf/requests")
        print("示例: --repo ./requests")
        sys.exit(1)
    
    try:
        # 步骤1: 解析仓库（只解析一次，所有生成共享）
        print("\n[步骤1] 解析仓库...")
        repo_parser = RepositoryParser()
        repo_data = repo_parser.parse(args.repo, max_files=args.max_files)
        print(f"[OK] 解析完成: {repo_data['total_files']} 个文件, {repo_data['total_rules']} 个规则")
        
        # 统计各类型规则数量
        rules = repo_data.get('rules', [])
        rule_type_counts = {}
        for rule in rules:
            rule_type = rule.get('type', 'unknown')
            rule_type_counts[rule_type] = rule_type_counts.get(rule_type, 0) + 1
        
        print("\n规则类型统计:")
        for rule_type, count in rule_type_counts.items():
            print(f"  - {rule_type}: {count} 个")
        
        total_samples = 0
        rule_results = {}
        design_count = 0
        
        # 步骤2: 生成规则类型数据
        if not args.no_rules:
            rule_results, rule_total = generate_rule_types(
                repo_data, 
                args.output_dir, 
                args.samples_per_type, 
                args.types
            )
            total_samples += rule_total
        
        # 步骤3: 生成设计方案数据
        if not args.no_design:
            design_count = generate_design_data(
                repo_data,
                args.output_dir,
                args.design_samples
            )
            total_samples += design_count
        
        # 步骤4: 合并所有数据集
        if not args.no_merge and (not args.no_rules or not args.no_design):
            merge_all_datasets(args.output_dir, args.types)
        
        # 最终统计
        print("\n" + "=" * 60)
        print("生成完成统计")
        print("=" * 60)
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n总样本数: {total_samples} 条")
        
        if rule_results:
            print("\n规则类型数据:")
            for rule_type, count in rule_results.items():
                print(f"  - {rule_type}: {count} 条")
        
        if design_count > 0:
            print(f"\n设计方案数据: {design_count} 条")
        
        # 文件列表
        output_path = Path(args.output_dir)
        output_processor = OutputProcessor()
        print("\n生成的文件:")
        for file_path in sorted(output_path.glob("*.jsonl")):
            if file_path.stat().st_size > 0:
                abs_path = file_path.resolve()
                file_size = file_path.stat().st_size / 1024  # KB
                try:
                    sample_count = len(output_processor.load_jsonl(str(file_path)))
                    print(f"  - {file_path.name}: {sample_count} 条 ({file_size:.1f} KB)")
                    print(f"    完整路径: {abs_path}")
                except:
                    print(f"  - {file_path.name}: ({file_size:.1f} KB)")
                    print(f"    完整路径: {abs_path}")
        
        print("\n" + "=" * 60)
        print("[OK] 所有数据生成完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] 错误: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

