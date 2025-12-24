"""
基于LLM的训练数据生成脚本
使用Qwen模型生成训练数据
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

from src.parser import RepositoryParser
from src.generator import LLMGenerator
from src.validator import DataValidator
from src.output import OutputProcessor


def generate_llm_qa_data(repo_data, output_dir, num_samples, llm_generator):
    """生成LLM问答对数据"""
    print("\n" + "=" * 60)
    print("生成LLM问答对数据")
    print("=" * 60)
    
    output_processor = OutputProcessor()
    validator = DataValidator()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"目标数量: {num_samples}")
    
    # 使用LLM生成问答对（带验证反馈循环生成）
    validated = llm_generator.generate_batch_qa_with_retry(
        repo_data, num_samples, validator, max_retries=3
    )
    print(f"生成并验证通过: {len(validated)}/{num_samples} 个样本")
    
    # 保存到文件
    output_file = output_path / "llm_qa.jsonl"
    output_processor.save_jsonl(validated, str(output_file))
    print(f"已保存到: {output_file}")
    
    return len(validated)


def generate_llm_design_data(repo_data, output_dir, num_samples, llm_generator):
    """生成LLM设计方案数据"""
    print("\n" + "=" * 60)
    print("生成LLM设计方案数据")
    print("=" * 60)
    
    output_processor = OutputProcessor()
    validator = DataValidator()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"目标数量: {num_samples}")
    
    # 使用LLM生成设计方案（带验证反馈循环生成）
    validated = llm_generator.generate_batch_design_with_retry(
        repo_data, num_samples, validator, max_retries=3
    )
    print(f"生成并验证通过: {len(validated)}/{num_samples} 个样本")
    
    # 保存数据
    output_file = output_path / "llm_design.jsonl"
    output_processor.save_jsonl(validated, str(output_file))
    print(f"已保存到: {output_file}")
    
    return len(validated)


def merge_llm_datasets(output_dir):
    """合并LLM数据集"""
    print("\n" + "=" * 60)
    print("合并LLM数据集")
    print("=" * 60)
    
    output_processor = OutputProcessor()
    output_path = Path(output_dir)
    
    all_samples = []
    
    # 合并问答对数据
    qa_file = output_path / "llm_qa.jsonl"
    if qa_file.exists():
        samples = output_processor.load_jsonl(str(qa_file))
        all_samples.extend(samples)
        print(f"  已加载 llm_qa: {len(samples)} 条")
    
    # 合并设计方案数据
    design_file = output_path / "llm_design.jsonl"
    if design_file.exists():
        samples = output_processor.load_jsonl(str(design_file))
        all_samples.extend(samples)
        print(f"  已加载 llm_design: {len(samples)} 条")
    
    # 保存合并文件
    merged_file = output_path / "llm_all_data.jsonl"
    output_processor.save_jsonl(all_samples, str(merged_file))
    print(f"\n合并文件已保存: {merged_file}")
    print(f"总样本数: {len(all_samples)} 条")
    
    return len(all_samples)


def main():
    parser = argparse.ArgumentParser(
        description='使用LLM生成训练数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认参数
  python generate_llm_data.py
  
  # 自定义参数
  python generate_llm_data.py --qa-samples 100 --design-samples 50 --max-files 50
  
  # 只生成问答对
  python generate_llm_data.py --no-design
  
  # 只生成设计方案
  python generate_llm_data.py --no-qa
        """
    )
    
    parser.add_argument(
        '--repo',
        type=str,
        default='https://github.com/psf/requests',
        help='仓库路径（默认: https://github.com/psf/requests，可以是本地路径或GitHub URL）'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='LLM模型路径（默认: 从环境变量QWEN_MODEL_PATH获取，或使用models/qwen3-4b）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/llm_datasets',
        help='输出目录（默认: output/llm_datasets）'
    )
    parser.add_argument(
        '--qa-samples',
        type=int,
        default=400,
        help='问答对样本数量（默认: 400）'
    )
    parser.add_argument(
        '--design-samples',
        type=int,
        default=100,
        help='设计方案样本数量（默认: 100）'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=100,
        help='最大解析文件数（默认: 100）'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='设备（cuda/cpu），默认自动选择'
    )
    parser.add_argument(
        '--no-qa',
        action='store_true',
        help='不生成问答对数据'
    )
    parser.add_argument(
        '--no-design',
        action='store_true',
        help='不生成设计方案数据'
    )
    parser.add_argument(
        '--no-merge',
        action='store_true',
        help='不生成合并文件'
    )
    
    args = parser.parse_args()
    
    # 打印开始信息
    print("=" * 60)
    print("基于LLM的训练数据生成")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"仓库: {args.repo}")
    print(f"模型路径: {args.model_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"问答对样本数: {args.qa_samples} 条")
    print(f"设计方案样本数: {args.design_samples} 条")
    print(f"最大文件数: {args.max_files}")
    print(f"设备: {args.device or '自动选择'}")
    print("=" * 60)
    
    try:
        # 步骤1: 解析仓库
        print("\n[步骤1] 解析仓库...")
        print(f"  仓库路径: {args.repo}")
        
        # 如果是本地路径且不存在，提前检查并给出提示
        if not (args.repo.startswith('http://') or args.repo.startswith('https://')):
            repo_path = Path(args.repo)
            if not repo_path.is_absolute():
                repo_path = Path.cwd() / repo_path
            if not repo_path.exists():
                print(f"\n  ⚠ 警告: 本地路径不存在: {args.repo}")
                print(f"  提示: 如果这是GitHub仓库，请使用完整URL，例如:")
                print(f"    --repo https://github.com/owner/repo")
                print(f"  或者确保本地路径正确")
        
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
        
        # 步骤2: 加载LLM模型
        print("\n[步骤2] 加载LLM模型...")
        if args.model_path:
            print(f"  使用指定的模型路径: {args.model_path}")
        else:
            import os
            env_path = os.getenv('QWEN_MODEL_PATH')
            if env_path:
                print(f"  从环境变量获取模型路径: {env_path}")
            else:
                print(f"  使用默认模型路径: models/qwen3-4b")
                print(f"  提示: 如果模型不在默认位置，请使用 --model-path 指定路径")
        
        try:
            llm_generator = LLMGenerator(model_path=args.model_path, device=args.device)
            print("[OK] 模型加载完成")
        except FileNotFoundError as e:
            print(f"\n[ERROR] 模型路径错误: {str(e)}")
            print(f"\n解决方案:")
            print(f"  1. 使用 --model-path 指定模型路径:")
            print(f"     python generate_llm_data.py --model-path /path/to/your/model")
            print(f"  2. 或设置环境变量:")
            print(f"     export QWEN_MODEL_PATH=/path/to/your/model")
            sys.exit(1)
        
        total_samples = 0
        qa_count = 0
        design_count = 0
        
        # 步骤3: 生成问答对数据
        if not args.no_qa:
            qa_count = generate_llm_qa_data(
                repo_data,
                args.output_dir,
                args.qa_samples,
                llm_generator
            )
            total_samples += qa_count
        
        # 步骤4: 生成设计方案数据
        if not args.no_design:
            design_count = generate_llm_design_data(
                repo_data,
                args.output_dir,
                args.design_samples,
                llm_generator
            )
            total_samples += design_count
        
        # 步骤5: 合并所有数据集
        if not args.no_merge and (not args.no_qa or not args.no_design):
            merge_llm_datasets(args.output_dir)
        
        # 最终统计
        print("\n" + "=" * 60)
        print("生成完成统计")
        print("=" * 60)
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n总样本数: {total_samples} 条")
        
        if qa_count > 0:
            print(f"问答对数据: {qa_count} 条")
        
        if design_count > 0:
            print(f"设计方案数据: {design_count} 条")
        
        # 文件列表
        output_path = Path(args.output_dir)
        output_processor = OutputProcessor()
        print("\n生成的文件:")
        for file_path in sorted(output_path.glob("*.jsonl")):
            if file_path.stat().st_size > 0:
                file_size = file_path.stat().st_size / 1024  # KB
                try:
                    sample_count = len(output_processor.load_jsonl(str(file_path)))
                    print(f"  - {file_path.name}: {sample_count} 条 ({file_size:.1f} KB)")
                except:
                    print(f"  - {file_path.name}: ({file_size:.1f} KB)")
        
        print("\n" + "=" * 60)
        print("[OK] LLM数据生成完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] 错误: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
