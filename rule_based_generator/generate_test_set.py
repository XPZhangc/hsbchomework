"""
生成测试集
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

from src.parser import RepositoryParser
from src.generator.mcq_generator import MCQGenerator
from src.validator import DataValidator
from src.output import OutputProcessor


def main():
    parser = argparse.ArgumentParser(
        description='生成选择题测试集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认参数（默认仓库: ./requests-main/requests-main）
  python generate_test_set.py
  
  # 自定义题目数量
  python generate_test_set.py --num-questions 100
  
  # 指定仓库路径（GitHub URL）
  python generate_test_set.py --repo https://github.com/psf/requests
  
  # 指定本地仓库路径
  python generate_test_set.py --repo ./my_repo
        """
    )
    
    parser.add_argument(
        '--repo',
        type=str,
        default='./requests-main/requests-main',
        help='仓库路径（GitHub URL或本地路径，默认: ./requests-main/requests-main，例如: https://github.com/psf/requests 或 ./requests-main/requests-main）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/test_set.jsonl',
        help='输出文件路径（默认: output/test_set.jsonl）'
    )
    parser.add_argument(
        '--num-questions',
        type=int,
        default=100,
        help='题目数量（默认: 100）'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=100,
        help='最大解析文件数（默认: 100）'
    )
    
    args = parser.parse_args()
    
    # 处理相对路径：如果是相对路径，尝试相对于项目根目录
    repo_path = args.repo
    if not repo_path.startswith('http://') and not repo_path.startswith('https://'):
        repo_path_obj = Path(repo_path)
        # 如果是相对路径且不存在，尝试多个可能的位置
        if not repo_path_obj.is_absolute() and not repo_path_obj.exists():
            script_dir = Path(__file__).parent
            project_root = script_dir.parent
            
            # 清理路径（移除 ./ 前缀）
            clean_path = repo_path.lstrip('./')
            
            # 尝试1: 相对于项目根目录
            alt_path = project_root / clean_path
            if alt_path.exists():
                repo_path = str(alt_path.resolve())
            else:
                # 尝试2: 相对于脚本所在目录
                alt_path = script_dir / clean_path
                if alt_path.exists():
                    repo_path = str(alt_path.resolve())
                else:
                    # 尝试3: 相对于当前工作目录（原始路径）
                    # 如果都不存在，保持原路径，让repository_parser给出错误
                    pass
        elif repo_path_obj.exists():
            # 如果路径存在，转换为绝对路径
            repo_path = str(repo_path_obj.resolve())
    
    print("=" * 60)
    print("生成选择题测试集")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"仓库: {repo_path}")
    print(f"输出: {args.output}")
    print(f"题目数量: {args.num_questions} 道")
    print(f"最大文件数: {args.max_files}")
    print("=" * 60)
    
    try:
        # 步骤1: 解析仓库
        print("\n[1/3] 解析仓库...")
        
        # 检查路径是否存在（对于本地路径）
        if not repo_path.startswith('http://') and not repo_path.startswith('https://'):
            repo_path_check = Path(repo_path)
            if not repo_path_check.exists():
                print(f"\n[错误] 仓库路径不存在: {repo_path}")
                print("\n提示:")
                print("  1. 使用GitHub URL（推荐）:")
                print("     python generate_test_set.py --repo https://github.com/psf/requests")
                print("  2. 指定本地仓库路径:")
                print("     python generate_test_set.py --repo ./your_repo")
                print("  3. 如果使用默认路径，请确保项目根目录下有 'requests-main/requests-main' 目录")
                sys.exit(1)
        
        repo_parser = RepositoryParser()
        repo_data = repo_parser.parse(repo_path, max_files=args.max_files)
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
        
        # 步骤2: 生成选择题
        print(f"\n[2/3] 生成 {args.num_questions} 道选择题...")
        mcq_generator = MCQGenerator()
        questions = mcq_generator.generate(repo_data, num_questions=args.num_questions)
        print(f"[OK] 生成了 {len(questions)} 道选择题")
        
        # 统计题目类型
        question_types = {}
        for q in questions:
            rule_type = q.get('metadata', {}).get('rule_type', 'unknown')
            question_types[rule_type] = question_types.get(rule_type, 0) + 1
        
        print("\n题目类型分布:")
        for q_type, count in question_types.items():
            print(f"  - {q_type}: {count} 道")
        
        # 步骤3: 验证和保存
        print("\n[3/3] 保存测试集...")
        output_processor = OutputProcessor()
        output_path = output_processor.save_jsonl(questions, args.output)
        print(f"[OK] 测试集已保存到: {output_path}")
        
        # 统计信息
        file_size = Path(output_path).stat().st_size / 1024  # KB
        print(f"\n文件大小: {file_size:.1f} KB")
        
        # 显示示例题目
        if questions:
            print("\n" + "=" * 60)
            print("示例题目:")
            print("=" * 60)
            sample = questions[0]
            print(f"\n问题: {sample['question']}")
            print("\n选项:")
            for letter, option in sample['options'].items():
                marker = "[*]" if letter == sample['correct_answer'] else "[ ]"
                print(f"  {marker} {letter}. {option}")
            print(f"\n正确答案: {sample['correct_answer']}")
            print(f"解释: {sample['explanation']}")
        
        print("\n" + "=" * 60)
        print(f"完成！生成了 {len(questions)} 道选择题")
        print("=" * 60)
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n[ERROR] 错误: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

