"""
查看选择题测试集
"""
import json
import sys
from pathlib import Path


def view_test_set(file_path="output/test_set.jsonl", num_samples=5):
    """查看测试集"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"错误: 文件不存在: {file_path}")
        return
    
    print("=" * 60)
    print("选择题测试集统计")
    print("=" * 60)
    
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    
    print(f"\n总题目数: {len(questions)}")
    
    # 统计题目类型
    type_counts = {}
    for q in questions:
        rule_type = q.get('metadata', {}).get('rule_type', 'unknown')
        type_counts[rule_type] = type_counts.get(rule_type, 0) + 1
    
    print("\n题目类型分布:")
    for rule_type, count in sorted(type_counts.items()):
        print(f"  - {rule_type}: {count} 道")
    
    # 显示示例题目
    print("\n" + "=" * 60)
    print(f"示例题目（前 {min(num_samples, len(questions))} 道）")
    print("=" * 60)
    
    for i, question in enumerate(questions[:num_samples], 1):
        print(f"\n题目 {i}:")
        print(f"  问题: {question['question']}")
        print("\n  选项:")
        for letter in ['A', 'B', 'C', 'D']:
            marker = "[*]" if letter == question['correct_answer'] else "[ ]"
            option = question['options'][letter]
            print(f"    {marker} {letter}. {option}")
        print(f"\n  正确答案: {question['correct_answer']}")
        print(f"  解释: {question['explanation']}")
        if question.get('code_snippets'):
            print(f"  代码文件: {question['code_snippets'][0]['file_path']}")
        print("-" * 60)


if __name__ == '__main__':
    file_path = sys.argv[1] if len(sys.argv) > 1 else "output/test_set.jsonl"
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    view_test_set(file_path, num_samples)

