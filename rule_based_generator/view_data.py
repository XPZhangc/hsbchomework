"""
查看生成的训练数据
"""
import json
import sys
from pathlib import Path

def view_data(file_path="output/training_data.jsonl"):
    """查看训练数据"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"错误: 文件不存在: {file_path}")
        return
    
    print("=" * 60)
    print("训练数据统计")
    print("=" * 60)
    
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"\n总样本数: {len(samples)}")
    
    # 统计
    qa_count = sum(1 for s in samples if s['type'] == 'qa')
    design_count = sum(1 for s in samples if s['type'] == 'design')
    
    print(f"  - QA样本: {qa_count}")
    print(f"  - 设计方案: {design_count}")
    
    # 显示示例
    print("\n" + "=" * 60)
    print("示例样本")
    print("=" * 60)
    
    for i, sample in enumerate(samples[:3], 1):
        print(f"\n样本 {i}:")
        print(f"  类型: {sample['type']}")
        
        if sample['type'] == 'qa':
            print(f"  问题: {sample['question']}")
            print(f"  答案: {sample['answer'][:150]}...")
        else:
            print(f"  需求: {sample['demand']}")
            print(f"  方案: {sample['scheme'][:150]}...")
        
        print(f"  代码片段数: {len(sample['code_snippets'])}")
        if sample['code_snippets']:
            print(f"  文件: {sample['code_snippets'][0]['file_path']}")

if __name__ == '__main__':
    file_path = sys.argv[1] if len(sys.argv) > 1 else "output/training_data.jsonl"
    view_data(file_path)

