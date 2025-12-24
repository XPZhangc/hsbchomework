#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""合并三个训练数据集的loss曲线到一张图上"""

import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def extract_loss_from_trainer_state(log_dir: Path):
    """从trainer_state.json中提取loss数据"""
    train_losses = []
    eval_losses = []
    train_steps = []
    eval_steps = []
    
    trainer_state_file = log_dir / "trainer_state.json"
    if trainer_state_file.exists():
        try:
            with open(trainer_state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            if 'log_history' in state:
                for log_entry in state['log_history']:
                    if 'loss' in log_entry and 'eval_loss' not in log_entry:
                        train_losses.append(log_entry['loss'])
                        if 'step' in log_entry:
                            train_steps.append(log_entry['step'])
                    
                    if 'eval_loss' in log_entry:
                        eval_losses.append(log_entry['eval_loss'])
                        if 'step' in log_entry:
                            eval_steps.append(log_entry['step'])
        except Exception as e:
            print(f"警告: 从{log_dir}加载loss失败: {e}")
    
    return train_losses, eval_losses, train_steps, eval_steps


def main():
    # 模型输出目录
    base_dir = Path("./output/models/qwen25_05b_lora")
    
    # 三个数据集
    datasets = [
        ("training_data_all", "全部数据"),
        ("training_data_general_llm", "通用+LLM"),
        ("training_data_general_rule", "通用+规则")
    ]
    
    # 提取所有loss数据
    all_data = {}
    for dataset_name, display_name in datasets:
        dataset_dir = base_dir / dataset_name
        if not dataset_dir.exists():
            print(f"警告: 目录不存在: {dataset_dir}")
            continue
        
        train_losses, eval_losses, train_steps, eval_steps = extract_loss_from_trainer_state(dataset_dir)
        
        if not train_losses and not eval_losses:
            print(f"警告: {dataset_name} 没有找到loss数据")
            continue
        
        all_data[display_name] = {
            'train_losses': train_losses,
            'eval_losses': eval_losses,
            'train_steps': train_steps if train_steps else list(range(len(train_losses))),
            'eval_steps': eval_steps if eval_steps else list(range(len(eval_losses)))
        }
        print(f"已加载 {display_name}: 训练loss {len(train_losses)} 个点, 验证loss {len(eval_losses)} 个点")
    
    if not all_data:
        print("错误: 没有找到任何loss数据")
        sys.exit(1)
    
    # 绘制合并的loss曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # 绘制训练loss
    for idx, (display_name, data) in enumerate(all_data.items()):
        if data['train_losses']:
            ax1.plot(data['train_steps'], data['train_losses'], 
                    label=f'{display_name} (训练)', 
                    color=colors[idx % len(colors)],
                    linestyle='-', linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('训练步数 (Steps)', fontsize=12)
    ax1.set_ylabel('训练Loss', fontsize=12)
    ax1.set_title('训练Loss曲线对比', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 绘制验证loss
    for idx, (display_name, data) in enumerate(all_data.items()):
        if data['eval_losses']:
            ax2.plot(data['eval_steps'], data['eval_losses'], 
                    label=f'{display_name} (验证)', 
                    color=colors[idx % len(colors)],
                    linestyle='--', linewidth=2, alpha=0.8, marker='o', markersize=4)
    
    ax2.set_xlabel('训练步数 (Steps)', fontsize=12)
    ax2.set_ylabel('验证Loss', fontsize=12)
    ax2.set_title('验证Loss曲线对比', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存合并的loss曲线
    output_file = base_dir / "combined_loss_curves.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n合并的loss曲线已保存到: {output_file}")
    
    # 删除原来的单图
    deleted_count = 0
    for dataset_name, display_name in datasets:
        old_file = base_dir / dataset_name / "loss_curve.png"
        if old_file.exists():
            old_file.unlink()
            deleted_count += 1
            print(f"已删除: {old_file}")
    
    print(f"\n共删除 {deleted_count} 个原始loss曲线图")
    print("完成！")


if __name__ == '__main__':
    main()
