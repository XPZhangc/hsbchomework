"""
Qwen模型LoRA微调训练脚本（使用LLaMA-Factory）
支持GitHub直接下载运行（使用相对路径）

功能特性:
- ✅ 自动处理training_datasets和validation_datasets目录下的所有数据集
- ✅ 自动将JSONL格式转换为ShareGPT格式
- ✅ 每个epoch结束后在验证集上评估
- ✅ 自动保存验证loss最低的checkpoint
- ✅ 自动绘制loss收敛曲线
- ✅ 默认10个epoch训练
- ✅ 使用LoRA微调
- ✅ 使用LLaMA-Factory训练框架

使用方法:
    python train_qwen_lora.py
    python train_qwen_lora.py --model-path ./models/qwen25_05b
"""
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import shutil
import time

print("="*60, file=sys.stderr, flush=True)
print("脚本开始执行...", file=sys.stderr, flush=True)
print("="*60, file=sys.stderr, flush=True)

print("[1/3] 导入标准库完成", file=sys.stderr, flush=True)

print("[2/3] 正在导入torch（首次导入可能需要10-30秒初始化CUDA，请耐心等待）...", file=sys.stderr, flush=True)
start_time = time.time()
try:
    import torch
    elapsed = time.time() - start_time
    print(f"[2/3] torch导入成功（耗时: {elapsed:.2f}秒）: {torch.__version__}", file=sys.stderr, flush=True)
except ImportError as e:
    print("错误: 请安装torch: pip install torch", file=sys.stderr, flush=True)
    print(f"详细错误: {e}", file=sys.stderr, flush=True)
    sys.exit(1)
except Exception as e:
    elapsed = time.time() - start_time
    print(f"错误: 导入torch时出错（耗时: {elapsed:.2f}秒）: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

print("[3/3] 正在导入LLaMA-Factory...", file=sys.stderr, flush=True)
start_time = time.time()
try:
    from llamafactory.train.tuner import run_exp
    from llamafactory.hparams import ModelArguments, DataArguments, FinetuningArguments, TrainingArguments
    elapsed = time.time() - start_time
    print(f"[3/3] LLaMA-Factory导入成功（耗时: {elapsed:.2f}秒）", file=sys.stderr, flush=True)
except ImportError as e:
    print("错误: 请安装LLaMA-Factory:", file=sys.stderr, flush=True)
    print("  pip install llamafactory", file=sys.stderr, flush=True)
    print(f"详细错误: {e}", file=sys.stderr, flush=True)
    sys.exit(1)
except Exception as e:
    elapsed = time.time() - start_time
    print(f"错误: 导入LLaMA-Factory时出错（耗时: {elapsed:.2f}秒）: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

print("所有导入完成，进入main函数...", file=sys.stderr, flush=True)

try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: 未安装matplotlib，无法绘制loss曲线。运行: pip install matplotlib")


def load_jsonl_dataset(file_path: Path) -> List[Dict[str, Any]]:
    """加载JSONL格式的数据集"""
    data = []
    if not file_path.exists():
        raise FileNotFoundError(f"数据集文件不存在: {file_path}")
    
    print(f"正在加载数据集: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
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


def load_all_datasets(dataset_dir: Path) -> List[Dict[str, Any]]:
    """加载目录下所有JSONL数据集文件"""
    all_data = []
    
    if not dataset_dir.exists():
        print(f"警告: 数据集目录不存在: {dataset_dir}")
        return all_data
    
    jsonl_files = list(dataset_dir.glob("*.jsonl"))
    if len(jsonl_files) == 0:
        print(f"警告: 在 {dataset_dir} 中未找到JSONL文件")
        return all_data
    
    print(f"\n找到 {len(jsonl_files)} 个数据集文件:")
    for jsonl_file in jsonl_files:
        print(f"  - {jsonl_file.name}")
        data = load_jsonl_dataset(jsonl_file)
        all_data.extend(data)
    
    print(f"\n总共加载 {len(all_data)} 条数据")
    return all_data


def find_dataset_pairs(train_dir: Path, val_dir: Path) -> List[Tuple[Path, Path, str]]:
    """
    找到训练集和验证集中的对应文件对
    
    Returns:
        List of (train_file, val_file, dataset_name) tuples
    """
    pairs = []
    matched_val_stems = set()  # 记录已匹配的验证集文件，避免重复
    
    if not train_dir.exists() or not val_dir.exists():
        return pairs
    
    train_files = {f.stem: f for f in train_dir.glob("*.jsonl")}
    val_files = {f.stem: f for f in val_dir.glob("*.jsonl")}
    
    # 匹配训练集和验证集文件
    # 例如: training_data_all_train.jsonl 对应 training_data_all_val.jsonl
    # 通过去除 _train 和 _val 后缀来匹配
    
    for train_stem, train_file in train_files.items():
        # 提取基础名称（去除_train后缀）
        if train_stem.endswith("_train"):
            dataset_name = train_stem[:-6]  # 移除 "_train"
            val_stem = dataset_name + "_val"
            
            if val_stem in val_files and val_stem not in matched_val_stems:
                pairs.append((train_file, val_files[val_stem], dataset_name))
                matched_val_stems.add(val_stem)
    
    # 按数据集名称排序，确保训练顺序一致
    pairs.sort(key=lambda x: x[2])
    
    return pairs


def convert_to_sharegpt_format(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将JSONL格式的数据转换为ShareGPT格式
    
    ShareGPT格式:
    {
        "conversation": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    """
    sharegpt_data = []
    skipped = 0
    
    system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant specialized in code analysis and software design."
    
    for item in data:
        if not item or len(item) == 0:
            skipped += 1
            continue
        
        conversation = [{"role": "system", "content": system_message}]
        
        data_type = item.get('type', 'qa')
        
        if data_type == 'qa':
            question = item.get('question', '')
            answer = item.get('answer', '')
            if not question or not answer:
                skipped += 1
                continue
            
            # 构建用户消息（包含代码片段）
            user_content = question
            code_snippets = item.get('code_snippets', [])
            if code_snippets and len(code_snippets) > 0:
                code_text = "\n相关代码:\n"
                for snippet in code_snippets[:3]:  # 最多使用3个代码片段
                    file_path = snippet.get('file_path', '')
                    code = snippet.get('code', '')
                    if code:
                        code_text += f"文件: {file_path}\n```python\n{code}\n```\n\n"
                user_content = code_text + user_content
            
            conversation.append({"role": "user", "content": user_content})
            conversation.append({"role": "assistant", "content": answer})
        
        elif data_type == 'design':
            demand = item.get('demand', '')
            scheme = item.get('scheme', '')
            if not demand or not scheme:
                skipped += 1
                continue
            
            # 构建用户消息（包含代码片段）
            user_content = f"设计需求: {demand}"
            code_snippets = item.get('code_snippets', [])
            if code_snippets and len(code_snippets) > 0:
                code_text = "\n参考代码:\n"
                for snippet in code_snippets[:3]:
                    file_path = snippet.get('file_path', '')
                    code = snippet.get('code', '')
                    if code:
                        code_text += f"文件: {file_path}\n```python\n{code}\n```\n\n"
                user_content = code_text + user_content
            
            conversation.append({"role": "user", "content": user_content})
            conversation.append({"role": "assistant", "content": scheme})
        
        else:
            skipped += 1
            continue
        
        sharegpt_data.append({"conversation": conversation})
    
    if skipped > 0:
        print(f"跳过无效数据: {skipped} 条")
    print(f"已转换 {len(sharegpt_data)} 条ShareGPT格式数据")
    
    return sharegpt_data


def save_sharegpt_dataset(data: List[Dict[str, Any]], output_file: Path):
    """保存为ShareGPT格式的JSON文件"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"ShareGPT格式数据已保存到: {output_file}")


def resolve_model_path(model_path: str) -> str:
    """解析模型路径，支持相对路径和绝对路径"""
    # 如果是绝对路径且存在，直接返回
    if os.path.isabs(model_path) and Path(model_path).exists():
        return model_path
    
    # 尝试多个可能的相对路径
    possible_paths = [
        Path.cwd() / model_path,
        Path(__file__).parent / model_path,
        Path(__file__).parent.parent / model_path,
    ]
    
    for path in possible_paths:
        abs_path = path.resolve()
        if abs_path.exists():
            print(f"找到模型路径: {abs_path}")
            return str(abs_path)
    
    # 如果都找不到，检查是否是HuggingFace模型ID
    if '/' in model_path and not model_path.startswith('http'):
        return model_path
    
    print(f"警告: 模型路径不存在: {model_path}")
    print(f"将尝试从HuggingFace下载（如果是模型ID）")
    return model_path


def plot_loss_curves(train_losses: List[float], eval_losses: List[float], 
                     output_file: Path, train_steps: List[int] = None, 
                     eval_steps: List[int] = None):
    """绘制loss收敛曲线"""
    if not HAS_MATPLOTLIB:
        print("警告: matplotlib未安装，跳过绘制loss曲线")
        return
    
    plt.figure(figsize=(12, 6))
    
    # 如果没有steps，使用索引
    if not train_steps:
        train_steps = list(range(len(train_losses)))
    if not eval_steps:
        eval_steps = list(range(len(eval_losses)))
    
    # 绘制训练loss
    if train_losses:
        plt.plot(train_steps[:len(train_losses)], train_losses, 
                label='Train Loss', marker='o', markersize=3, linewidth=1.5, alpha=0.7)
    
    # 绘制验证loss
    if eval_losses:
        plt.plot(eval_steps[:len(eval_losses)], eval_losses, 
                label='Validation Loss', marker='s', markersize=4, linewidth=2, alpha=0.8)
    
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Loss曲线已保存到: {output_file}")
    plt.close()


def extract_loss_from_llamafactory_logs(log_dir: Path) -> Tuple[List[float], List[float], List[int], List[int]]:
    """从LLaMA-Factory的训练日志中提取loss数据"""
    train_losses = []
    eval_losses = []
    train_steps = []
    eval_steps = []
    
    # 查找trainer_state.json
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
            print(f"警告: 从trainer_state.json加载loss失败: {e}")
    
    return train_losses, eval_losses, train_steps, eval_steps


def main():
    parser = argparse.ArgumentParser(
        description='Qwen模型LoRA微调训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认参数训练
  python train_qwen_lora.py
  
  # 自定义模型路径
  python train_qwen_lora.py --model-path ./models/qwen25_05b
        """
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='./models/qwen25_05b',
        help='模型路径（默认: ./models/qwen25_05b，也可以是HuggingFace模型ID）'
    )
    parser.add_argument(
        '--train-dir',
        type=str,
        default='./output/training_datasets',
        help='训练数据集目录（默认: ./output/training_datasets）'
    )
    parser.add_argument(
        '--val-dir',
        type=str,
        default='./output/validation_datasets',
        help='验证数据集目录（默认: ./output/validation_datasets）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output/models/qwen25_05b_lora',
        help='输出目录（默认: ./output/models/qwen25_05b_lora）'
    )
    parser.add_argument(
        '--lora-r',
        type=int,
        default=8,
        help='LoRA rank（默认: 8）'
    )
    parser.add_argument(
        '--lora-alpha',
        type=int,
        default=32,
        help='LoRA alpha（默认: 32）'
    )
    parser.add_argument(
        '--lora-dropout',
        type=float,
        default=0.1,
        help='LoRA dropout（默认: 0.1）'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=2048,
        help='最大序列长度（默认: 2048）'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='训练批次大小（默认: 4）'
    )
    parser.add_argument(
        '--gradient-accumulation-steps',
        type=int,
        default=4,
        help='梯度累积步数（默认: 4）'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-4,
        help='学习率（默认: 2e-4）'
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=10,
        help='训练轮数（默认: 10）'
    )
    parser.add_argument(
        '--save-steps',
        type=int,
        default=500,
        help='保存检查点的步数（默认: 500）'
    )
    parser.add_argument(
        '--eval-steps',
        type=int,
        default=500,
        help='评估步数（默认: 500）'
    )
    parser.add_argument(
        '--logging-steps',
        type=int,
        default=10,
        help='日志记录步数（默认: 10）'
    )
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=100,
        help='预热步数（默认: 100）'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='使用FP16精度训练（节省显存）'
    )
    
    args = parser.parse_args()
    
    # 打印配置信息
    print("="*60)
    print("Qwen模型LoRA微调训练")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模型路径: {args.model_path}")
    print(f"训练集目录: {args.train_dir}")
    print(f"验证集目录: {args.val_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"梯度累积: {args.gradient_accumulation_steps}")
    print(f"学习率: {args.learning_rate}")
    print(f"最大长度: {args.max_length}")
    print("="*60)
    
    # 解析模型路径
    model_path = resolve_model_path(args.model_path)
    
    # 确定设备（延迟CUDA检查，避免在导入时就初始化）
    print("\n检查CUDA可用性...", flush=True)
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {device}")
        if device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    except Exception as e:
        print(f"检查CUDA时出错，将使用CPU: {e}")
        device = 'cpu'
    
    try:
        # 步骤1: 找到训练集和验证集的对应文件对
        print("\n[步骤1] 查找数据集文件对...")
        train_dir = Path(args.train_dir)
        val_dir = Path(args.val_dir)
        
        dataset_pairs = find_dataset_pairs(train_dir, val_dir)
        
        if len(dataset_pairs) == 0:
            raise ValueError(f"未找到匹配的训练集和验证集文件对！\n训练集目录: {train_dir}\n验证集目录: {val_dir}")
        
        print(f"\n找到 {len(dataset_pairs)} 个数据集对:")
        for i, (train_file, val_file, dataset_name) in enumerate(dataset_pairs, 1):
            print(f"  {i}. {dataset_name}")
            print(f"     训练集: {train_file.name}")
            print(f"     验证集: {val_file.name}")
        
        # 解析模型路径（只解析一次）
        resolved_model_path = resolve_model_path(args.model_path)
        print(f"\n使用模型路径: {resolved_model_path}")
        
        # 确定设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n使用设备: {device}")
        if device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # 创建ShareGPT数据集目录
        sharegpt_dir = Path(args.output_dir).parent / "llamafactory_datasets"
        sharegpt_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建dataset_info.json
        dataset_info = {}
        
        # 对每个数据集对进行训练
        for dataset_idx, (train_file, val_file, dataset_name) in enumerate(dataset_pairs, 1):
            print("\n" + "="*60)
            print(f"开始训练数据集 {dataset_idx}/{len(dataset_pairs)}: {dataset_name}")
            print("="*60)
            
            # 步骤2: 加载数据集并转换为ShareGPT格式
            print(f"\n[步骤2] 加载并转换数据集 {dataset_name}...")
            train_data = load_jsonl_dataset(train_file)
            if len(train_data) == 0:
                print(f"警告: 训练数据集为空: {train_file}，跳过")
                continue
            
            val_data = load_jsonl_dataset(val_file)
            if len(val_data) == 0:
                print(f"警告: 验证数据集为空: {val_file}，将跳过验证")
                val_data = None
            
            # 转换为ShareGPT格式
            train_sharegpt = convert_to_sharegpt_format(train_data)
            train_sharegpt_file = sharegpt_dir / f"{dataset_name}_train.json"
            save_sharegpt_dataset(train_sharegpt, train_sharegpt_file)
            
            val_sharegpt_file = None
            if val_data:
                val_sharegpt = convert_to_sharegpt_format(val_data)
                val_sharegpt_file = sharegpt_dir / f"{dataset_name}_val.json"
                save_sharegpt_dataset(val_sharegpt, val_sharegpt_file)
            
            # 添加到dataset_info
            dataset_info[f"{dataset_name}_train"] = {
                "file_name": str(train_sharegpt_file.absolute()),
                "columns": {"messages": "conversation"},
                "formatting": "sharegpt",
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                    "system_tag": "system"
                }
            }
            
            if val_sharegpt_file:
                dataset_info[f"{dataset_name}_val"] = {
                    "file_name": str(val_sharegpt_file.absolute()),
                    "columns": {"messages": "conversation"},
                    "formatting": "sharegpt",
                    "tags": {
                        "role_tag": "role",
                        "content_tag": "content",
                        "user_tag": "user",
                        "assistant_tag": "assistant",
                        "system_tag": "system"
                    }
                }
            
            # 步骤3: 配置LLaMA-Factory训练参数
            print(f"\n[步骤3] 配置LLaMA-Factory训练参数...")
            output_dir = Path(args.output_dir) / dataset_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存dataset_info.json
            dataset_info_file = sharegpt_dir / "dataset_info.json"
            with open(dataset_info_file, 'w', encoding='utf-8') as f:
                json.dump(dataset_info, f, ensure_ascii=False, indent=2)
            
            # 配置LLaMA-Factory参数（使用字典格式）
            # LLaMA-Factory的run_exp接受一个字典，包含所有参数
            llamafactory_args = {
                # ModelArguments
                'model_name_or_path': resolved_model_path,
                'trust_remote_code': True,
                
                # DataArguments
                'dataset': f"{dataset_name}_train",  # 训练数据集
                'eval_dataset': f"{dataset_name}_val" if val_sharegpt_file else None,  # 评估数据集（单独指定）
                'dataset_dir': str(sharegpt_dir),
                'cutoff_len': args.max_length,
                'val_size': 0.0,  # 不使用内置验证集分割，使用单独的验证集文件
                
                # FinetuningArguments
                'finetuning_type': 'lora',
                'lora_target': 'q_proj,k_proj,v_proj,o_proj',
                'lora_rank': args.lora_r,
                'lora_alpha': args.lora_alpha,
                'lora_dropout': args.lora_dropout,
                'stage': 'sft',  # 监督微调
                
                # TrainingArguments
                'output_dir': str(output_dir),
                'do_train': True,  # 必须明确指定进行训练
                'do_eval': True if val_sharegpt_file else False,  # 如果有验证集则进行评估
                'num_train_epochs': args.num_epochs,
                'per_device_train_batch_size': args.batch_size,
                'per_device_eval_batch_size': args.batch_size,
                'gradient_accumulation_steps': args.gradient_accumulation_steps,
                'learning_rate': args.learning_rate,
                'warmup_steps': args.warmup_steps,
                'logging_steps': args.logging_steps,
                'save_steps': args.save_steps,
                'eval_steps': args.eval_steps,
                'eval_strategy': "epoch" if val_sharegpt_file else "no",
                'save_strategy': "epoch",
                'save_total_limit': 3,
                'load_best_model_at_end': True if val_sharegpt_file else False,
                'metric_for_best_model': "eval_loss" if val_sharegpt_file else None,
                'greater_is_better': False,
                'fp16': args.fp16,
                'logging_dir': str(output_dir / "logs"),
                'report_to': "tensorboard",
            }
            
            # 步骤4: 开始训练
            print(f"\n[步骤4] 开始训练数据集 {dataset_name}...")
            print("="*60)
            print("使用LLaMA-Factory进行训练...")
            print("="*60)
            
            try:
                run_exp(llamafactory_args)
            except Exception as e:
                print(f"训练过程中出错: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # 步骤5: 提取loss并绘制曲线
            print(f"\n[步骤5] 提取loss并绘制曲线 {dataset_name}...")
            train_losses, eval_losses, train_steps, eval_steps = extract_loss_from_llamafactory_logs(output_dir)
            
            if train_losses or eval_losses:
                loss_curve_file = output_dir / "loss_curve.png"
                plot_loss_curves(train_losses, eval_losses, loss_curve_file, train_steps, eval_steps)
                
                # 找到最佳验证loss
                best_eval_loss = min(eval_losses) if eval_losses else None
                if best_eval_loss:
                    print(f"最佳验证loss: {best_eval_loss:.4f}")
            else:
                print("警告: 未找到loss数据")
            
            print(f"\n数据集 {dataset_name} 训练完成！")
            print(f"模型保存位置: {output_dir}")
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"已清理GPU内存，准备下一个数据集训练...")
        
        print("\n" + "="*60)
        print("所有数据集训练完成！")
        print("="*60)
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"共训练了 {len(dataset_pairs)} 个数据集")
        print(f"输出目录: {Path(args.output_dir)}")
        
    except Exception as e:
        print(f"\n[ERROR] 训练失败: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断", file=sys.stderr, flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"\n未捕获的错误: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
