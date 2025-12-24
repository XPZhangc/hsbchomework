"""
Qwen模型微调训练脚本
支持LoRA/QLoRA微调，使用生成的训练数据集

使用方法:
    python train_model.py --dataset ./output/training_datasets/training_data_general_rule.jsonl
    python train_model.py --dataset ./output/training_datasets/training_data_all.jsonl --use-lora
"""
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        TrainingArguments, Trainer,
        DataCollatorForLanguageModeling
    )
    from datasets import Dataset
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("错误: 请安装必要的依赖:")
    print("  pip install transformers torch datasets peft bitsandbytes")
    sys.exit(1)

try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("警告: 未安装peft库，无法使用LoRA微调。运行: pip install peft")


def load_jsonl_dataset(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL格式的训练数据集"""
    data = []
    if not Path(file_path).exists():
        raise FileNotFoundError(f"数据集文件不存在: {file_path}")
    
    print(f"正在加载数据集: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                # 跳过空对象
                if not item or len(item) == 0:
                    continue
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"警告: 第{line_num}行JSON解析失败: {str(e)}")
                continue
    
    print(f"已加载 {len(data)} 条有效数据")
    return data


def format_qa_prompt(question: str, answer: str, code_snippets: List[Dict] = None) -> str:
    """
    格式化问答对为训练提示词（Qwen格式）
    
    Args:
        question: 问题
        answer: 答案
        code_snippets: 代码片段列表（可选）
    
    Returns:
        格式化后的提示词
    """
    # Qwen格式：使用<|im_start|>和<|im_end|>标记
    user_content = question
    assistant_content = answer
    
    # 如果有代码片段，添加到用户内容中
    if code_snippets and len(code_snippets) > 0:
        code_text = "\n相关代码:\n"
        for snippet in code_snippets[:3]:  # 最多使用3个代码片段
            file_path = snippet.get('file_path', '')
            code = snippet.get('code', '')
            if code:
                code_text += f"文件: {file_path}\n```python\n{code}\n```\n\n"
        user_content = code_text + user_content
    
    # Qwen格式（添加system消息）
    prompt = (
        "<|im_start|>system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant specialized in code analysis and software design.<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_content}<|im_end|>"
    )
    
    return prompt


def format_design_prompt(demand: str, scheme: str, code_snippets: List[Dict] = None) -> str:
    """
    格式化设计方案为训练提示词（Qwen格式）
    
    Args:
        demand: 设计需求
        scheme: 设计方案
        code_snippets: 代码片段列表（可选）
    
    Returns:
        格式化后的提示词
    """
    user_content = f"设计需求: {demand}"
    assistant_content = scheme
    
    # 如果有代码片段，添加到用户内容中
    if code_snippets and len(code_snippets) > 0:
        code_text = "\n参考代码:\n"
        for snippet in code_snippets[:3]:
            file_path = snippet.get('file_path', '')
            code = snippet.get('code', '')
            if code:
                code_text += f"文件: {file_path}\n```python\n{code}\n```\n\n"
        user_content = code_text + user_content
    
    # Qwen格式（添加system消息）
    prompt = (
        "<|im_start|>system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant specialized in code analysis and software design.<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_content}<|im_end|>"
    )
    
    return prompt


def prepare_training_data(data: List[Dict[str, Any]], tokenizer, max_length: int = 2048) -> Dataset:
    """
    准备训练数据，转换为模型输入格式
    
    Args:
        data: 原始数据列表
        tokenizer: tokenizer
        max_length: 最大序列长度
    
    Returns:
        处理后的数据集
    """
    print("\n准备训练数据...")
    
    texts = []
    skipped = 0
    
    for item in data:
        # 跳过空数据
        if not item or len(item) == 0:
            skipped += 1
            continue
        
        data_type = item.get('type', 'qa')
        
        if data_type == 'qa':
            question = item.get('question', '')
            answer = item.get('answer', '')
            if not question or not answer:
                skipped += 1
                continue
            code_snippets = item.get('code_snippets', [])
            text = format_qa_prompt(question, answer, code_snippets)
        
        elif data_type == 'design':
            demand = item.get('demand', '')
            scheme = item.get('scheme', '')
            if not demand or not scheme:
                skipped += 1
                continue
            code_snippets = item.get('code_snippets', [])
            text = format_design_prompt(demand, scheme, code_snippets)
        
        else:
            # 未知类型，跳过
            skipped += 1
            continue
        
        texts.append(text)
    
    if skipped > 0:
        print(f"跳过无效数据: {skipped} 条")
    print(f"已准备 {len(texts)} 条训练文本")
    
    if len(texts) == 0:
        raise ValueError("没有有效的训练数据！")
    
    # Tokenize
    print("正在tokenize数据...")
    def tokenize_function(examples):
        # 对于因果语言模型，labels就是input_ids
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )
        # 设置labels（因果语言模型需要）
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized
    
    dataset = Dataset.from_dict({'text': texts})
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text'],
        desc="Tokenizing"
    )
    
    print(f"Tokenize完成，样本数: {len(tokenized_dataset)}")
    
    return tokenized_dataset


def resolve_model_path(model_path: str) -> str:
    """
    解析模型路径，支持相对路径和绝对路径
    
    Args:
        model_path: 模型路径
    
    Returns:
        解析后的绝对路径
    """
    from pathlib import Path
    
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
    
    # 如果都找不到，检查是否是HuggingFace模型ID（包含/但不以http开头）
    if '/' in model_path and not model_path.startswith('http'):
        # 可能是HuggingFace模型ID，返回原路径让transformers处理
        return model_path
    
    # 如果都找不到且不是HuggingFace模型ID，给出警告
    print(f"警告: 模型路径不存在: {model_path}")
    print(f"将尝试从HuggingFace下载（如果是模型ID）")
    return model_path


def main():
    parser = argparse.ArgumentParser(
        description='Qwen模型微调训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认参数训练
  python train_model.py --dataset ./output/training_datasets/training_data_general_rule.jsonl
  
  # 使用LoRA微调（推荐，节省显存）
  python train_model.py --dataset ./output/training_datasets/training_data_all.jsonl --use-lora
  
  # 使用QLoRA微调（最节省显存）
  python train_model.py --dataset ./output/training_datasets/training_data_all.jsonl --use-qlora
  
  # 自定义模型路径和输出目录
  python train_model.py \\
    --dataset ./output/training_datasets/training_data_general_rule.jsonl \\
    --model-path ./models/qwen25_05b \\
    --output-dir ./output/checkpoints/qwen25_05b_finetuned
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='训练数据集路径（JSONL格式）'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='./models/qwen25_05b',
        help='模型路径（默认: ./models/qwen25_05b，也可以是HuggingFace模型ID）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output/checkpoints',
        help='输出目录（默认: ./output/checkpoints）'
    )
    parser.add_argument(
        '--use-lora',
        action='store_true',
        help='使用LoRA微调（推荐，节省显存）'
    )
    parser.add_argument(
        '--use-qlora',
        action='store_true',
        help='使用QLoRA微调（最节省显存，需要bitsandbytes）'
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
        default=3,
        help='训练轮数（默认: 3）'
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
        default=50,
        help='日志记录步数（默认: 50）'
    )
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=100,
        help='预热步数（默认: 100）'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='设备（cuda/cpu），默认自动选择'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='使用FP16精度训练（节省显存）'
    )
    parser.add_argument(
        '--bf16',
        action='store_true',
        help='使用BF16精度训练（需要支持BF16的GPU）'
    )
    
    args = parser.parse_args()
    
    # 检查依赖
    if not HAS_TRANSFORMERS:
        print("错误: 请安装transformers和相关依赖")
        sys.exit(1)
    
    if (args.use_lora or args.use_qlora) and not HAS_PEFT:
        print("错误: 使用LoRA/QLoRA需要安装peft库: pip install peft")
        sys.exit(1)
    
    if args.use_qlora:
        try:
            import bitsandbytes
        except ImportError:
            print("错误: 使用QLoRA需要安装bitsandbytes: pip install bitsandbytes")
            sys.exit(1)
    
    # 打印配置信息
    print("="*60)
    print("Qwen模型微调训练")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据集: {args.dataset}")
    print(f"模型路径: {args.model_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"微调方法: {'QLoRA' if args.use_qlora else 'LoRA' if args.use_lora else '全量微调'}")
    print(f"批次大小: {args.batch_size}")
    print(f"梯度累积: {args.gradient_accumulation_steps}")
    print(f"学习率: {args.learning_rate}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"最大长度: {args.max_length}")
    print("="*60)
    
    # 解析模型路径
    model_path = resolve_model_path(args.model_path)
    
    # 确定设备
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    try:
        # 步骤1: 加载数据集
        print("\n[步骤1] 加载数据集...")
        raw_data = load_jsonl_dataset(args.dataset)
        
        if len(raw_data) == 0:
            raise ValueError("数据集为空！")
        
        # 步骤2: 加载模型和tokenizer
        print("\n[步骤2] 加载模型和tokenizer...")
        
        # 解析模型路径
        resolved_model_path = resolve_model_path(args.model_path)
        print(f"使用模型路径: {resolved_model_path}")
        
        # 检查是否是本地路径
        path_obj = Path(resolved_model_path)
        is_local_path = path_obj.exists() and path_obj.is_dir()
        
        # 如果是HuggingFace模型ID（包含/但不以http开头），允许下载
        is_hf_model_id = '/' in resolved_model_path and not resolved_model_path.startswith('http') and not is_local_path
        
        if is_hf_model_id:
            print("检测到HuggingFace模型ID，将从网络下载")
        elif is_local_path:
            print(f"使用本地模型: {resolved_model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            resolved_model_path,
            trust_remote_code=True,
            local_files_only=is_local_path and not is_hf_model_id
        )
        
        # 设置pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载模型
        model_kwargs = {
            'trust_remote_code': True,
        }
        
        if is_local_path and not is_hf_model_id:
            model_kwargs['local_files_only'] = True
        
        if args.use_qlora:
            # QLoRA需要4bit量化
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs['quantization_config'] = bnb_config
            model_kwargs['device_map'] = 'auto'
        elif args.use_lora:
            # LoRA可以使用float16
            if args.fp16:
                model_kwargs['torch_dtype'] = torch.float16
            model_kwargs['device_map'] = 'auto'
        else:
            # 全量微调
            if args.fp16:
                model_kwargs['torch_dtype'] = torch.float16
            elif args.bf16:
                model_kwargs['torch_dtype'] = torch.bfloat16
        
        print("正在加载模型（这可能需要几分钟）...")
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_path,
            **model_kwargs
        )
        
        # 如果device_map未使用，手动移动到设备
        if 'device_map' not in model_kwargs or model_kwargs.get('device_map') is None:
            model = model.to(device)
        
        # 配置LoRA
        if args.use_lora or args.use_qlora:
            print("\n[步骤3] 配置LoRA...")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen的注意力层
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        # 步骤3: 准备训练数据
        print("\n[步骤3] 准备训练数据...")
        train_dataset = prepare_training_data(raw_data, tokenizer, max_length=args.max_length)
        
        # 步骤4: 配置训练参数
        print("\n[步骤4] 配置训练参数...")
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            save_total_limit=3,
            fp16=args.fp16 and not args.use_qlora,  # QLoRA已经量化，不需要fp16
            bf16=args.bf16 and not args.use_qlora,
            logging_dir=f"{args.output_dir}/logs",
            report_to="tensorboard" if Path(args.output_dir).exists() else None,
            remove_unused_columns=False,
            dataloader_pin_memory=True,
        )
        
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # 因果语言模型，不使用MLM
        )
        
        # 步骤5: 创建Trainer
        print("\n[步骤5] 创建Trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # 步骤6: 开始训练
        print("\n[步骤6] 开始训练...")
        print("="*60)
        trainer.train()
        
        # 步骤7: 保存模型
        print("\n[步骤7] 保存模型...")
        if args.use_lora or args.use_qlora:
            # LoRA模型保存adapter
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"LoRA adapter已保存到: {args.output_dir}")
        else:
            # 全量微调保存完整模型
            trainer.save_model()
            tokenizer.save_pretrained(args.output_dir)
            print(f"完整模型已保存到: {args.output_dir}")
        
        print("\n" + "="*60)
        print("训练完成！")
        print("="*60)
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"模型保存位置: {args.output_dir}")
        
    except Exception as e:
        print(f"\n[ERROR] 训练失败: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
