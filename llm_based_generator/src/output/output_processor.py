"""
输出处理器：格式化为JSONL训练集，支持导出
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional


class OutputProcessor:
    """处理训练数据输出"""
    
    def __init__(self):
        self.output_format = 'jsonl'
    
    def save_jsonl(self, samples: List[Dict[str, Any]], output_path: str) -> str:
        """
        保存为JSONL格式
        
        Args:
            samples: 样本列表
            output_path: 输出文件路径
            
        Returns:
            保存的文件路径
        """
        # 确保输出目录存在
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入JSONL文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                json_line = json.dumps(sample, ensure_ascii=False)
                f.write(json_line + '\n')
        
        return str(output_file)
    
    def save_json(self, samples: List[Dict[str, Any]], output_path: str) -> str:
        """
        保存为JSON格式（数组）
        
        Args:
            samples: 样本列表
            output_path: 输出文件路径
            
        Returns:
            保存的文件路径
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        return str(output_file)
    
    def convert_to_huggingface(self, samples: List[Dict[str, Any]], 
                              output_dir: str) -> Dict[str, str]:
        """
        转换为Hugging Face数据集格式
        
        Args:
            samples: 样本列表
            output_dir: 输出目录
            
        Returns:
            生成的文件路径字典
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存为JSONL（Hugging Face标准格式）
        jsonl_path = output_path / 'train.jsonl'
        self.save_jsonl(samples, str(jsonl_path))
        
        # 创建数据集信息文件（可选）
        info = {
            'total_samples': len(samples),
            'qa_samples': sum(1 for s in samples if s['type'] == 'qa'),
            'design_samples': sum(1 for s in samples if s['type'] == 'design'),
            'format': 'jsonl'
        }
        
        info_path = output_path / 'dataset_info.json'
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        
        return {
            'data_file': str(jsonl_path),
            'info_file': str(info_path)
        }
    
    def format_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化单个样本，确保符合标准格式
        
        Args:
            sample: 原始样本
            
        Returns:
            格式化后的样本
        """
        formatted = {
            'type': sample.get('type', 'qa'),
        }
        
        if formatted['type'] == 'qa':
            formatted['question'] = sample.get('question', '')
            formatted['answer'] = sample.get('answer', '')
        elif formatted['type'] == 'design':
            formatted['demand'] = sample.get('demand', '')
            formatted['scheme'] = sample.get('scheme', '')
        
        # 代码片段
        formatted['code_snippets'] = sample.get('code_snippets', [])
        
        # 推理过程
        formatted['reasoning_trace'] = sample.get('reasoning_trace', '')
        
        # 元数据
        formatted['metadata'] = sample.get('metadata', {})
        
        return formatted
    
    def load_jsonl(self, input_path: str) -> List[Dict[str, Any]]:
        """
        从JSONL文件加载数据
        
        Args:
            input_path: 输入文件路径
            
        Returns:
            样本列表
        """
        samples = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        sample = json.loads(line)
                        samples.append(sample)
                    except json.JSONDecodeError:
                        continue
        return samples
    
    def merge_datasets(self, file_paths: List[str], output_path: str) -> str:
        """
        合并多个数据集文件
        
        Args:
            file_paths: 输入文件路径列表
            output_path: 输出文件路径
            
        Returns:
            合并后的文件路径
        """
        all_samples = []
        for file_path in file_paths:
            if file_path.endswith('.jsonl'):
                samples = self.load_jsonl(file_path)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    samples = json.load(f)
            all_samples.extend(samples)
        
        return self.save_jsonl(all_samples, output_path)
    
    def split_dataset(self, input_path: str, train_ratio: float = 0.8, 
                     output_dir: str = 'output') -> Dict[str, str]:
        """
        分割数据集为训练集和验证集
        
        Args:
            input_path: 输入文件路径
            train_ratio: 训练集比例
            output_dir: 输出目录
            
        Returns:
            生成的文件路径字典
        """
        samples = self.load_jsonl(input_path)
        
        # 随机打乱
        import random
        random.shuffle(samples)
        
        # 分割
        split_idx = int(len(samples) * train_ratio)
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]
        
        # 保存
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_path = output_path / 'train.jsonl'
        val_path = output_path / 'val.jsonl'
        
        self.save_jsonl(train_samples, str(train_path))
        self.save_jsonl(val_samples, str(val_path))
        
        return {
            'train': str(train_path),
            'val': str(val_path)
        }
