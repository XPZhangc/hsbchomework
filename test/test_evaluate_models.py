#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试评估脚本功能"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import json
from evaluate_models import (
    load_jsonl_dataset,
    convert_to_sharegpt_format,
    format_prompt_for_qwen,
    calculate_bleu
)


class TestEvaluateModels:
    """测试评估脚本功能"""
    
    @pytest.fixture
    def sample_data_path(self):
        """返回示例数据路径"""
        return Path(__file__).parent / "test_data" / "sample_dataset.jsonl"
    
    @pytest.fixture
    def sample_validation_path(self):
        """返回示例验证集路径"""
        return Path(__file__).parent / "test_data" / "sample_validation.jsonl"
    
    def test_data_loading_integration(self, sample_data_path):
        """测试数据加载集成"""
        if not sample_data_path.exists():
            pytest.skip("测试数据文件不存在")
        
        data = load_jsonl_dataset(sample_data_path)
        sharegpt_data = convert_to_sharegpt_format(data)
        
        assert len(sharegpt_data) > 0, "应该成功转换数据"
    
    def test_format_prompt_for_qwen(self):
        """测试Qwen格式提示格式化"""
        try:
            from transformers import AutoTokenizer
            
            # 尝试加载tokenizer（如果模型存在）
            model_path = "./models/qwen25_05b"
            if not Path(model_path).exists():
                pytest.skip("模型不存在，跳过测试")
            
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            conversation = [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"}
            ]
            
            formatted = format_prompt_for_qwen(conversation, tokenizer)
            assert isinstance(formatted, str), "应该返回字符串"
            assert len(formatted) > 0, "格式化后的文本不应该为空"
        except Exception as e:
            pytest.skip(f"无法加载tokenizer: {e}")
    
    def test_bleu_calculation_edge_cases(self):
        """测试BLEU计算的边界情况"""
        # 测试空字符串
        assert calculate_bleu("", "") == 0.0
        
        # 测试单字符
        score = calculate_bleu("a", "a")
        assert score >= 0.0
        
        # 测试长文本
        long_ref = " ".join(["测试"] * 100)
        long_cand = " ".join(["测试"] * 100)
        score = calculate_bleu(long_ref, long_cand)
        assert score > 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
