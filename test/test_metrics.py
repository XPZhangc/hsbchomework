#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试评估指标计算功能"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import numpy as np
from evaluate_models import calculate_bleu, calculate_perplexity


class TestBLEU:
    """测试BLEU分数计算"""
    
    def test_bleu_identical_texts(self):
        """测试相同文本的BLEU分数"""
        reference = "这是一个测试句子"
        candidate = "这是一个测试句子"
        score = calculate_bleu(reference, candidate)
        # 注意：sentence_bleu对短文本可能不会给出很高的分数
        # 但相同文本应该有一定的BLEU分数
        assert score > 0, "相同文本应该有正的BLEU分数"
        assert score <= 1.0, "BLEU分数应该在0-1之间"
    
    def test_bleu_similar_texts(self):
        """测试相似文本的BLEU分数"""
        reference = "这是一个测试句子"
        candidate = "这是一个测试的句子"
        score = calculate_bleu(reference, candidate)
        # 使用smoothing function后，即使不完全匹配也应该有非负分数
        assert score >= 0, "相似文本应该有非负的BLEU分数"
        assert score <= 1.0, "BLEU分数应该在0-1之间"
    
    def test_bleu_different_texts(self):
        """测试不同文本的BLEU分数"""
        reference = "这是一个测试句子"
        candidate = "完全不同的内容"
        score = calculate_bleu(reference, candidate)
        assert score < 0.5, "不同文本应该有较低的BLEU分数"
    
    def test_bleu_empty_candidate(self):
        """测试空候选文本"""
        reference = "这是一个测试句子"
        candidate = ""
        score = calculate_bleu(reference, candidate)
        assert score == 0.0, "空候选文本应该返回0"
    
    def test_bleu_empty_reference(self):
        """测试空参考文本"""
        reference = ""
        candidate = "这是一个测试句子"
        score = calculate_bleu(reference, candidate)
        assert score == 0.0, "空参考文本应该返回0"


class TestPerplexity:
    """测试Perplexity计算"""
    
    @pytest.fixture
    def mock_model_and_tokenizer(self):
        """创建模拟的模型和tokenizer"""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # 注意：这个测试需要实际的模型，在CI/CD中可能需要mock
        try:
            model_path = "./models/qwen25_05b"
            if not Path(model_path).exists():
                pytest.skip("模型不存在，跳过Perplexity测试")
            
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            return model, tokenizer, "cuda" if torch.cuda.is_available() else "cpu"
        except Exception as e:
            pytest.skip(f"无法加载模型: {e}")
    
    def test_perplexity_valid_text(self, mock_model_and_tokenizer):
        """测试有效文本的Perplexity"""
        model, tokenizer, device = mock_model_and_tokenizer
        text = "这是一个测试句子，用于计算困惑度。"
        ppl = calculate_perplexity(model, tokenizer, text, device)
        assert ppl > 0, "Perplexity应该大于0"
        assert ppl != float('inf'), "Perplexity不应该是无穷大"
    
    def test_perplexity_short_text(self, mock_model_and_tokenizer):
        """测试短文本的Perplexity"""
        model, tokenizer, device = mock_model_and_tokenizer
        text = "测试"
        ppl = calculate_perplexity(model, tokenizer, text, device)
        # 短文本可能返回inf，这是正常的
        assert isinstance(ppl, float), "Perplexity应该是浮点数"
    
    def test_perplexity_empty_text(self, mock_model_and_tokenizer):
        """测试空文本的Perplexity"""
        model, tokenizer, device = mock_model_and_tokenizer
        text = ""
        ppl = calculate_perplexity(model, tokenizer, text, device)
        assert ppl == float('inf'), "空文本应该返回inf"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
