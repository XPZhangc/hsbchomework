#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试数据加载功能"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from evaluate_models import load_jsonl_dataset, convert_to_sharegpt_format


class TestDataLoading:
    """测试数据加载功能"""
    
    @pytest.fixture
    def sample_dataset_path(self):
        """返回示例数据集路径"""
        return Path(__file__).parent / "test_data" / "sample_dataset.jsonl"
    
    def test_load_jsonl_dataset(self, sample_dataset_path):
        """测试JSONL数据集加载"""
        if not sample_dataset_path.exists():
            pytest.skip("测试数据文件不存在")
        
        data = load_jsonl_dataset(sample_dataset_path)
        assert len(data) > 0, "应该加载到数据"
        assert isinstance(data, list), "返回应该是列表"
        assert all(isinstance(item, dict) for item in data), "每个项目应该是字典"
    
    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        nonexistent_path = Path(__file__).parent / "test_data" / "nonexistent.jsonl"
        with pytest.raises(FileNotFoundError):
            load_jsonl_dataset(nonexistent_path)
    
    def test_convert_to_sharegpt_format(self, sample_dataset_path):
        """测试转换为ShareGPT格式"""
        if not sample_dataset_path.exists():
            pytest.skip("测试数据文件不存在")
        
        data = load_jsonl_dataset(sample_dataset_path)
        sharegpt_data = convert_to_sharegpt_format(data)
        
        assert len(sharegpt_data) > 0, "应该转换出数据"
        assert all('conversation' in item for item in sharegpt_data), "每个项目应该有conversation字段"
        assert all(isinstance(item['conversation'], list) for item in sharegpt_data), "conversation应该是列表"
    
    def test_sharegpt_format_structure(self, sample_dataset_path):
        """测试ShareGPT格式结构"""
        if not sample_dataset_path.exists():
            pytest.skip("测试数据文件不存在")
        
        data = load_jsonl_dataset(sample_dataset_path)
        sharegpt_data = convert_to_sharegpt_format(data)
        
        if len(sharegpt_data) > 0:
            item = sharegpt_data[0]
            assert 'conversation' in item, "应该有conversation字段"
            assert len(item['conversation']) >= 2, "conversation应该至少包含user和assistant消息"
            
            roles = [msg['role'] for msg in item['conversation']]
            assert 'user' in roles, "应该有user角色"
            assert 'assistant' in roles, "应该有assistant角色"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
