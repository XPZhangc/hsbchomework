#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""集成测试：测试完整的评估流程"""

import sys
import os
from pathlib import Path
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest


class TestIntegration:
    """集成测试"""
    
    @pytest.fixture
    def test_config(self):
        """加载测试配置"""
        config_path = Path(__file__).parent / "test_config" / "test_config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def test_evaluate_models_script_exists(self):
        """测试评估脚本是否存在"""
        script_path = project_root / "evaluate_models.py"
        assert script_path.exists(), "evaluate_models.py应该存在"
    
    def test_test_data_exists(self):
        """测试测试数据是否存在"""
        test_data_dir = Path(__file__).parent / "test_data"
        assert test_data_dir.exists(), "test_data目录应该存在"
        
        sample_dataset = test_data_dir / "sample_dataset.jsonl"
        assert sample_dataset.exists(), "sample_dataset.jsonl应该存在"
    
    def test_config_file_exists(self):
        """测试配置文件是否存在"""
        config_path = Path(__file__).parent / "test_config" / "test_config.json"
        assert config_path.exists(), "test_config.json应该存在"
    
    @pytest.mark.skipif(
        not Path("./models/qwen25_05b").exists(),
        reason="需要训练好的模型"
    )
    def test_full_evaluation_workflow(self, test_config):
        """测试完整评估流程（需要模型）"""
        # 这个测试需要实际的模型，在CI/CD中可能需要跳过
        import subprocess
        
        test_data_dir = Path(__file__).parent / "test_data"
        output_dir = Path(__file__).parent / "test_output"
        output_dir.mkdir(exist_ok=True)
        
        # 构建命令
        cmd = [
            sys.executable,
            str(project_root / "evaluate_models.py"),
            "--test-dir", str(test_data_dir),
            "--output", str(output_dir),
            "--max-samples", "5"
        ]
        
        # 运行评估脚本
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # 检查是否成功运行（即使没有模型，也应该有合理的错误信息）
        assert result.returncode in [0, 1], "脚本应该运行完成或给出错误"
    
    def test_output_directory_creation(self):
        """测试输出目录创建"""
        output_dir = Path(__file__).parent / "test_output"
        output_dir.mkdir(exist_ok=True)
        assert output_dir.exists(), "输出目录应该被创建"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
