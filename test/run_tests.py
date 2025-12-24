#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""运行所有测试的便捷脚本"""

import sys
import subprocess
from pathlib import Path

def main():
    """运行所有测试"""
    test_dir = Path(__file__).parent
    
    print("=" * 60)
    print("运行模型评估测试套件")
    print("=" * 60)
    
    # 运行pytest
    cmd = [
        sys.executable, "-m", "pytest",
        str(test_dir),
        "-v",
        "--tb=short"
    ]
    
    print(f"\n执行命令: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("所有测试通过！")
        print("=" * 60)
        print("\n注意：")
        print("- 如果看到 'SKIPPED' 测试，这是正常的")
        print("- 跳过的测试通常是因为模型文件不存在（已通过.gitignore排除）")
        print("- 这些测试会在有模型文件时自动运行")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("部分测试失败，请查看上面的输出")
        print("=" * 60)
        sys.exit(1)

if __name__ == '__main__':
    main()
