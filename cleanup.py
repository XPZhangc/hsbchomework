"""
清理项目中的临时文件和不需要的文件
"""
import shutil
from pathlib import Path

def cleanup():
    """清理不需要的文件"""
    base_dir = Path(__file__).parent
    
    # 要删除的文件列表
    files_to_delete = [
        "convert_to_llamafactory_format.py",  # 功能已集成到train_with_llamafactory.py
    ]
    
    # 要清理的目录（保留结构，只删除内容）
    dirs_to_clean = [
        # 可以清理的中间文件目录（可选）
        # "output/llamafactory_configs",  # 配置文件，保留
        # "output/llamafactory_datasets",  # 转换后的数据集，保留
    ]
    
    print("="*60)
    print("清理项目文件")
    print("="*60)
    
    # 删除文件
    deleted_files = []
    for file_path in files_to_delete:
        full_path = base_dir / file_path
        if full_path.exists():
            try:
                full_path.unlink()
                deleted_files.append(file_path)
                print(f"✓ 已删除: {file_path}")
            except Exception as e:
                print(f"✗ 删除失败 {file_path}: {str(e)}")
        else:
            print(f"- 文件不存在: {file_path}")
    
    # 清理目录内容（如果需要）
    for dir_path in dirs_to_clean:
        full_path = base_dir / dir_path
        if full_path.exists():
            try:
                shutil.rmtree(full_path)
                full_path.mkdir(parents=True, exist_ok=True)
                print(f"✓ 已清理目录: {dir_path}")
            except Exception as e:
                print(f"✗ 清理目录失败 {dir_path}: {str(e)}")
    
    print("\n" + "="*60)
    print(f"清理完成！删除了 {len(deleted_files)} 个文件")
    print("="*60)

if __name__ == '__main__':
    cleanup()
