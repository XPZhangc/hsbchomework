"""
整理output目录的文件
按照以下规则分类：
1. 规则生成的数据集 -> rule_datasets/
2. LLM生成的数据集 -> llm_datasets/ (保持原位置)
3. 训练集 -> training_datasets/ (保持原位置)
4. 测试集/验证集 -> validation_datasets/
5. 训练好的模型 -> models/
6. 日志和图片 -> logs_and_plots/
7. 删除冗余文件
"""
import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

def get_file_size_mb(file_path: Path) -> float:
    """获取文件大小（MB）"""
    try:
        return file_path.stat().st_size / (1024 * 1024)
    except:
        return 0.0

def move_files(file_mappings: List[Tuple[Path, Path]], dry_run: bool = False):
    """移动文件列表"""
    moved_count = 0
    total_size = 0.0
    
    for src, dst in file_mappings:
        if not src.exists():
            continue
        
        # 创建目标目录
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        # 如果目标文件已存在，添加序号
        if dst.exists():
            base_name = dst.stem
            extension = dst.suffix
            counter = 1
            while dst.exists():
                dst = dst.parent / f"{base_name}_{counter}{extension}"
                counter += 1
        
        if dry_run:
            size_mb = get_file_size_mb(src)
            print(f"  [DRY RUN] {src} -> {dst} ({size_mb:.2f} MB)")
            total_size += size_mb
        else:
            try:
                shutil.move(str(src), str(dst))
                size_mb = get_file_size_mb(dst)
                print(f"  [OK] {src.name} -> {dst}")
                total_size += size_mb
                moved_count += 1
            except Exception as e:
                print(f"  [ERROR] 移动失败 {src.name}: {str(e)}")
    
    return moved_count, total_size

def delete_files(file_list: List[Path], dry_run: bool = False):
    """删除文件列表"""
    deleted_count = 0
    total_size = 0.0
    
    for file_path in file_list:
        if not file_path.exists():
            continue
        
        size_mb = get_file_size_mb(file_path)
        
        if dry_run:
            print(f"  [DRY RUN] 删除: {file_path} ({size_mb:.2f} MB)")
            total_size += size_mb
        else:
            try:
                file_path.unlink()
                print(f"  [OK] 删除: {file_path.name}")
                total_size += size_mb
                deleted_count += 1
            except Exception as e:
                print(f"  [ERROR] 删除失败 {file_path.name}: {str(e)}")
    
    return deleted_count, total_size

def delete_dirs(dir_list: List[Path], dry_run: bool = False):
    """删除目录列表"""
    deleted_count = 0
    total_size = 0.0
    
    for dir_path in dir_list:
        if not dir_path.exists():
            continue
        
        # 计算目录大小
        size_mb = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file()) / (1024 * 1024)
        
        if dry_run:
            print(f"  [DRY RUN] 删除目录: {dir_path} ({size_mb:.2f} MB)")
            total_size += size_mb
        else:
            try:
                shutil.rmtree(dir_path)
                print(f"  [OK] 删除目录: {dir_path.name}")
                total_size += size_mb
                deleted_count += 1
            except Exception as e:
                print(f"  [ERROR] 删除失败 {dir_path.name}: {str(e)}")
    
    return deleted_count, total_size

def organize_output(output_dir: Path, dry_run: bool = False):
    """整理output目录"""
    print("="*60)
    print("整理output目录")
    print("="*60)
    print(f"输出目录: {output_dir}")
    print(f"模式: {'预览模式（不实际执行）' if dry_run else '执行模式'}")
    print("="*60)
    
    if not output_dir.exists():
        print(f"错误: 输出目录不存在: {output_dir}")
        sys.exit(1)
    
    # 定义目标目录
    rule_datasets_dir = output_dir / "rule_datasets"
    llm_datasets_dir = output_dir / "llm_datasets"  # 保持原位置
    training_datasets_dir = output_dir / "training_datasets"  # 保持原位置
    validation_datasets_dir = output_dir / "validation_datasets"
    models_dir = output_dir / "models"
    logs_and_plots_dir = output_dir / "logs_and_plots"
    
    # 1. 规则生成的数据集 -> rule_datasets/
    print("\n[1] 整理规则生成的数据集...")
    rule_datasets_mappings = []
    datasets_dir = output_dir / "datasets"
    if datasets_dir.exists():
        # 移动所有JSONL文件（规则相关的数据集）
        for jsonl_file in datasets_dir.glob("*.jsonl"):
            dst = rule_datasets_dir / jsonl_file.name
            rule_datasets_mappings.append((jsonl_file, dst))
        
        # 移动parquet文件（如果有通用数据集）
        for parquet_file in datasets_dir.glob("*.parquet"):
            dst = rule_datasets_dir / parquet_file.name
            rule_datasets_mappings.append((parquet_file, dst))
    
    # 测试集也放到rule_datasets（因为是基于规则生成的）
    test_set_file = output_dir / "test_set.jsonl"
    if test_set_file.exists():
        rule_datasets_mappings.append((test_set_file, rule_datasets_dir / "test_set.jsonl"))
    
    moved_count, size = move_files(rule_datasets_mappings, dry_run)
    print(f"  移动了 {moved_count} 个文件 ({size:.2f} MB)")
    
    # 2. LLM生成的数据集 -> llm_datasets/ (已经在正确位置，只需确认)
    print("\n[2] 检查LLM生成的数据集...")
    llm_dir = output_dir / "llm_datasets"
    if llm_dir.exists():
        llm_files = list(llm_dir.glob("*.jsonl"))
        print(f"  LLM数据集已在正确位置: {llm_dir}")
        print(f"  包含 {len(llm_files)} 个文件")
    else:
        print("  LLM数据集目录不存在")
    
    # 3. 训练集 -> training_datasets/ (只保留切分后的训练集)
    print("\n[3] 整理训练集...")
    training_mappings = []
    training_dir = output_dir / "training_datasets"
    
    # 从split_datasets移动训练集文件到training_datasets
    split_datasets_dir = output_dir / "split_datasets"
    if split_datasets_dir.exists():
        train_files = list(split_datasets_dir.glob("*_train.jsonl"))
        for train_file in train_files:
            dst = training_datasets_dir / train_file.name
            training_mappings.append((train_file, dst))
    
    # 从validation_datasets移动训练集文件到training_datasets（如果存在）
    validation_dir = output_dir / "validation_datasets"
    if validation_dir.exists():
        train_files = list(validation_dir.glob("*_train.jsonl"))
        for train_file in train_files:
            dst = training_datasets_dir / train_file.name
            training_mappings.append((train_file, dst))
    
    moved_count, size = move_files(training_mappings, dry_run)
    print(f"  移动了 {moved_count} 个训练集文件 ({size:.2f} MB)")
    
    # 删除training_datasets中的原始合并数据集文件（不是切分后的）
    if training_dir.exists():
        original_files_to_delete = []
        for file_path in training_dir.glob("*.jsonl"):
            # 如果不是切分后的训练集文件（不包含_train），则删除
            if "_train" not in file_path.stem:
                original_files_to_delete.append(file_path)
        
        if original_files_to_delete:
            print(f"  删除原始合并数据集文件: {len(original_files_to_delete)} 个")
            deleted_count, deleted_size = delete_files(original_files_to_delete, dry_run)
            print(f"  删除了 {deleted_count} 个文件 ({deleted_size:.2f} MB)")
    
    # 4. 验证集 -> validation_datasets/ (只保留切分后的验证集)
    print("\n[4] 整理验证集...")
    validation_mappings = []
    
    # 从split_datasets移动验证集文件到validation_datasets
    if split_datasets_dir.exists():
        val_files = list(split_datasets_dir.glob("*_val.jsonl"))
        for val_file in val_files:
            dst = validation_datasets_dir / val_file.name
            validation_mappings.append((val_file, dst))
    
    moved_count, size = move_files(validation_mappings, dry_run)
    print(f"  移动了 {moved_count} 个验证集文件 ({size:.2f} MB)")
    
    # 删除validation_datasets中的训练集文件（如果存在）
    if validation_dir.exists():
        train_files_to_delete = list(validation_dir.glob("*_train.jsonl"))
        if train_files_to_delete:
            print(f"  删除validation_datasets中的训练集文件: {len(train_files_to_delete)} 个")
            deleted_count, deleted_size = delete_files(train_files_to_delete, dry_run)
            print(f"  删除了 {deleted_count} 个文件 ({deleted_size:.2f} MB)")
    
    # 5. 训练好的模型 -> models/
    print("\n[5] 整理训练好的模型...")
    model_mappings = []
    checkpoints_dir = output_dir / "llamafactory_checkpoints"
    if checkpoints_dir.exists():
        # 移动整个checkpoints目录到models
        for checkpoint_subdir in checkpoints_dir.iterdir():
            if checkpoint_subdir.is_dir():
                dst = models_dir / checkpoint_subdir.name
                model_mappings.append((checkpoint_subdir, dst))
    
    # 也检查是否有其他checkpoints目录
    other_checkpoints = output_dir / "checkpoints"
    if other_checkpoints.exists():
        for checkpoint_subdir in other_checkpoints.iterdir():
            if checkpoint_subdir.is_dir():
                dst = models_dir / checkpoint_subdir.name
                model_mappings.append((checkpoint_subdir, dst))
    
    moved_count, size = move_files(model_mappings, dry_run)
    print(f"  移动了 {moved_count} 个目录 ({size:.2f} MB)")
    
    # 6. 日志和图片 -> logs_and_plots/
    print("\n[6] 整理日志和图片...")
    log_mappings = []
    
    # 从模型目录中提取日志文件（trainer_log.jsonl, trainer_state.json等）
    for checkpoint_dir in [output_dir / "llamafactory_checkpoints", output_dir / "checkpoints"]:
        if checkpoint_dir.exists():
            for subdir in checkpoint_dir.iterdir():
                if subdir.is_dir():
                    # 查找日志文件
                    for log_file in subdir.glob("trainer_*.json*"):
                        # 创建带数据集名称的文件名
                        new_name = f"{subdir.name}_{log_file.name}"
                        dst = logs_and_plots_dir / new_name
                        log_mappings.append((log_file, dst))
                    
                    # 查找loss曲线图片
                    for img_file in subdir.glob("loss_curve.*"):
                        new_name = f"{subdir.name}_{img_file.name}"
                        dst = logs_and_plots_dir / new_name
                        log_mappings.append((img_file, dst))
    
    # 查找output根目录下的图片文件
    image_extensions = ['.png', '.jpg', '.jpeg', '.svg', '.pdf']
    for ext in image_extensions:
        for img_file in output_dir.glob(f"*{ext}"):
            if img_file.is_file():
                dst = logs_and_plots_dir / img_file.name
                log_mappings.append((img_file, dst))
    
    moved_count, size = move_files(log_mappings, dry_run)
    print(f"  移动了 {moved_count} 个文件 ({size:.2f} MB)")
    
    # 7. 删除冗余文件和目录
    print("\n[7] 删除冗余文件...")
    files_to_delete = []
    dirs_to_delete = []
    
    # 删除空的datasets目录（如果已移动所有文件）
    if datasets_dir.exists():
        remaining_files = [f for f in datasets_dir.iterdir() if f.is_file()]
        remaining_dirs = [f for f in datasets_dir.iterdir() if f.is_dir()]
        
        if len(remaining_files) == 0 and len(remaining_dirs) == 0:
            dirs_to_delete.append(datasets_dir)
        else:
            if remaining_files:
                print(f"  警告: datasets目录仍有 {len(remaining_files)} 个文件未移动: {[f.name for f in remaining_files]}")
            if remaining_dirs:
                print(f"  警告: datasets目录仍有 {len(remaining_dirs)} 个子目录未移动: {[d.name for d in remaining_dirs]}")
    
    # 删除空的split_datasets目录（如果已移动所有文件）
    if split_datasets_dir.exists():
        remaining_files = list(split_datasets_dir.glob("*"))
        if len(remaining_files) == 0:
            dirs_to_delete.append(split_datasets_dir)
        else:
            print(f"  警告: split_datasets目录仍有 {len(remaining_files)} 个文件未移动")
    
    # 删除冗余的旧文件
    redundant_files = [
        output_dir / "training_data.jsonl",  # 旧的训练数据文件
    ]
    for file_path in redundant_files:
        if file_path.exists():
            files_to_delete.append(file_path)
    
    # 删除LLaMA-Factory中间文件目录
    llamafactory_datasets_dir = output_dir / "llamafactory_datasets"
    if llamafactory_datasets_dir.exists():
        # 移动dataset_info.json到logs_and_plots
        info_file = llamafactory_datasets_dir / "dataset_info.json"
        if info_file.exists():
            dst = logs_and_plots_dir / "dataset_info.json"
            log_mappings.append((info_file, dst))
        
        # 移动JSON格式的数据集文件到logs_and_plots（作为备份）
        for json_file in llamafactory_datasets_dir.glob("*.json"):
            if json_file.name != "dataset_info.json":
                dst = logs_and_plots_dir / f"llamafactory_{json_file.name}"
                log_mappings.append((json_file, dst))
        
        # 检查是否还有其他文件
        remaining_files = [f for f in llamafactory_datasets_dir.glob("*") 
                          if f.name not in ["dataset_info.json"] and not f.name.endswith('.json')]
        if len(remaining_files) == 0:
            dirs_to_delete.append(llamafactory_datasets_dir)
    
    llamafactory_configs_dir = output_dir / "llamafactory_configs"
    if llamafactory_configs_dir.exists():
        # 配置文件移动到logs_and_plots
        for config_file in llamafactory_configs_dir.glob("*.yaml"):
            dst = logs_and_plots_dir / config_file.name
            log_mappings.append((config_file, dst))
        
        remaining_files = list(llamafactory_configs_dir.glob("*"))
        if len(remaining_files) == 0:
            dirs_to_delete.append(llamafactory_configs_dir)
    
    # 先移动配置文件
    if log_mappings:
        moved_count, size = move_files(log_mappings, dry_run)
        print(f"  移动了 {moved_count} 个配置文件 ({size:.2f} MB)")
    
    deleted_files, deleted_size = delete_files(files_to_delete, dry_run)
    deleted_dirs, deleted_dirs_size = delete_dirs(dirs_to_delete, dry_run)
    
    print(f"  删除了 {deleted_files} 个文件 ({deleted_size:.2f} MB)")
    print(f"  删除了 {deleted_dirs} 个目录 ({deleted_dirs_size:.2f} MB)")
    
    # 总结
    print("\n" + "="*60)
    print("整理完成！")
    print("="*60)
    print("\n目录结构:")
    print(f"  rule_datasets/        - 规则生成的数据集")
    print(f"  llm_datasets/         - LLM生成的数据集")
    print(f"  training_datasets/   - 切分后的训练集 (*_train.jsonl)")
    print(f"  validation_datasets/ - 切分后的验证集 (*_val.jsonl)")
    print(f"  models/               - 训练好的模型")
    print(f"  logs_and_plots/       - 日志和图片")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description='整理output目录的文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 预览模式（不实际执行）
  python organize_output.py --dry-run
  
  # 执行整理
  python organize_output.py
  
  # 指定output目录
  python organize_output.py --output-dir ./output
        """
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='output目录路径（默认: ./output）'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='预览模式，不实际执行文件移动和删除'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    organize_output(output_dir, dry_run=args.dry_run)

if __name__ == '__main__':
    main()
