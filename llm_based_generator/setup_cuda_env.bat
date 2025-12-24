@echo off
REM 安装CUDA版本PyTorch的脚本
REM 需要先安装Python 3.11或3.12

echo ========================================
echo 安装CUDA版本PyTorch
echo ========================================
echo.

REM 检查Python版本
python --version
echo.

REM 检查是否有CUDA
nvidia-smi
echo.

echo 正在使用镜像源安装CUDA版本的PyTorch...
echo 使用CUDA 12.1版本（兼容CUDA 12.6）
echo.

REM 卸载CPU版本（如果存在）
python -m pip uninstall torch torchvision torchaudio -y

REM 安装CUDA版本（使用镜像源加速）
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -i https://pypi.tuna.tsinghua.edu.cn/simple

echo.
echo ========================================
echo 验证安装
echo ========================================
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available()); print('CUDA版本:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU名称:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo.
echo 安装完成！
pause








