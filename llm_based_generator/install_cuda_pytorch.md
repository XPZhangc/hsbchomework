# 安装CUDA版本PyTorch指南

## 问题
Python 3.13 目前没有PyTorch CUDA版本的预编译包。

## 解决方案

### 方案1：安装Python 3.11或3.12（推荐）

1. **下载Python 3.11或3.12**
   - 访问：https://www.python.org/downloads/
   - 下载Python 3.11.x 或 3.12.x（Windows安装程序）

2. **安装时选择"Add Python to PATH"**

3. **验证安装**
   ```bash
   python3.11 --version
   # 或
   python3.12 --version
   ```

4. **使用镜像源安装CUDA版本PyTorch**
   ```bash
   # 对于Python 3.11
   python3.11 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -i https://pypi.tuna.tsinghua.edu.cn/simple
   
   # 对于Python 3.12
   python3.12 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

5. **验证CUDA支持**
   ```bash
   python3.11 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

### 方案2：使用conda（如果已安装）

```bash
# 创建新环境
conda create -n llm_gen python=3.11 -y
conda activate llm_gen

# 安装CUDA版本PyTorch（使用conda-forge镜像）
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge
```

### 方案3：使用当前Python 3.13（CPU版本）

如果暂时无法切换Python版本，可以：
1. 使用CPU模式运行（较慢）
2. 或者等待PyTorch发布Python 3.13的CUDA版本

## 推荐配置

- **Python版本**: 3.11 或 3.12
- **CUDA版本**: 12.1（兼容CUDA 12.6）
- **PyTorch版本**: 2.9.1+cu121

## 安装后测试

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```








