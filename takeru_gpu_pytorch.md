# GPU対応PyTorch環境構築の知見

## 環境情報

### ハードウェア
- **GPU**: NVIDIA A100 80GB PCIe × 2台
- **CUDA Version**: 11.6 (nvidia-smi出力)
- **ドライバーバージョン**: 510.47.03

### ソフトウェア
- **OS**: Linux 5.4.0-164-generic
- **Python**: 3.11.13 (CPython)
- **uv**: 0.8.2
- **PyTorch**: 2.7.1+cu118

## 環境構築手順

### 1. GPU情報の確認

```bash
nvidia-smi
```

CUDA バージョンとGPUの種類・メモリ容量を確認する。

### 2. uv仮想環境の作成

```bash
uv venv
```

`.venv` ディレクトリに仮想環境が作成される。

### 3. PyTorchのインストール

CUDA 11.6が利用可能だったが、PyTorchの公式サポートを考慮してCUDA 11.8版を使用:

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**インストールされたパッケージ**:
- torch==2.7.1+cu118
- torchvision==0.22.1+cu118
- torchaudio==2.7.1+cu118
- triton==3.3.1
- 各種NVIDIA CUDA関連パッケージ (cublas, cudnn, nccl等)

### 4. 動作確認

```bash
source .venv/bin/activate
python test_gpu.py
```

## GPU動作確認結果

### 認識されたGPU情報
- GPU 0: NVIDIA A100 80GB PCIe (79.17 GB)
- GPU 1: NVIDIA A100 80GB PCIe (79.17 GB)

### PyTorch CUDA情報
- `torch.cuda.is_available()`: True
- `torch.version.cuda`: 11.8
- `torch.cuda.device_count()`: 2

### テンソルのGPU配置
```python
x = torch.rand(5, 3).cuda()  # cuda:0に配置される
```

## 重要なポイント

### CUDAバージョンの選択
- システムのCUDAバージョン (11.6) より新しいCUDA 11.8のPyTorchをインストール
- PyTorchのCUDAライブラリは独立しているため、システムのCUDAより新しくても問題なく動作
- 後方互換性があるため、CUDA 11.8対応版はCUDA 11.6環境でも動作する

### uvの利点
- 従来のvenv + pipよりも高速
- パッケージ解決が効率的
- インストール時間が大幅に短縮 (今回は約136ms)

### マルチGPU環境
- デフォルトでは `cuda:0` が使用される
- 特定のGPUを使用する場合:
  ```python
  device = torch.device("cuda:1")  # GPU 1を指定
  x = torch.rand(5, 3).to(device)
  ```
- 環境変数で可視GPUを制限:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python script.py  # GPU 0のみ使用
  ```

## トラブルシューティング

### spackのエラーについて
出力に表示される以下のエラーは無害:
```
AttributeError: module 'collections' has no attribute 'MutableMapping'
```
これはシステムにインストールされているspackの古いバージョンとPython 3.11の非互換性によるもので、uv環境やPyTorchの動作には影響しない。

## テストスクリプト (test_gpu.py)

```python
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Number of GPUs:", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")

    # Test tensor on GPU
    x = torch.rand(5, 3).cuda()
    print("\nTest tensor on GPU:")
    print(x)
    print(f"Tensor device: {x.device}")
else:
    print("\nWarning: CUDA is not available!")
```

## 参考情報

- PyTorch公式: https://pytorch.org/
- CUDA対応版のインストール: https://pytorch.org/get-started/locally/
- uvドキュメント: https://github.com/astral-sh/uv
