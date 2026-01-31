# pyro-project

SGLD (Stochastic Gradient Langevin Dynamics) を用いた学習係数 (Learning Coefficient / LLC) の数値推定実験。
Deep Linear Network (DLN) および Self-Linear Attention (SLA) モデルに対して、Aoyagi (2024) の理論値との比較を行う。

## ディレクトリ構成

```
pyro/
├── figures/           # 出力画像
├── archive/           # 旧版・バックアップ
├── analyze.py         # 理論解析 (λ の境界・位相解析プロット)
├── sgld_hello.py      # DLN 基盤実装 (SGLD + Aoyagi Theorem 1)
├── sgld_hello_s.py    # SLA 基盤実装 (Self-Linear Attention)
├── sgld_hello_logplot.py              # DLN log-log プロット (layer-wise rank制限)
├── sgld_hello_s_logplot.py            # SLA log-log プロット (Frobenius norm)
├── sgld_hello_lsa_logplot.py          # LSA α制御版実験
├── sgld_hello_lsa_logplot_bn.py       # LSA Bottleneck (global rank制限) 版
├── sgld_hello_lsa_logplot_layerwise.py # LSA layer-wise版 (保存用)
├── sgld.py            # SGLD オプティマイザ基礎実装
├── hello.py           # Pyro/HMC チュートリアル (w^4)
└── hello2.py          # HMC チュートリアル (合成回帰データ)
```

## セットアップ

```bash
uv sync
```

## 実行例

```bash
# DLN 実験
CUDA_VISIBLE_DEVICES=1 uv run python -u sgld_hello_logplot.py

# LSA Bottleneck 版実験
CUDA_VISIBLE_DEVICES=1 uv run python -u sgld_hello_lsa_logplot_bn.py

# 理論解析プロット
uv run python analyze.py
```

## 依存関係

- Python 3.11
- PyTorch 2.7+ (CUDA 11.8)
- Pyro-PPL 1.9+
- matplotlib, numpy
