# pyro-project

SGLD (Stochastic Gradient Langevin Dynamics) を用いた学習係数 (Learning Coefficient / LLC) の数値推定実験。

- **DLN**: Deep Linear Network に対して Aoyagi (2024) の理論値と比較
- **LSA**: Self-Linear Attention モデルに対して、2つの DLN 理論値（d×d 部分 vs (d+1)×(d+1) 全体）と SGLD 推定値を比較

## ディレクトリ構成

```
pyro/
├── figures/                                # 出力画像
├── archive/                                # 旧版・バックアップ
├── archive2/                               # 旧版・バックアップ (第2世代)
│
│  # --- 共通モジュール ---
├── lsa_common.py                           # LSA 実験共通 (モデル, SGLD, 理論値, ランナー, プロット)
├── lsa_alpha.py                            # LSA α制御実験モジュール (config生成, 実験実行, プロット)
├── lsa_simple.py                           # LSA シンプル実験モジュール (α制御なし, λ_full のみ)
├── full_attention.py                       # Full Attention モデル & 訓練 (Full Attention → LSA 変換)
│
│  # --- DLN 実験 ---
├── sgld_hello_logplot.py                   # DLN log-log プロット実験
│
│  # --- LSA 実験 ---
├── sgld_hello_lsa_logplot.py               # LSA α制御版実験 (ボトルネック制約なし)
├── sgld_hello_lsa_logplot_bn.py            # LSA α制御版実験 (ボトルネック制約あり)
├── sgld_hello_lsa_logplot_simple.py        # LSA シンプル版実験 (α制御なし)
│
│  # --- Full Attention 実験 ---
├── sgld_full_attention_multi_dl_seeds.py   # Full Attention 軌跡 + 複数d_l×seed比較
│
│  # --- その他 ---
├── analyze.py                              # 理論解析 (λ の境界・位相解析プロット)
├── main.py                                 # エントリポイント
└── pyproject.toml                          # プロジェクト設定
```

## LSA 実験の理論値比較

各実験点について、SGLD で推定した LLC を以下の2つの理論値と比較する:

| 理論値 | 計算方法 | 意味 |
|---|---|---|
| `λ_matrix` | `compute_llc_theoretical_dln([d, d_l, d], rank_M)` | B の d×d 部分 (M) を DLN とみなした RLCT |
| `λ_full` | `compute_llc_theoretical_dln([d+1, d_l, d+1], rank_B)` | B の (d+1)×(d+1) 全体を DLN とみなした RLCT |

プロットは α (= rank(B) - rank(M)) ごとにグループ化し、各行 2列で表示。

## セットアップ

```bash
uv sync
```

## 実行例

```bash
# DLN 実験
CUDA_VISIBLE_DEVICES=1 uv run python -u sgld_hello_logplot.py

# LSA ボトルネック制約あり実験
CUDA_VISIBLE_DEVICES=1 uv run python -u sgld_hello_lsa_logplot_bn.py

# LSA ボトルネック制約なし実験
CUDA_VISIBLE_DEVICES=1 uv run python -u sgld_hello_lsa_logplot.py

# LSA シンプル版実験
CUDA_VISIBLE_DEVICES=1 uv run python -u sgld_hello_lsa_logplot_simple.py

# Full Attention 複数 d_l × seed 比較実験
CUDA_VISIBLE_DEVICES=1 uv run python -u sgld_full_attention_multi_dl_seeds.py

# 理論解析プロット
uv run python analyze.py
```

## 依存関係

- Python 3.11
- PyTorch 2.7+ (CUDA 11.8)
- Pyro-PPL 1.9+
- matplotlib, numpy
