# Full Attention モデル & 訓練モジュール
# Full Attention → LSA 実験用

from typing import Tuple, List
import torch
import torch.nn as nn
from tqdm import tqdm

from lsa_common import (
    SelfLinearAttention,
    generate_icl_batch,
    compute_llc_theoretical_dln
)


# ============================================================
# 1. Full Attention モデル
# ============================================================

class FullAttention(nn.Module):
    """
    Full Attention モデル for In-Context Learning

    モデル式:
        Output = (X @ W_Q @ W_K.T @ X.T) @ X @ W_V @ W_O.T

    次元の流れ:
        X:                  (batch, N+1, d+1)
        W_Q, W_K, W_V, W_O: (d+1, d_l)

        X @ W_Q:            (batch, N+1, d_l)
        W_K.T:              (d_l, d+1)
        X @ W_Q @ W_K.T:    (batch, N+1, d+1)
        (... ) @ X.T:       (batch, N+1, N+1)   # Attention行列
        Attn @ X:           (batch, N+1, d+1)
        (... ) @ W_V:       (batch, N+1, d_l)
        (... ) @ W_O.T:     (batch, N+1, d+1)

        予測値 = Output[:, -1, -1]  # query行の最終成分 (スカラー)
    """

    def __init__(self, input_dim: int, latent_dim: int, init_scale: float = 0.01):
        super().__init__()
        self.input_dim = input_dim      # d+1
        self.latent_dim = latent_dim    # d_l

        # 全て同じサイズ: (d+1, d_l)
        self.W_Q = nn.Parameter(torch.randn(input_dim, latent_dim) * init_scale)
        self.W_K = nn.Parameter(torch.randn(input_dim, latent_dim) * init_scale)
        self.W_V = nn.Parameter(torch.randn(input_dim, latent_dim) * init_scale)
        self.W_O = nn.Parameter(torch.randn(input_dim, latent_dim) * init_scale)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: (batch, N+1, d+1) プロンプト行列

        Returns:
            pred: (batch,) query行の最終成分（予測値）
        """
        # Attention行列: (batch, N+1, N+1)
        attn = X @ self.W_Q @ self.W_K.T @ X.transpose(-2, -1)

        # 出力: (batch, N+1, d+1)
        out = attn @ X @ self.W_V @ self.W_O.T

        # query行の最終成分を返す
        return out[:, -1, -1]  # (batch,)

    def get_B_matrix(self) -> torch.Tensor:
        """W_Q @ W_K.T を抽出"""
        return self.W_Q @ self.W_K.T

    def get_full_output(self, X: torch.Tensor) -> torch.Tensor:
        """
        フル出力を取得（デバッグ用）

        Returns:
            out: (batch, N+1, d+1)
        """
        attn = X @ self.W_Q @ self.W_K.T @ X.transpose(-2, -1)
        out = attn @ X @ self.W_V @ self.W_O.T
        return out


# ============================================================
# 2. 訓練関数
# ============================================================

def train_full_attention_with_checkpoints(
    d: int,
    d_l: int,
    N: int = 20,
    num_tasks: int = 10000,
    epochs: int = 100,
    batch_size: int = 500,
    lr: float = 1e-3,
    init_scale: float = 0.01,
    checkpoint_epochs: List[int] = None,
    device: str = 'cpu',
    verbose: bool = True
) -> Tuple[FullAttention, List[float], List[Tuple[int, dict]]]:
    """
    Full Attention モデルを訓練し、指定エポックでチェックポイントを保存

    Args:
        d: 入力次元
        d_l: 潜在次元
        N: context examples数
        num_tasks: 訓練タスク数
        epochs: エポック数
        batch_size: バッチサイズ
        lr: 学習率
        init_scale: 初期化スケール
        checkpoint_epochs: チェックポイントを保存するエポックのリスト
        device: デバイス
        verbose: 進捗表示

    Returns:
        model: 訓練済みモデル
        loss_history: 損失履歴
        checkpoints: [(epoch, state_dict), ...] のリスト
    """
    if checkpoint_epochs is None:
        checkpoint_epochs = list(range(0, epochs + 1, max(1, epochs // 10)))

    input_dim = d + 1
    model = FullAttention(input_dim, d_l, init_scale).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 訓練データ生成
    X_data, y_data = generate_icl_batch(num_tasks, d, N, device=device)
    y_data = y_data.squeeze(-1)  # (num_tasks,)

    loss_history = []
    checkpoints = []
    num_batches = (num_tasks + batch_size - 1) // batch_size

    # epoch=0 のチェックポイント（初期状態）
    if 0 in checkpoint_epochs:
        checkpoints.append((0, {k: v.clone() for k, v in model.state_dict().items()}))

    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Training Full Attention")

    for epoch in iterator:
        epoch_loss = 0.0
        indices = torch.randperm(num_tasks, device=device)

        for i in range(num_batches):
            batch_idx = indices[i * batch_size:(i + 1) * batch_size]
            X_batch = X_data[batch_idx]
            y_batch = y_data[batch_idx]

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = torch.mean((pred - y_batch) ** 2)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)

        # チェックポイント保存
        current_epoch = epoch + 1
        if current_epoch in checkpoint_epochs:
            checkpoints.append((current_epoch, {k: v.clone() for k, v in model.state_dict().items()}))

        if verbose and current_epoch % 10 == 0:
            tqdm.write(f"Epoch {current_epoch}/{epochs}, Loss: {avg_loss:.6f}")

    return model, loss_history, checkpoints


# ============================================================
# 3. B行列の抽出とランク解析
# ============================================================

def analyze_B_matrix(
    B: torch.Tensor,
    d: int,
    rtol: float = 1e-3,
    verbose: bool = True
) -> dict:
    """
    B = W_Q @ W_K.T のランク構造を解析

    Args:
        B: (d+1, d+1) の行列
        d: 入力次元（d+1 - 1）
        rtol: ランク判定の相対許容誤差 (torch.linalg.matrix_rank の rtol)
        verbose: 詳細表示

    Returns:
        dict: {
            'B': B行列,
            'M': d×d部分,
            'rank_B': Bのランク,
            'rank_M': Mのランク,
            'alpha': rank_B - rank_M,
            'singular_values_B': Bの特異値,
            'singular_values_M': Mの特異値
        }
    """
    with torch.no_grad():
        # M部分 (d×d) の抽出
        M = B[:d, :d]

        # 特異値計算（表示用）
        _, S_B, _ = torch.linalg.svd(B, full_matrices=False)
        _, S_M, _ = torch.linalg.svd(M, full_matrices=False)

        # rtol ベースでランク判定
        rank_B = torch.linalg.matrix_rank(B, rtol=rtol).item()
        rank_M = torch.linalg.matrix_rank(M, rtol=rtol).item()
        alpha = rank_B - rank_M

        if verbose:
            print(f"\n=== B行列解析 ===")
            print(f"B shape: {B.shape}")
            print(f"M shape: {M.shape}")
            print(f"rank(B): {rank_B} (rtol={rtol})")
            print(f"rank(M): {rank_M} (rtol={rtol})")
            print(f"alpha = rank(B) - rank(M): {alpha}")
            print(f"Top 5 singular values of B: {S_B[:5].tolist()}")
            print(f"Top 5 singular values of M: {S_M[:5].tolist()}")

        return {
            'B': B,
            'M': M,
            'rank_B': rank_B,
            'rank_M': rank_M,
            'alpha': alpha,
            'singular_values_B': S_B,
            'singular_values_M': S_M
        }


def extract_and_build_lsa(
    trained_model: FullAttention,
    d: int,
    d_l: int,
    rtol: float = 1e-3,
    device: str = 'cpu',
    verbose: bool = True
) -> Tuple[SelfLinearAttention, dict]:
    """
    訓練済みFull AttentionからB行列を抽出し、LSAモデルを構築

    Args:
        trained_model: 訓練済みFullAttentionモデル
        d: 入力次元
        d_l: 潜在次元
        rtol: ランク判定の相対許容誤差 (torch.linalg.matrix_rank の rtol)
        device: デバイス
        verbose: 詳細表示

    Returns:
        lsa_model: 構築されたLSAモデル
        analysis: B行列の解析結果
    """
    with torch.no_grad():
        # B行列を抽出
        B = trained_model.get_B_matrix()

        # ランク構造を解析
        analysis = analyze_B_matrix(B, d, rtol, verbose)
        rank_B = analysis['rank_B']

        # B = W_Q @ W_K.T を SVD 分解して W_Q, W_K を再構成
        U_B, S_B, Vh_B = torch.linalg.svd(B, full_matrices=False)

        # 使用するランク（d_l以下に制限）
        effective_r = min(rank_B, d_l)
        sqrt_S = torch.sqrt(S_B[:effective_r])
        U_r = U_B[:, :effective_r]
        Vh_r = Vh_B[:effective_r, :]

        # LSAモデルを構築
        input_dim = d + 1
        lsa_model = SelfLinearAttention(input_dim, d_l).to(device)

        # W_Q, W_K を設定
        W_Q_new = torch.zeros(input_dim, d_l, device=device)
        W_K_new = torch.zeros(input_dim, d_l, device=device)
        W_Q_new[:, :effective_r] = U_r @ torch.diag(sqrt_S)
        W_K_new[:, :effective_r] = Vh_r.T @ torch.diag(sqrt_S)

        lsa_model.W_Q.data = W_Q_new
        lsa_model.W_K.data = W_K_new

        if verbose:
            # 再構成誤差の確認
            B_reconstructed = lsa_model.W_Q @ lsa_model.W_K.T
            recon_error = torch.norm(B - B_reconstructed).item()
            print(f"\nB行列再構成誤差: {recon_error:.6e}")

        return lsa_model, analysis


# ============================================================
# 4. デモ/テスト用
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Full Attention モデルのテスト")
    print("=" * 60)

    # パラメータ
    d = 5
    d_l = 8
    N = 20
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 1. モデル訓練
    print("\n--- Step 1: Full Attention 訓練 ---")
    model, loss_history, _ = train_full_attention_with_checkpoints(
        d=d,
        d_l=d_l,
        N=N,
        num_tasks=5000,
        epochs=50,
        batch_size=500,
        lr=1e-3,
        checkpoint_epochs=[],
        device=device,
        verbose=True
    )

    print(f"最終損失: {loss_history[-1]:.6f}")

    # 2. B行列抽出とLSA構築
    print("\n--- Step 2: B行列抽出 & LSA構築 ---")
    lsa_model, analysis = extract_and_build_lsa(
        model, d, d_l, device=device, verbose=True
    )

    # 3. 理論値計算
    print("\n--- Step 3: 理論値計算 ---")
    rank_M = analysis['rank_M']
    rank_B = analysis['rank_B']
    lambda_M = compute_llc_theoretical_dln([d, d_l, d], rank_M)
    lambda_B = compute_llc_theoretical_dln([d + 1, d_l, d + 1], rank_B)

    print(f"λ_M [d={d}, d_l={d_l}, d={d}] (rank_M={rank_M}): {lambda_M:.4f}")
    print(f"λ_B [d+1={d+1}, d_l={d_l}, d+1={d+1}] (rank_B={rank_B}): {lambda_B:.4f}")

    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)
