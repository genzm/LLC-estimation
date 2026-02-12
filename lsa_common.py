# LSA 実験共通モジュール
# SGLD, SelfLinearAttention, 理論値計算, データ生成, LLC推定等を集約

from typing import Literal, Union, List, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np


# ============================================================
# 1. LSA モデル
# ============================================================

class SelfLinearAttention(nn.Module):
    """
    Self-Linear Attention (LSA) モデル for In-Context Learning

    モデル出力:
        h_LSA(X, w) = X @ W_Q @ W_K.T @ X.T ∈ R^{(N+1) × (N+1)}

    これはアテンション行列全体を出力する。
    損失関数は Frobenius ノルムの2乗を使用。
    """
    def __init__(self, input_dim, latent_dim, init_scale=0.01):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.W_Q = nn.Parameter(torch.randn(input_dim, latent_dim) * init_scale)
        self.W_K = nn.Parameter(torch.randn(input_dim, latent_dim) * init_scale)

    def get_effective_matrix(self):
        """B = W_Q @ W_K.T の d×d 部分を取得"""
        d = self.input_dim - 1
        return self.W_Q[:d, :] @ self.W_K[:d, :].T

    def get_effective_rank(self):
        M = self.get_effective_matrix()
        return torch.linalg.matrix_rank(M).item()

    def forward(self, X):
        """
        Args:
            X: (Batch, N+1, d+1) プロンプト行列

        Returns:
            H: (Batch, N+1, N+1) アテンション行列

        計算:
            H = X @ W_Q @ W_K.T @ X.T
        """
        # B = W_Q @ W_K.T: (d+1, d+1)
        B = self.W_Q @ self.W_K.T

        # H = X @ B @ X.T: (Batch, N+1, N+1)
        H = X @ B @ X.transpose(-2, -1)

        return H


# ============================================================
# 2. 理論値計算 (閉形式公式)
# ============================================================

def compute_llc_theoretical_dln(H: List[int], r: int) -> float:
    """
    2層DLN (H = [d, d_l, d]) の RLCT 理論値計算

    前提: H[0] = H[-1] (入力次元 = 出力次元)

    3つの領域で異なる公式:
    - 線形領域 (d_l <= r): λ = (d * (d_l + r) - d_l * r) / 2
    - 遷移領域 (r < d_l < 2d - r): λ = (4d * term - term²) / 8 (±0.5 for odd)
    - 飽和領域 (d_l >= 2d - r): λ = d² / 2

    Args:
        H: [d, d_l, d] 形式の次元リスト
        r: ランク

    Returns:
        RLCT 理論値
    """
    d = H[0]   # = H[-1]
    d_l = H[1]  # 中間層次元

    # バリデーション
    if r > d or r > d_l or r < 1 or d_l < 1:
        return 0.0

    if d_l >= 2 * d - r:
        # 飽和領域
        return (d ** 2) / 2.0
    elif d_l <= r:
        # 線形領域
        return (d * (d_l + r) - d_l * r) / 2.0
    else:
        # 遷移領域
        term = d_l + r
        base = 4 * d * term - (term ** 2)
        return base / 8.0 if term % 2 == 0 else (base + 1) / 8.0


# ============================================================
# 3. データ生成 (ICL プロンプト)
# ============================================================

def generate_icl_batch(num_tasks: int, d: int, N: int, Lambda=None, device='cpu'):
    """バッチ化された ICL プロンプトを生成（ベクトル化版）"""
    if Lambda is None:
        Lambda = torch.eye(d, device=device)

    m = torch.randn(num_tasks, d, device=device)
    L = torch.linalg.cholesky(Lambda)
    z = torch.randn(num_tasks, N + 1, d, device=device)
    x_all = z @ L.T

    x_examples = x_all[:, :N, :]
    x_query = x_all[:, N, :]

    y_examples = torch.einsum('td,tnd->tn', m, x_examples)
    y_query = torch.einsum('td,td->t', m, x_query)

    X_batch = torch.zeros(num_tasks, N + 1, d + 1, device=device)
    X_batch[:, :N, :d] = x_examples
    X_batch[:, :N, d] = y_examples
    X_batch[:, N, :d] = x_query
    X_batch[:, N, d] = 0

    return X_batch, y_query.unsqueeze(-1)


# ============================================================
# 4. SGLD オプティマイザ
# ============================================================

class SGLD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, noise_level=1., elasticity=0.,
                 temperature: Union[Literal['adaptive'], float]=1.,
                 bounding_box_size=None, num_samples=1, batch_size=None):
        defaults = dict(lr=lr, noise_level=noise_level, elasticity=elasticity,
                        temperature=temperature, bounding_box_size=None,
                        num_samples=num_samples, batch_size=batch_size)
        super(SGLD, self).__init__(params, defaults)

        for group in self.param_groups:
            if group['elasticity'] != 0:
                for p in group['params']:
                    param_state = self.state[p]
                    param_state['initial_param'] = torch.clone(p.data).detach()
            if group['temperature'] == "adaptive":
                group['temperature'] = np.log(group["num_samples"])

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                with torch.no_grad():
                    if p.grad is None:
                        continue

                    dw = p.grad * (group["num_samples"] / group["batch_size"]) / group['temperature']

                    if group['elasticity'] != 0:
                        initial_param = self.state[p]['initial_param']
                        dw.add_((p - initial_param), alpha=group['elasticity'])

                    p.add_(dw, alpha=-0.5 * group['lr'])

                    noise = torch.normal(mean=0., std=group['noise_level'],
                                         size=dw.size(), device=dw.device)
                    p.add_(noise, alpha=group['lr'] ** 0.5)

                    if group['bounding_box_size'] is not None:
                        torch.clamp_(p, min=-group['bounding_box_size'],
                                     max=group['bounding_box_size'])


# ============================================================
# 5. 損失関数
# ============================================================

def frobenius_loss(H_pred, H_true, normalize=True):
    """
    Frobenius ノルムの2乗を計算

    Args:
        H_pred: (Batch, N+1, N+1) 予測アテンション行列
        H_true: (Batch, N+1, N+1) 真のアテンション行列
        normalize: True の場合、要素数で正規化（MSE形式）

    Returns:
        loss: スカラー（普通は出力次元での平均をバッチレベルで和をとったもの、バッチサイズ全体の和だと勾配爆発する）
    """
    diff = H_pred - H_true
    if normalize:
        # MSE: 要素あたりの平均二乗誤差 × バッチサイズ
        # これにより N に依存しないスケールになる
        # 今回とは特にバッチサイズが(N+1)^2に比例するので、勾配爆発を抑えるために出力次元でmeanを取って、バッチサイズで和を取るようにする
        return (diff ** 2).mean(dim=(-2, -1)).sum()
    else:
        return (diff ** 2).sum()


# ============================================================
# 6. SGLD による LLC 推定 (正規版)
# ============================================================

def estimate_llc_with_sgld(
    teacher_model: SelfLinearAttention,
    x_data: torch.Tensor,
    *,
    num_steps: int = 5000,
    batch_size: int = 500,
    lr: float = 1e-5,
    elasticity: float = 1.0,
    noise_std: float = 1.0,
    burn_in_ratio: float = 0.9,
) -> Tuple[Optional[float], Optional[float]]:
    """
    SGLD を用いて LLC (Local Learning Coefficient) を推定

    teacher_model から input_dim, latent_dim, device を自動取得する。
    d_l / device 等の冗長引数は不要。

    Args:
        teacher_model: 真のパラメータを持つ LSA モデル
        x_data: (num_data, N+1, d+1) ICL プロンプトデータ
        num_steps: SGLD ステップ数
        batch_size: ミニバッチサイズ
        lr: 学習率
        elasticity: SGLD の弾性係数
        noise_std: 観測ノイズの標準偏差
        burn_in_ratio: burn-in 期間の比率

    Returns:
        (est_llc, std_error) or (None, None) (推定失敗時)
    """
    num_data = x_data.shape[0]
    input_dim = teacher_model.input_dim
    d_l = teacher_model.latent_dim
    device = next(teacher_model.parameters()).device

    teacher_model.eval()

    student_model = SelfLinearAttention(input_dim, d_l).to(device)
    student_model.load_state_dict(teacher_model.state_dict())

    initial_model = SelfLinearAttention(input_dim, d_l).to(device)
    initial_model.load_state_dict(teacher_model.state_dict())
    initial_model.eval()

    with torch.no_grad():
        H_true = teacher_model(x_data)
        H_data = H_true + torch.randn_like(H_true) * noise_std

    beta = 1.0 / np.log(num_data)
    nll_scale = 1.0 / (2.0 * noise_std**2)

    optimizer = SGLD(
        student_model.parameters(),
        lr=lr,
        num_samples=num_data,
        batch_size=batch_size,
        temperature=1.0,
        elasticity=elasticity
    )

    energy_diff_trace = []
    student_model.train()

    for _ in range(num_steps):
        indices = torch.randperm(num_data, device=device)[:batch_size]
        x_batch = x_data[indices]
        H_batch = H_data[indices]

        optimizer.zero_grad()

        H_pred = student_model(x_batch)
        batch_loss = frobenius_loss(H_pred, H_batch)
        energy_current = batch_loss * nll_scale * (num_data / batch_size)

        (beta * batch_loss * nll_scale).backward()
        optimizer.step()

        with torch.no_grad():
            H_pred_star = initial_model(x_batch)
            batch_loss_star = frobenius_loss(H_pred_star, H_batch)
            energy_star = batch_loss_star * nll_scale * (num_data / batch_size)

        diff = energy_current.item() - energy_star.item()
        energy_diff_trace.append(diff)

    burn_in = int(num_steps * burn_in_ratio)
    valid_diffs = [d for d in energy_diff_trace[burn_in:] if np.isfinite(d)]

    if len(valid_diffs) < 100:
        return None, None

    mean_energy_diff = np.mean(valid_diffs)
    est_llc = beta * mean_energy_diff
    std_error = beta * np.std(valid_diffs) / np.sqrt(len(valid_diffs))

    return est_llc, std_error


# ============================================================
# 7. ランク指定モデルビルダー (シンプル版)
# ============================================================

def create_lsa_rank_only(
    d: int, d_l: int, rank: int,
    init_scale: float = 0.1, device: str = 'cpu'
) -> Tuple[SelfLinearAttention, int]:
    """
    rank(B) のみ指定して LSA モデルを生成するシンプルなビルダー

    ランダム (d+1)×(d+1) 行列を SVD → 上位 rank 個の特異値を残す → W_Q, W_K に分解

    Args:
        d: 入力次元
        d_l: 潜在次元
        rank: 目標 rank(B)
        init_scale: 初期化スケール
        device: デバイス

    Returns:
        model: 生成された LSA モデル
        rank_actual: 実測 rank(B)
    """
    input_dim = d + 1

    with torch.no_grad():
        # ランダム行列を SVD → 上位 rank 個の特異値を残す
        A = torch.randn(input_dim, input_dim, device=device)
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)

        effective_r = min(rank, len(S), d_l)
        S_trunc = torch.abs(S[:effective_r]) * init_scale + 0.1
        sqrt_S = torch.sqrt(S_trunc)

        W_Q_new = torch.zeros(input_dim, d_l, device=device)
        W_K_new = torch.zeros(input_dim, d_l, device=device)
        W_Q_new[:, :effective_r] = U[:, :effective_r] @ torch.diag(sqrt_S)
        W_K_new[:, :effective_r] = Vh[:effective_r, :].T @ torch.diag(sqrt_S)

        model = SelfLinearAttention(input_dim, d_l, init_scale).to(device)
        model.W_Q.data = W_Q_new
        model.W_K.data = W_K_new

        # 実効ランク計測
        B_check = model.W_Q @ model.W_K.T
        rank_actual = torch.linalg.matrix_rank(B_check).item()

    return model, rank_actual


# ============================================================
# 8. 実験設定
# ============================================================

# (名前, d範囲, N, ステップサイズ, ステップ数, データ数, 試行回数)
# 注: d が大きいと H のスケールが大きくなるため、学習率を下げる
CONFIGS = [
    ("medium", (15, 25), 100, 5e-7, 10000, 100000, 100),
    ("large",  (25, 35), 100, 5e-7, 10000, 100000, 100),
]
