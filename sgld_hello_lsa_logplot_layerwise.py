# SGLD による LLC (Local Learning Coefficient) 推定 - SLA Log-Log Plot 版
# 論文: Lau et al. (2023) の Algorithm 1 に準拠
# 理論値: λ_SLA = λ_matrix + (d + 0.5)
# Figure 3 (Left) の再現: 複数規模のモデルで理論値 vs 推定値をプロット
#
# モデル出力: h_SLA(X, w) = X @ W_Q @ W_K.T @ X.T ∈ R^{(N+1) × (N+1)}
# 損失関数: K(w) = (1/2) E_X[||h_SLA(X, w) - h_SLA(X, w*)||_F^2]
#
# ※ Layer-wise ランク制限版（保存用）

from typing import Literal, Union, List, Tuple, Optional, Set
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# 1. SLA モデル
# ============================================================

class SelfLinearAttention(nn.Module):
    """
    Self-Linear Attention (SLA) モデル for In-Context Learning

    モデル出力:
        h_SLA(X, w) = X @ W_Q @ W_K.T @ X.T ∈ R^{(N+1) × (N+1)}

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
# 2. 理論値計算 (Aoyagi 2024, Theorem 1)
# ============================================================

def compute_deficiency(H: List[int], r: int) -> List[int]:
    return [h - r for h in H]


def find_bottleneck_set(M: List[int]) -> Set[int]:
    n = len(M)
    sorted_indices = list(np.argsort(M))
    sorted_M = [M[i] for i in sorted_indices]

    for k in range(1, n + 1):
        l = k - 1
        subset_indices = set(sorted_indices[:k])
        max_M_in = sorted_M[k - 1]
        sum_M_in = sum(sorted_M[:k])

        if k < n:
            min_M_out = sorted_M[k]
        else:
            min_M_out = float('inf')

        cond1 = max_M_in < min_M_out
        cond2 = True if l == 0 else sum_M_in >= l * max_M_in
        cond3 = True if (k >= n or l == 0) else sum_M_in < l * min_M_out

        if cond1 and cond2 and cond3:
            return subset_indices

    return set(range(n))


def compute_llc_theoretical_dln(H: List[int], r: int) -> float:
    M = compute_deficiency(H, r)
    bottleneck = find_bottleneck_set(M)
    l = len(bottleneck) - 1

    H_1 = H[0]
    H_Lp1 = H[-1]
    base_term = (-r**2 + r * (H_1 + H_Lp1)) / 2

    if l == 0:
        return base_term

    S_M = sum(M[i] for i in bottleneck)
    M_ceil = int(np.ceil(S_M / l))
    a = S_M - (M_ceil - 1) * l

    adj1 = a * (l - a) / (4 * l)
    adj2 = l * (l - 1) / 4 * (S_M / l)**2

    interaction = 0
    bottleneck_list = sorted(bottleneck)
    for i in range(len(bottleneck_list)):
        for j in range(i + 1, len(bottleneck_list)):
            interaction += M[bottleneck_list[i]] * M[bottleneck_list[j]]
    interaction /= 2

    return base_term + adj1 - adj2 + interaction


def compute_llc_theoretical_sla(d: int, d_l: int, r: int) -> float:
    """
    SLA モデルの LLC 理論値計算 (Note_1208_Yang.pdf 準拠)

    λ_SLA = λ_matrix + (d + 0.5)

    - λ_matrix: W_Q[:d,:] @ W_K[:d,:].T の RLCT (2層DLN として計算)
    - (d + 0.5): Cross part (履歴-クエリ相互作用) の寄与
    """
    max_possible_rank = min(d, d_l)
    if r > max_possible_rank:
        raise ValueError(f"Rank r={r} is impossible for d={d}, d_l={d_l}")

    H = [d, d_l, d]
    lambda_matrix = compute_llc_theoretical_dln(H, r)

    # Note Eq.(28)-(30) より: Cross part の寄与 = (d + 0.5)
    lambda_sla = lambda_matrix + (d + 0.5)

    return lambda_sla


# ============================================================
# 3. モデル生成 (Lau et al. 2023, Appendix J.1 準拠)
# ============================================================

def generate_random_sla_config(
    d_low: int = 5,
    d_high: int = 15,
    dl_multiplier_low: float = 1.5,
    dl_multiplier_high: float = 3.0
) -> Tuple[int, int]:
    """
    ランダムな SLA 構成を生成（d_l >= d を保証）

    Returns:
        d: 入力次元
        d_l: 潜在次元
        (ランク r は返さない - layer-wise に決定される)
    """
    d = np.random.randint(d_low, d_high + 1)
    multiplier = np.random.uniform(dl_multiplier_low, dl_multiplier_high)
    d_l = int(d * multiplier)
    return d, d_l


def create_random_rank_sla(d: int, d_l: int, init_scale: float = 0.1, device='cpu', target_alpha: int = None, max_retries: int = 50) -> Tuple[SelfLinearAttention, int, int]:
    """
    論文 Appendix J.1 に準拠した SLA モデル生成関数。
    W_Q と W_K それぞれに独立に確率0.5でランダムなランク制約を適用する。

    Args:
        d: 入力次元 (x の次元、つまり input_dim - 1)
        d_l: 潜在次元
        init_scale: 初期化スケール
        device: デバイス
        target_alpha: 目標とする α = rank(B) - rank(M)
            - None: 従来通り（ほぼ α = 1）
            - 1: 明示的に α = 1 を保証（従来と同じ）
            - 2: α = 2 を生成（W_Q[:d,:] と W_K[:d,:] が部分的に非重複な部分空間を使用）
        max_retries: target_alpha を達成するための最大リトライ回数

    Returns:
        model: 生成された SLA モデル
        rank_M: M = W_Q[:d,:] @ W_K[:d,:].T のランク (d×d 部分)
        rank_B: B = W_Q @ W_K.T のランク ((d+1)×(d+1) 全体)
    """
    for _ in range(max_retries):
        input_dim = d + 1
        model = SelfLinearAttention(input_dim, d_l, init_scale).to(device)

        if target_alpha == 2:
            # α = 2 を生成: W_Q[:d,:] と W_K[:d,:] が部分的に非重複な部分空間を使用
            # 重複次元数 < d となるように構成
            with torch.no_grad():
                if d_l < d + 1:
                    # d_l が小さすぎる場合、α=2 は不可能なので従来と同じ
                    pass
                else:
                    # d_l 次元空間を分割
                    # W_Q[:d,:] は前半 (0 ~ split_Q-1) を使用
                    # W_K[:d,:] は後半 (d_l - split_K ~ d_l-1) を使用
                    # 重複 = max(0, split_Q + split_K - d_l)
                    # 重複 < d となる条件: 2k - d_l < d → k < (d_l + d) / 2

                    max_k = (d_l + d - 1) // 2  # 重複 < d を保証
                    min_k = max(1, (d_l + 1) // 2)  # 最低限の重複を確保

                    if max_k < min_k:
                        max_k = min_k

                    k = np.random.randint(min_k, max_k + 1)
                    split_Q = k
                    split_K = k

                    # W_Q[:d,:] を構成: 前半 split_Q 次元のみ使用
                    W_Q_part = torch.randn(d, split_Q, device=device) * init_scale
                    model.W_Q.data[:d, :].zero_()
                    model.W_Q.data[:d, :split_Q] = W_Q_part

                    # W_K[:d,:] を構成: 後半 split_K 次元のみ使用
                    W_K_part = torch.randn(d, split_K, device=device) * init_scale
                    model.W_K.data[:d, :].zero_()
                    model.W_K.data[:d, d_l - split_K:] = W_K_part

                    # さらにランク制約を適用（確率 0.5）
                    if np.random.rand() < 0.5:
                        W_Q_d = model.W_Q.data[:d, :split_Q].clone()
                        max_rank_Q = min(d, split_Q)
                        if max_rank_Q > 0:
                            target_rank_Q = np.random.randint(1, max_rank_Q + 1)
                            if target_rank_Q < max_rank_Q:
                                U, S, Vh = torch.linalg.svd(W_Q_d, full_matrices=False)
                                S[target_rank_Q:] = 0
                                model.W_Q.data[:d, :split_Q] = U @ torch.diag(S) @ Vh

                    if np.random.rand() < 0.5:
                        W_K_d = model.W_K.data[:d, d_l - split_K:].clone()
                        max_rank_K = min(d, split_K)
                        if max_rank_K > 0:
                            target_rank_K = np.random.randint(1, max_rank_K + 1)
                            if target_rank_K < max_rank_K:
                                U, S, Vh = torch.linalg.svd(W_K_d, full_matrices=False)
                                S[target_rank_K:] = 0
                                model.W_K.data[:d, d_l - split_K:] = U @ torch.diag(S) @ Vh
        else:
            # 従来通り (target_alpha = None or 1)
            with torch.no_grad():
                if np.random.rand() < 0.5:
                    W_Q_d = model.W_Q.data[:d, :].clone()
                    max_rank_Q = min(d, d_l)
                    target_rank_Q = np.random.randint(0, max_rank_Q + 1)

                    if target_rank_Q == 0:
                        model.W_Q.data[:d, :].zero_()
                    elif target_rank_Q < max_rank_Q:
                        U, S, Vh = torch.linalg.svd(W_Q_d, full_matrices=False)
                        S[target_rank_Q:] = 0
                        model.W_Q.data[:d, :] = U @ torch.diag(S) @ Vh

                if np.random.rand() < 0.5:
                    W_K_d = model.W_K.data[:d, :].clone()
                    max_rank_K = min(d, d_l)
                    target_rank_K = np.random.randint(0, max_rank_K + 1)

                    if target_rank_K == 0:
                        model.W_K.data[:d, :].zero_()
                    elif target_rank_K < max_rank_K:
                        U, S, Vh = torch.linalg.svd(W_K_d, full_matrices=False)
                        S[target_rank_K:] = 0
                        model.W_K.data[:d, :] = U @ torch.diag(S) @ Vh

        # 実効ランクの計測
        with torch.no_grad():
            M = model.get_effective_matrix()
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
            max_s = S[0].item() if S.numel() > 0 else 0
            if max_s == 0:
                rank_M = 0
            else:
                tol = max_s * max(M.shape) * torch.finfo(M.dtype).eps
                rank_M = (S > tol).sum().item()

            B = model.W_Q @ model.W_K.T
            _, S_B, _ = torch.linalg.svd(B, full_matrices=False)
            max_s_B = S_B[0].item() if S_B.numel() > 0 else 0
            if max_s_B == 0:
                rank_B = 0
            else:
                tol_B = max_s_B * max(B.shape) * torch.finfo(B.dtype).eps
                rank_B = (S_B > tol_B).sum().item()

        # α をチェック
        alpha = rank_B - rank_M

        # target_alpha が指定されている場合、一致するまでリトライ
        if target_alpha is None:
            return model, rank_M, rank_B
        elif alpha == target_alpha:
            return model, rank_M, rank_B
        # リトライ

    # max_retries 回試してもダメだった場合、最後の結果を返す
    return model, rank_M, rank_B


# 旧実装 (Global Rank Constraint) - 参考用にコメントアウト
# def generate_random_sla_config_old(...) -> Tuple[int, int, int]:
#     """ランダムな SLA 構成を生成（d_l >= d を保証）"""
#     ...
#     return d, d_l, r
#
# def create_sla_with_rank(d: int, d_l: int, r: int, init_scale: float = 0.1):
#     """指定されたランクを持つ SLA モデルを作成 (旧実装)"""
#     ...


# ============================================================
# 4. データ生成 (ICL プロンプト)
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
# 5. SGLD オプティマイザ
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
# 6. 単一実験の関数化
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


def run_single_experiment(
    d_range: Tuple[int, int],
    dl_multiplier_range: Tuple[float, float] = (0.8, 1.2),  # d_l/d が小さいと α の効果が見える
    N: int = 100,
    lr: float = 1e-5,
    elasticity: float = 1.0,
    num_steps: int = 5000,
    num_data: int = 5000,
    batch_size: int = 500,
    noise_std: float = 1.0,
    burn_in_ratio: float = 0.9,
    device: str = 'cpu',
    target_alpha: int = None
) -> Optional[dict]:
    """
    単一の SLA 実験を実行

    損失関数: K(w) = (1/2) E_X[||h_SLA(X, w) - h_SLA(X, w*)||_F^2]

    Returns:
        dict: {params, true_llc, est_llc, std_error, d, d_l, r} or None (失敗時)
    """
    try:
        # --- モデル生成 (Layer-wise Rank Constraint) ---
        d, d_l = generate_random_sla_config(
            d_low=d_range[0], d_high=d_range[1],
            dl_multiplier_low=dl_multiplier_range[0],
            dl_multiplier_high=dl_multiplier_range[1]
        )

        # Teacher モデル生成 & 実効ランクの計測
        teacher_model, rank_M, rank_B = create_random_rank_sla(d, d_l, device=device, target_alpha=target_alpha)

        # 理論値計算
        # 1. λ_SLA = λ_matrix + (d + 0.5)  (d×d 部分 + 補正)
        llc_true = compute_llc_theoretical_sla(d, d_l, rank_M)

        # 2. λ_matrix (d×d 部分、補正なし)
        H_dln = [d, d_l, d]
        lambda_matrix = compute_llc_theoretical_dln(H_dln, rank_M)

        # 3. λ_full ((d+1)×(d+1) 全体を DLN とみなす)
        H_full = [d+1, d_l, d+1]
        lambda_full = compute_llc_theoretical_dln(H_full, rank_B)

        # rank_M=0 または llc_true=0 の場合はスキップ（相対誤差が計算できない）
        if rank_M == 0 or llc_true == 0:
            return None

        # Student/Initial モデル構築 (Teacher と同じ構造・パラメータで初期化)
        input_dim = d + 1
        student_model = SelfLinearAttention(input_dim, d_l).to(device)
        student_model.load_state_dict(teacher_model.state_dict())
        initial_model = SelfLinearAttention(input_dim, d_l).to(device)
        initial_model.load_state_dict(teacher_model.state_dict())
        initial_model.eval()

        # パラメータ数
        param_count = sum(p.numel() for p in student_model.parameters())

        # データ生成（X のみ使用、y は不要）
        x_data, _ = generate_icl_batch(num_data, d, N, device=device)

        # 真のアテンション行列を計算（ノイズ付き）
        with torch.no_grad():
            H_true = teacher_model(x_data)
            # 行列値ガウシアンノイズを追加
            H_data = H_true + torch.randn_like(H_true) * noise_std

        # SGLD 設定
        beta = 1.0 / np.log(num_data)
        optimizer = SGLD(
            student_model.parameters(),
            lr=lr,
            num_samples=num_data,
            batch_size=batch_size,
            temperature=1.0,
            elasticity=elasticity
        )

        # NLL スケール: p(Y|X,w) ∝ exp(-||Y - H||_F^2 / (2σ^2))
        nll_scale = 1.0 / (2.0 * noise_std**2)

        energy_diff_trace = []
        student_model.train()

        # SGLD ループ
        for step in range(num_steps):
            indices = torch.randperm(num_data, device=device)[:batch_size]
            x_batch = x_data[indices]
            H_batch = H_data[indices]

            optimizer.zero_grad()

            # Frobenius ノルム損失
            H_pred = student_model(x_batch)
            batch_loss = frobenius_loss(H_pred, H_batch)
            energy_current = batch_loss * nll_scale * (num_data / batch_size)

            (beta * batch_loss * nll_scale).backward()
            optimizer.step()

            # Control variate: 初期モデル（= 真のパラメータ）での損失
            with torch.no_grad():
                H_pred_star = initial_model(x_batch)
                batch_loss_star = frobenius_loss(H_pred_star, H_batch)
                energy_star = batch_loss_star * nll_scale * (num_data / batch_size)

            diff = energy_current.item() - energy_star.item()
            energy_diff_trace.append(diff)

        # LLC 計算
        burn_in = int(num_steps * burn_in_ratio)
        valid_diffs = energy_diff_trace[burn_in:]

        # NaN/Inf チェック
        valid_diffs = [d for d in valid_diffs if np.isfinite(d)]
        if len(valid_diffs) < 100:
            return None

        mean_energy_diff = np.mean(valid_diffs)
        llc_est = beta * mean_energy_diff
        std_error = beta * np.std(valid_diffs) / np.sqrt(len(valid_diffs))

        alpha = rank_B - rank_M
        return {
            "params": param_count,
            "true_llc": llc_true,        # λ_SLA = λ_matrix + (d + 0.5)
            "lambda_matrix": lambda_matrix,  # λ_matrix (d×d 部分)
            "lambda_full": lambda_full,   # λ_full ((d+1)×(d+1) 全体)
            "est_llc": llc_est,
            "std_error": std_error,
            "d": d,
            "d_l": d_l,
            "rank_M": rank_M,             # d×d 部分のランク
            "rank_B": rank_B,             # (d+1)×(d+1) 全体のランク
            "alpha": alpha                # α = rank(B) - rank(M)
        }

    except Exception as e:
        print(f"  実験失敗: {e}")
        return None


# ============================================================
# 7. 実験設定
# ============================================================

# (名前, d範囲, N, ステップサイズ, ステップ数, データ数, 試行回数)
# 注: d が大きいと H のスケールが大きくなるため、学習率を下げる
CONFIGS = [
    ("medium", (15, 25), 100, 5e-7, 10000, 100000, 100),
    ("large",  (25, 35), 100, 5e-7, 10000, 100000, 100),
]


def run_all_experiments(configs=CONFIGS, device='cpu', alpha_list=[None], dl_multiplier_range=(0.8, 1.2)):
    """
    全規模の実験を実行

    Args:
        configs: 実験設定のリスト
        device: デバイス
        alpha_list: 実行する α のリスト（None は従来通り ≈ α=1、2 は明示的に α=2 を生成）
        dl_multiplier_range: d_l/d の範囲
    """
    results = []

    for target_alpha in alpha_list:
        alpha_name = f"α={target_alpha}" if target_alpha else "α≈1"
        print(f"\n{'#'*60}")
        print(f"# {alpha_name} の実験")
        print(f"{'#'*60}")

        for name, d_range, N, lr, num_steps, num_data, num_trials in configs:
            print(f"\n{'='*50}")
            print(f"規模: {name} (d={d_range}, N={N}, n={num_data}) [{alpha_name}]")
            print(f"{'='*50}")

            for _ in tqdm(range(num_trials), desc=f"{name} {alpha_name}"):
                result = run_single_experiment(
                    d_range=d_range,
                    dl_multiplier_range=dl_multiplier_range,
                    N=N,
                    lr=lr,
                    num_steps=num_steps,
                    num_data=num_data,
                    device=device,
                    target_alpha=target_alpha
                )
                if result is not None:
                    result["scale"] = name
                    result["target_alpha"] = target_alpha
                    results.append(result)

            # 途中経過
            scale_results = [r for r in results if r["scale"] == name and r.get("target_alpha") == target_alpha]
            if scale_results:
                errors = [abs(r["est_llc"] - r["true_llc"]) / r["true_llc"] * 100
                          for r in scale_results]
                print(f"  成功: {len(scale_results)}/{num_trials}, 平均誤差: {np.mean(errors):.1f}%")

    return results


def plot_results(results: List[dict], save_path: str = None, dl_multiplier_range: Tuple[float, float] = None):
    """Log-Log プロットを作成（α でグループ化、2行3列）"""
    if not results:
        print("結果がありません")
        return

    # α でグループ分け
    alpha_values = sorted(set(r.get("alpha", 1) for r in results))

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # タイトルに dl_multiplier_range を表示
    if dl_multiplier_range:
        fig.suptitle(f'SLA LLC Estimation (dl_multiplier_range = {dl_multiplier_range})', fontsize=14, y=0.98)

    # 色マップ: α=1 は青系、α=2 は赤系
    colors = {1: 'tab:blue', 2: 'tab:red', 0: 'tab:green'}

    for row, alpha_val in enumerate(alpha_values[:2]):  # 最大2行
        alpha_results = [r for r in results if r.get("alpha", 1) == alpha_val]
        if not alpha_results:
            continue

        true_llcs = np.array([r["true_llc"] for r in alpha_results])
        lambda_matrix = np.array([r["lambda_matrix"] for r in alpha_results])
        lambda_full = np.array([r["lambda_full"] for r in alpha_results])
        est_llcs = np.array([r["est_llc"] for r in alpha_results])
        color = colors.get(alpha_val, 'tab:gray')

        # === 列1: λ_SLA vs 推定値 ===
        ax1 = axes[row, 0]
        ax1.scatter(true_llcs, est_llcs, c=color, alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

        min_val1 = min(true_llcs.min(), est_llcs.min()) * 0.8
        max_val1 = max(true_llcs.max(), est_llcs.max()) * 1.2
        ax1.plot([min_val1, max_val1], [min_val1, max_val1], 'k--', alpha=0.5, label='y=x')

        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlim(min_val1, max_val1)
        ax1.set_ylim(min_val1, max_val1)

        ax1.set_xlabel(r'$\lambda_{SLA} = \lambda_{matrix} + (d + 0.5)$', fontsize=11)
        ax1.set_ylabel(r'Estimated LLC $\hat{\lambda}(w^*)$', fontsize=11)
        ax1.set_title(f'α={alpha_val}: λ_SLA vs Estimated', fontsize=11)

        errors1 = np.abs(est_llcs - true_llcs) / true_llcs * 100
        textstr1 = f'N={len(alpha_results)}\nMean: {np.mean(errors1):.1f}%\nMedian: {np.median(errors1):.1f}%'
        ax1.text(0.05, 0.95, textstr1, transform=ax1.transAxes, fontsize=9,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)

        # === 列2: λ_matrix vs 推定値 ===
        ax2 = axes[row, 1]
        ax2.scatter(lambda_matrix, est_llcs, c=color, alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

        min_val2 = min(lambda_matrix.min(), est_llcs.min()) * 0.8
        max_val2 = max(lambda_matrix.max(), est_llcs.max()) * 1.2
        ax2.plot([min_val2, max_val2], [min_val2, max_val2], 'k--', alpha=0.5, label='y=x')

        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlim(min_val2, max_val2)
        ax2.set_ylim(min_val2, max_val2)

        ax2.set_xlabel(r'$\lambda_{matrix}$ [d, d_l, d]', fontsize=11)
        ax2.set_ylabel(r'Estimated LLC $\hat{\lambda}(w^*)$', fontsize=11)
        ax2.set_title(f'α={alpha_val}: λ_matrix vs Estimated', fontsize=11)

        errors2 = np.abs(est_llcs - lambda_matrix) / lambda_matrix * 100
        textstr2 = f'N={len(alpha_results)}\nMean: {np.mean(errors2):.1f}%\nMedian: {np.median(errors2):.1f}%'
        ax2.text(0.05, 0.95, textstr2, transform=ax2.transAxes, fontsize=9,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)

        # === 列3: λ_full vs 推定値 ===
        ax3 = axes[row, 2]
        ax3.scatter(lambda_full, est_llcs, c=color, alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

        min_val3 = min(lambda_full.min(), est_llcs.min()) * 0.8
        max_val3 = max(lambda_full.max(), est_llcs.max()) * 1.2
        ax3.plot([min_val3, max_val3], [min_val3, max_val3], 'k--', alpha=0.5, label='y=x')

        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.set_xlim(min_val3, max_val3)
        ax3.set_ylim(min_val3, max_val3)

        ax3.set_xlabel(r'$\lambda_{full}$ [d+1, d_l, d+1]', fontsize=11)
        ax3.set_ylabel(r'Estimated LLC $\hat{\lambda}(w^*)$', fontsize=11)
        ax3.set_title(f'α={alpha_val}: λ_full vs Estimated', fontsize=11)

        errors3 = np.abs(est_llcs - lambda_full) / lambda_full * 100
        textstr3 = f'N={len(alpha_results)}\nMean: {np.mean(errors3):.1f}%\nMedian: {np.median(errors3):.1f}%'
        ax3.text(0.05, 0.95, textstr3, transform=ax3.transAxes, fontsize=9,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax3.legend(loc='lower right')
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"保存: {save_path}")

    plt.show()


def print_summary(results: List[dict]):
    """結果のサマリーを表示（α でグループ化）"""
    if not results:
        return

    print("\n" + "=" * 70)
    print("SLA 実験結果サマリー")
    print("=" * 70)

    # α でグループ分け
    alpha_values = sorted(set(r.get("alpha", 1) for r in results))

    for alpha_val in alpha_values:
        alpha_results = [r for r in results if r.get("alpha", 1) == alpha_val]
        if not alpha_results:
            continue

        print(f"\n{'='*60}")
        print(f"α = {alpha_val} のケース ({len(alpha_results)} 試行)")
        print(f"{'='*60}")

        # 3つの理論値との誤差を計算
        errors_sla = [abs(r["est_llc"] - r["true_llc"]) / r["true_llc"] * 100 for r in alpha_results]
        errors_matrix = [abs(r["est_llc"] - r["lambda_matrix"]) / r["lambda_matrix"] * 100 for r in alpha_results]
        errors_full = [abs(r["est_llc"] - r["lambda_full"]) / r["lambda_full"] * 100 for r in alpha_results]

        print(f"\n{'理論値':<35} {'平均誤差':<12} {'中央値誤差':<12}")
        print("-" * 60)
        print(f"{'(1) λ_SLA = λ_matrix + (d+0.5)':<35} {np.mean(errors_sla):>8.1f}%    {np.median(errors_sla):>8.1f}%")
        print(f"{'(2) λ_matrix [d, d_l, d]':<35} {np.mean(errors_matrix):>8.1f}%    {np.median(errors_matrix):>8.1f}%")
        print(f"{'(3) λ_full [d+1, d_l, d+1]':<35} {np.mean(errors_full):>8.1f}%    {np.median(errors_full):>8.1f}%")

        # ランクの統計
        rank_M_list = [r["rank_M"] for r in alpha_results]
        rank_B_list = [r["rank_B"] for r in alpha_results]

        print(f"\n--- ランク統計 ---")
        print(f"rank(M) [d×d]:       平均 {np.mean(rank_M_list):.1f}, 範囲 [{min(rank_M_list)}, {max(rank_M_list)}]")
        print(f"rank(B) [(d+1)×(d+1)]: 平均 {np.mean(rank_B_list):.1f}, 範囲 [{min(rank_B_list)}, {max(rank_B_list)}]")

    # 全体のα分布
    all_alpha = [r.get("alpha", 1) for r in results]
    print(f"\n{'='*60}")
    print(f"全体: α の分布 = {dict(zip(*np.unique(all_alpha, return_counts=True)))}")
    print("=" * 70)


# ============================================================
# 8. メイン実行
# ============================================================

# dl_multiplier_range の設定（α の効果を見るには小さい値が必要）
DL_MULTIPLIER_RANGE = (0.95, 1.05)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用デバイス: {device}")
    print(f"dl_multiplier_range: {DL_MULTIPLIER_RANGE}")

    results = run_all_experiments(configs=CONFIGS, device=device, dl_multiplier_range=DL_MULTIPLIER_RANGE)

    print_summary(results)

    plot_results(results, save_path="figures/lsa_llc_comparison.png", dl_multiplier_range=DL_MULTIPLIER_RANGE)
