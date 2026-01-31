# SGLD による LLC (Local Learning Coefficient) 推定 - SLA Log-Log Plot 版
# 論文: Lau et al. (2023) の Algorithm 1 に準拠
# 理論値: λ_SLA = λ_matrix + (d + 0.5)
# Figure 3 (Left) の再現: 複数規模のモデルで理論値 vs 推定値をプロット
#
# モデル出力: h_SLA(X, w) = X @ W_Q @ W_K.T @ X.T ∈ R^{(N+1) × (N+1)}
# 損失関数: K(w) = (1/2) E_X[||h_SLA(X, w) - h_SLA(X, w*)||_F^2]
#
# ※ Global ランク制限版: B = W_Q @ W_K.T を直接 rank = r に制限
#    d_l の範囲: r ≤ d_l ≤ 2d - r（ボトルネック制御用）

from typing import Literal, Union, List, Tuple, Optional
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
    if r > d or r < 1 or d_l < 1:
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
# 3. モデル生成 (Global Rank Constraint + α 制御版)
# ============================================================

def generate_random_sla_config(
    d_low: int = 5,
    d_high: int = 15,
    dl_ratio_range: Tuple[float, float] = (0.5, 1.5),
    alpha_list: List[int] = [0, 1, 2],
    max_retries: int = 100
) -> Optional[Tuple[int, int, int, int]]:
    """
    α 制御付き SLA 構成を生成

    決定順序:
        1. d を決める
        2. d_l を決める (d * ratio)
        3. α を決める (0, 1, 2)
        4. r_M を決める (制約を満たす範囲で)

    制約:
        - r_M ≥ 1
        - r_M ≤ d_l - α (分解可能条件: r_B = r_M + α ≤ d_l)
        - r_M ≤ 2d - d_l (ボトルネック条件)
        - r_M ≤ d - 1 (フルランクを避ける)

    Args:
        d_low, d_high: d の範囲
        dl_ratio_range: d_l/d の比率の範囲
        alpha_list: 選択可能な α のリスト
        max_retries: 有効な組み合わせを見つけるまでの最大リトライ回数

    Returns:
        (d, d_l, r_M, alpha) or None (有効な組み合わせが見つからない場合)
    """
    for _ in range(max_retries):
        # 1. d を決める
        d = np.random.randint(d_low, d_high + 1)

        # 2. d_l を決める
        dl_ratio = np.random.uniform(dl_ratio_range[0], dl_ratio_range[1])
        d_l = max(1, int(d * dl_ratio))

        # 3. α を決める
        alpha = np.random.choice(alpha_list)

        # 4. r_M の範囲を計算
        r_M_min = 1
        # r_M_max = min(d_l - alpha, 2 * d - d_l, d - 1)
        r_M_max = min(d_l - alpha, d - 1)

        # 有効な範囲かチェック
        if r_M_max >= r_M_min:
            r_M = np.random.randint(r_M_min, r_M_max + 1)
            return d, d_l, r_M, alpha

    # 有効な組み合わせが見つからなかった
    return None


def create_sla_with_alpha(
    d: int,
    d_l: int,
    r_M: int,
    alpha: int,
    init_scale: float = 0.1,
    device: str = 'cpu'
) -> Tuple[SelfLinearAttention, int, int]:
    """
    α を指定して SLA モデルを生成

    B = [ M    | b_col ]   (d+1) × (d+1)
        [------+-------]
        [b_row | corner]

    α の制御:
        - α = 0: b_col ∈ Col(M), b_row ∈ Row(M)
        - α = 1: b_col ∉ Col(M) xor b_row ∉ Row(M)
        - α = 2: b_col ∉ Col(M) and b_row ∉ Row(M)

    Args:
        d: 入力次元
        d_l: 潜在次元
        r_M: M のランク
        alpha: 目標 α (0, 1, 2)
        init_scale: 初期化スケール
        device: デバイス

    Returns:
        model: 生成された SLA モデル
        rank_M: 実測 rank(M)
        rank_B: 実測 rank(B)
    """
    input_dim = d + 1
    r_B = r_M + alpha  # B の目標ランク

    with torch.no_grad():
        # Step 1: M (d×d) を rank = r_M で作成
        # M = U_M @ diag(S_M) @ V_M.T
        U_M = torch.randn(d, r_M, device=device)
        U_M, _ = torch.linalg.qr(U_M)  # 正規直交化
        V_M = torch.randn(d, r_M, device=device)
        V_M, _ = torch.linalg.qr(V_M)
        S_M = torch.abs(torch.randn(r_M, device=device)) * init_scale + 0.1  # 正の特異値

        M = U_M @ torch.diag(S_M) @ V_M.T  # d × d, rank = r_M

        # Step 2: α に応じて b_col, b_row を構成
        if alpha == 0:
            # b_col ∈ Col(M), b_row ∈ Row(M)
            # b_col = M @ (何かのベクトル) = U_M @ (何か)
            coef_col = torch.randn(r_M, device=device) * init_scale
            b_col = U_M @ coef_col  # d × 1, Col(M) 内

            # b_row = (何か) @ M = (何か) @ V_M.T
            coef_row = torch.randn(r_M, device=device) * init_scale
            b_row = (V_M @ coef_row).unsqueeze(0)  # 1 × d, Row(M) 内

        elif alpha == 1:
            # 片方だけ独立 (ここでは b_col を独立にする)
            # b_col ∉ Col(M): U_M の直交補空間からベクトルを取る
            U_M_perp = torch.randn(d, 1, device=device)
            # Gram-Schmidt で U_M に直交化
            for i in range(r_M):
                U_M_perp = U_M_perp - (U_M[:, i:i+1].T @ U_M_perp) * U_M[:, i:i+1]
            U_M_perp = U_M_perp / (torch.norm(U_M_perp) + 1e-8)
            b_col = U_M_perp.squeeze() * init_scale  # d, Col(M) 外

            # b_row ∈ Row(M)
            coef_row = torch.randn(r_M, device=device) * init_scale
            b_row = (V_M @ coef_row).unsqueeze(0)  # 1 × d, Row(M) 内

        elif alpha == 2:
            # 両方独立
            # b_col ∉ Col(M)
            U_M_perp = torch.randn(d, 1, device=device)
            for i in range(r_M):
                U_M_perp = U_M_perp - (U_M[:, i:i+1].T @ U_M_perp) * U_M[:, i:i+1]
            U_M_perp = U_M_perp / (torch.norm(U_M_perp) + 1e-8)
            b_col = U_M_perp.squeeze() * init_scale

            # b_row ∉ Row(M)
            V_M_perp = torch.randn(d, 1, device=device)
            for i in range(r_M):
                V_M_perp = V_M_perp - (V_M[:, i:i+1].T @ V_M_perp) * V_M[:, i:i+1]
            V_M_perp = V_M_perp / (torch.norm(V_M_perp) + 1e-8)
            b_row = V_M_perp.squeeze().unsqueeze(0) * init_scale  # 1 × d

        else:
            raise ValueError(f"alpha must be 0, 1, or 2, got {alpha}")

        # Step 3: corner を決める (任意)
        corner = torch.randn(1, 1, device=device) * init_scale

        # Step 4: B を組み立てる
        # B = [[M, b_col], [b_row, corner]]
        B = torch.zeros(input_dim, input_dim, device=device)
        B[:d, :d] = M
        B[:d, d:] = b_col.unsqueeze(1)
        B[d:, :d] = b_row
        B[d:, d:] = corner

        # Step 5: B を W_Q, W_K に分解
        # B = W_Q @ W_K.T, where W_Q, W_K are (d+1) × d_l
        U_B, S_B, Vh_B = torch.linalg.svd(B, full_matrices=False)

        # 上位 r_B 個の特異値を使用
        effective_r = min(r_B, len(S_B), d_l)
        sqrt_S = torch.sqrt(S_B[:effective_r])
        U_r = U_B[:, :effective_r]
        Vh_r = Vh_B[:effective_r, :]

        W_Q_new = torch.zeros(input_dim, d_l, device=device)
        W_K_new = torch.zeros(input_dim, d_l, device=device)
        W_Q_new[:, :effective_r] = U_r @ torch.diag(sqrt_S)
        W_K_new[:, :effective_r] = Vh_r.T @ torch.diag(sqrt_S)

        # モデルを作成
        model = SelfLinearAttention(input_dim, d_l, init_scale).to(device)
        model.W_Q.data = W_Q_new
        model.W_K.data = W_K_new

        # 実効ランクの計測
        M_check = model.get_effective_matrix()
        _, S_M_check, _ = torch.linalg.svd(M_check, full_matrices=False)
        max_s_M = S_M_check[0].item() if S_M_check.numel() > 0 else 0
        if max_s_M == 0:
            rank_M_actual = 0
        else:
            tol_M = max_s_M * max(M_check.shape) * torch.finfo(M_check.dtype).eps
            rank_M_actual = (S_M_check > tol_M).sum().item()

        B_check = model.W_Q @ model.W_K.T
        _, S_B_check, _ = torch.linalg.svd(B_check, full_matrices=False)
        max_s_B = S_B_check[0].item() if S_B_check.numel() > 0 else 0
        if max_s_B == 0:
            rank_B_actual = 0
        else:
            tol_B = max_s_B * max(B_check.shape) * torch.finfo(B_check.dtype).eps
            rank_B_actual = (S_B_check > tol_B).sum().item()

    return model, rank_M_actual, rank_B_actual


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
    dl_ratio_range: Tuple[float, float] = (0.5, 1.5),
    alpha_list: List[int] = [0, 1, 2],
    N: int = 100,
    lr: float = 1e-5,
    elasticity: float = 1.0,
    num_steps: int = 5000,
    num_data: int = 5000,
    batch_size: int = 500,
    noise_std: float = 1.0,
    burn_in_ratio: float = 0.9,
    device: str = 'cpu'
) -> Optional[dict]:
    """
    単一の SLA 実験を実行 (α 制御版)

    損失関数: K(w) = (1/2) E_X[||h_SLA(X, w) - h_SLA(X, w*)||_F^2]

    Args:
        d_range: d の範囲
        dl_ratio_range: d_l/d の比率の範囲
        alpha_list: 選択可能な α のリスト
        ...

    Returns:
        dict: {params, true_llc, est_llc, std_error, d, d_l, r_M, alpha, ...} or None (失敗時)
    """
    try:
        # --- モデル生成 (α 制御版) ---
        config = generate_random_sla_config(
            d_low=d_range[0], d_high=d_range[1],
            dl_ratio_range=dl_ratio_range,
            alpha_list=alpha_list
        )
        if config is None:
            return None

        d, d_l, r_M, target_alpha = config

        # Teacher モデル生成 & 実効ランクの計測
        teacher_model, rank_M, rank_B = create_sla_with_alpha(d, d_l, r_M, target_alpha, device=device)

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

        actual_alpha = rank_B - rank_M
        return {
            "params": param_count,
            "true_llc": llc_true,        # λ_SLA = λ_matrix + (d + 0.5)
            "lambda_matrix": lambda_matrix,  # λ_matrix (d×d 部分)
            "lambda_full": lambda_full,   # λ_full ((d+1)×(d+1) 全体)
            "est_llc": llc_est,
            "std_error": std_error,
            "d": d,
            "d_l": d_l,
            "r_M": r_M,                   # 目標 rank(M)
            "target_alpha": target_alpha, # 目標 α
            "rank_M": rank_M,             # d×d 部分の実効ランク
            "rank_B": rank_B,             # (d+1)×(d+1) 全体の実効ランク
            "alpha": actual_alpha         # 実測 α = rank(B) - rank(M)
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


def run_all_experiments(
    configs=CONFIGS,
    device='cpu',
    dl_ratio_range=(0.5, 1.5),
    alpha_list=[0, 1, 2]
):
    """
    全規模の実験を実行 (α 制御版)

    Args:
        configs: 実験設定のリスト
        device: デバイス
        dl_ratio_range: d_l/d の比率の範囲
        alpha_list: 選択可能な α のリスト
    """
    results = []

    print(f"\n{'#'*60}")
    print(f"# α 制御版実験")
    print(f"# dl_ratio_range: {dl_ratio_range}")
    print(f"# alpha_list: {alpha_list}")
    print(f"{'#'*60}")

    for name, d_range, N, lr, num_steps, num_data, num_trials in configs:
        print(f"\n{'='*50}")
        print(f"規模: {name} (d={d_range}, N={N}, n={num_data})")
        print(f"{'='*50}")

        for _ in tqdm(range(num_trials), desc=f"{name}"):
            result = run_single_experiment(
                d_range=d_range,
                dl_ratio_range=dl_ratio_range,
                alpha_list=alpha_list,
                N=N,
                lr=lr,
                num_steps=num_steps,
                num_data=num_data,
                device=device
            )
            if result is not None:
                result["scale"] = name
                results.append(result)

        # 途中経過
        scale_results = [r for r in results if r["scale"] == name]
        if scale_results:
            errors = [abs(r["est_llc"] - r["true_llc"]) / r["true_llc"] * 100
                      for r in scale_results]
            alpha_dist = {}
            for r in scale_results:
                a = r.get("alpha", -1)
                alpha_dist[a] = alpha_dist.get(a, 0) + 1
            print(f"  成功: {len(scale_results)}/{num_trials}, 平均誤差: {np.mean(errors):.1f}%, α分布: {alpha_dist}")

    return results


def plot_results(results: List[dict], save_path: str = None, dl_ratio_range: Tuple[float, float] = None, alpha_list: List[int] = None):
    """Log-Log プロットを作成（α でグループ化、最大3行3列）"""
    if not results:
        print("結果がありません")
        return

    # α でグループ分け
    alpha_values = sorted(set(r.get("alpha", 0) for r in results))
    num_alphas = len(alpha_values)

    fig, axes = plt.subplots(num_alphas, 3, figsize=(20, 4 * num_alphas + 2))
    if num_alphas == 1:
        axes = axes.reshape(1, -1)

    # タイトル
    title = 'LSA LLC Estimation (α Control)'
    if dl_ratio_range:
        title += f'\ndl_ratio_range = {dl_ratio_range}'
    if alpha_list:
        title += f', alpha_list = {alpha_list}'
    fig.suptitle(title, fontsize=14, y=0.98)

    # 全結果のパラメータ数の範囲（colorbar のスケール統一用）
    all_params = np.array([r["params"] for r in results])
    vmin, vmax = np.log10(all_params.min()), np.log10(all_params.max())

    for row, alpha_val in enumerate(alpha_values):
        alpha_results = [r for r in results if r.get("alpha", 0) == alpha_val]
        if not alpha_results:
            continue

        true_llcs = np.array([r["true_llc"] for r in alpha_results])
        lambda_matrix = np.array([r["lambda_matrix"] for r in alpha_results])
        lambda_full = np.array([r["lambda_full"] for r in alpha_results])
        est_llcs = np.array([r["est_llc"] for r in alpha_results])
        params = np.array([r["params"] for r in alpha_results])
        log_params = np.log10(params)

        # === 列1: λ_SLA vs 推定値 ===
        ax1 = axes[row, 0]
        sc1 = ax1.scatter(true_llcs, est_llcs, c=log_params, cmap='magma', vmin=vmin, vmax=vmax,
                          alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

        min_val1 = min(true_llcs.min(), est_llcs.min()) * 0.8
        max_val1 = max(true_llcs.max(), est_llcs.max()) * 1.2
        ax1.plot([min_val1, max_val1], [min_val1, max_val1], 'k--', alpha=0.5, label='y=x')

        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlim(min_val1, max_val1)
        ax1.set_ylim(min_val1, max_val1)

        ax1.set_xlabel(r'$\lambda_{LSA} = \lambda_{matrix} + (d + 0.5)$', fontsize=11)
        ax1.set_ylabel(r'Estimated LLC $\hat{\lambda}(w^*)$', fontsize=11)
        ax1.set_title(f'α={alpha_val}: λ_LSA vs Estimated', fontsize=11)

        errors1 = np.abs(est_llcs - true_llcs) / true_llcs * 100
        textstr1 = f'N={len(alpha_results)}\nMean: {np.mean(errors1):.1f}%\nMedian: {np.median(errors1):.1f}%'
        ax1.text(0.05, 0.95, textstr1, transform=ax1.transAxes, fontsize=9,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)

        # === 列2: λ_matrix vs 推定値 ===
        ax2 = axes[row, 1]
        sc2 = ax2.scatter(lambda_matrix, est_llcs, c=log_params, cmap='magma', vmin=vmin, vmax=vmax,
                          alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

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
        sc3 = ax3.scatter(lambda_full, est_llcs, c=log_params, cmap='magma', vmin=vmin, vmax=vmax,
                          alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

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

        # 各行の右端に colorbar を追加（最後の列のみ）
        cbar = plt.colorbar(sc3, ax=ax3)
        cbar.set_label('Log10 Params', fontsize=9)

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

        # ランクと次元の統計
        r_list = [r.get("r_M", r["rank_M"]) for r in alpha_results]
        d_list = [r["d"] for r in alpha_results]
        d_l_list = [r["d_l"] for r in alpha_results]
        rank_M_list = [r["rank_M"] for r in alpha_results]
        rank_B_list = [r["rank_B"] for r in alpha_results]

        print(f"\n--- 次元・ランク統計 ---")
        print(f"d:                   平均 {np.mean(d_list):.1f}, 範囲 [{min(d_list)}, {max(d_list)}]")
        print(f"d_l:                 平均 {np.mean(d_l_list):.1f}, 範囲 [{min(d_l_list)}, {max(d_l_list)}]")
        print(f"r_M (目標):          平均 {np.mean(r_list):.1f}, 範囲 [{min(r_list)}, {max(r_list)}]")
        print(f"rank(M) [d×d]:       平均 {np.mean(rank_M_list):.1f}, 範囲 [{min(rank_M_list)}, {max(rank_M_list)}]")
        print(f"rank(B) [(d+1)×(d+1)]: 平均 {np.mean(rank_B_list):.1f}, 範囲 [{min(rank_B_list)}, {max(rank_B_list)}]")

    # 全体のα分布
    all_alpha = [r.get("alpha", 0) for r in results]
    print(f"\n{'='*60}")
    print(f"全体: α の分布 = {dict(zip(*np.unique(all_alpha, return_counts=True)))}")
    print("=" * 70)


# ============================================================
# 8. メイン実行
# ============================================================

# 設定
# DL_RATIO_RANGE = (0.5, 1.5)  # d_l/d の比率の範囲
DL_RATIO_RANGE = (2.5, 3.5)
ALPHA_LIST = [0, 1, 2]       # 選択可能な α のリスト

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用デバイス: {device}")
    print(f"dl_ratio_range: {DL_RATIO_RANGE}")
    print(f"alpha_list: {ALPHA_LIST}")

    results = run_all_experiments(
        configs=CONFIGS,
        device=device,
        dl_ratio_range=DL_RATIO_RANGE,
        alpha_list=ALPHA_LIST
    )

    print_summary(results)

    plot_results(results, save_path="figures/lsa_llc_comparison_d_l_large.png", dl_ratio_range=DL_RATIO_RANGE, alpha_list=ALPHA_LIST)