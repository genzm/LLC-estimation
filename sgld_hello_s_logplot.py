# SGLD による LLC (Local Learning Coefficient) 推定 - SLA Log-Log Plot 版
# 論文: Lau et al. (2023) の Algorithm 1 に準拠
# 理論値: λ_SLA = λ_matrix + (d + 0.5)
# Figure 3 (Left) の再現: 複数規模のモデルで理論値 vs 推定値をプロット
#
# モデル出力: h_SLA(X, w) = X @ W_Q @ W_K.T @ X.T ∈ R^{(N+1) × (N+1)}
# 損失関数: K(w) = (1/2) E_X[||h_SLA(X, w) - h_SLA(X, w*)||_F^2]

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
# 3. モデル生成
# ============================================================

def generate_random_sla_config(
    d_low: int = 5,
    d_high: int = 15,
    dl_multiplier_low: float = 1.5,
    dl_multiplier_high: float = 3.0
) -> Tuple[int, int, int]:
    """ランダムな SLA 構成を生成（d_l >= d を保証）"""
    d = np.random.randint(d_low, d_high + 1)
    multiplier = np.random.uniform(dl_multiplier_low, dl_multiplier_high)
    d_l = int(d * multiplier)
    max_rank = d
    if np.random.rand() < 0.5:
        r = np.random.randint(1, max_rank + 1)
    else:
        r = max_rank
    return d, d_l, r


def create_sla_with_rank(d: int, d_l: int, r: int, init_scale: float = 0.1) -> SelfLinearAttention:
    """指定されたランクを持つ SLA モデルを作成"""
    max_possible_rank = min(d, d_l)
    if r > max_possible_rank:
        raise ValueError(f"Rank r={r} is impossible for d={d}, d_l={d_l}")

    input_dim = d + 1
    model = SelfLinearAttention(input_dim, d_l, init_scale)

    with torch.no_grad():
        if r == 0:
            model.W_Q.zero_()
            model.W_K.zero_()
        else:
            W_Q_d = model.W_Q[:d, :].clone()
            W_K_d = model.W_K[:d, :].clone()
            M = W_Q_d @ W_K_d.T

            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
            S_truncated = S.clone()
            S_truncated[r:] = 0
            M_rank_r = U @ torch.diag(S_truncated) @ Vh

            W_Q_pinv = torch.linalg.pinv(W_Q_d)
            new_W_K_d = (W_Q_pinv @ M_rank_r).T
            model.W_K[:d, :] = new_W_K_d

    return model


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
    dl_multiplier_range: Tuple[float, float] = (1.5, 3.0),
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
    単一の SLA 実験を実行

    損失関数: K(w) = (1/2) E_X[||h_SLA(X, w) - h_SLA(X, w*)||_F^2]

    Returns:
        dict: {params, true_llc, est_llc, std_error, d, d_l, r} or None (失敗時)
    """
    try:
        # --- モデル生成 ---
        d, d_l, r = generate_random_sla_config(
            d_low=d_range[0], d_high=d_range[1],
            dl_multiplier_low=dl_multiplier_range[0],
            dl_multiplier_high=dl_multiplier_range[1]
        )

        # 理論値計算
        llc_true = compute_llc_theoretical_sla(d, d_l, r)

        # r=0 かつ llc_true=0 の場合はスキップ
        if llc_true == 0:
            return None

        # モデル構築
        input_dim = d + 1
        teacher_model = create_sla_with_rank(d, d_l, r).to(device)
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

        return {
            "params": param_count,
            "true_llc": llc_true,
            "est_llc": llc_est,
            "std_error": std_error,
            "d": d,
            "d_l": d_l,
            "r": r
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
    ("small",  (3, 8),   100, 5e-7, 10000, 100000,   30),
    ("medium", (8, 15),  100, 5e-7, 10000, 100000,   30),
    ("large",  (15, 25), 100, 1e-7, 50000, 1000000,  20),
]


def run_all_experiments(configs=CONFIGS, device='cpu'):
    """全規模の実験を実行"""
    results = []

    for name, d_range, N, lr, num_steps, num_data, num_trials in configs:
        print(f"\n{'='*50}")
        print(f"規模: {name} (d={d_range}, N={N}, n={num_data})")
        print(f"{'='*50}")

        for trial in tqdm(range(num_trials), desc=f"{name}"):
            result = run_single_experiment(
                d_range=d_range,
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
            print(f"  成功: {len(scale_results)}/{num_trials}, 平均誤差: {np.mean(errors):.1f}%")

    return results


def plot_results(results: List[dict], save_path: str = None):
    """Log-Log プロットを作成 (Figure 3 Left 再現)"""
    if not results:
        print("結果がありません")
        return

    true_llcs = np.array([r["true_llc"] for r in results])
    est_llcs = np.array([r["est_llc"] for r in results])
    params = np.array([r["params"] for r in results])

    fig, ax = plt.subplots(figsize=(8, 8))

    scatter = ax.scatter(
        true_llcs, est_llcs,
        c=np.log10(params),
        cmap='magma',
        alpha=0.7,
        s=50,
        edgecolors='white',
        linewidth=0.5
    )

    min_val = min(true_llcs.min(), est_llcs.min()) * 0.8
    max_val = max(true_llcs.max(), est_llcs.max()) * 1.2
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    ax.set_xlabel(r'True Learning Coefficient $\lambda_{SLA}$', fontsize=12)
    ax.set_ylabel(r'Estimated LLC $\hat{\lambda}(w^*)$', fontsize=12)
    ax.set_title('SLA Model: LLC Estimation at Global Minimum', fontsize=12)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Log10 Parameter Count', fontsize=10)

    relative_errors = np.abs(est_llcs - true_llcs) / true_llcs * 100
    textstr = f'N={len(results)}\nMean Error: {np.mean(relative_errors):.1f}%\nMedian Error: {np.median(relative_errors):.1f}%'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"保存: {save_path}")

    plt.show()


def print_summary(results: List[dict]):
    """結果のサマリーを表示"""
    if not results:
        return

    print("\n" + "=" * 60)
    print("SLA 実験結果サマリー")
    print("=" * 60)

    scales = sorted(set(r["scale"] for r in results),
                    key=lambda x: ["small", "medium", "large"].index(x) if x in ["small", "medium", "large"] else 99)

    for scale in scales:
        scale_results = [r for r in results if r["scale"] == scale]
        if scale_results:
            errors = [abs(r["est_llc"] - r["true_llc"]) / r["true_llc"] * 100
                      for r in scale_results]
            params = [r["params"] for r in scale_results]
            print(f"\n{scale}:")
            print(f"  試行数: {len(scale_results)}")
            print(f"  パラメータ数: {min(params):,} ~ {max(params):,}")
            print(f"  平均誤差: {np.mean(errors):.1f}%")
            print(f"  中央値誤差: {np.median(errors):.1f}%")
            print(f"  最大誤差: {np.max(errors):.1f}%")

    all_errors = [abs(r["est_llc"] - r["true_llc"]) / r["true_llc"] * 100 for r in results]
    print(f"\n全体:")
    print(f"  総試行数: {len(results)}")
    print(f"  平均誤差: {np.mean(all_errors):.1f}%")
    print(f"  中央値誤差: {np.median(all_errors):.1f}%")
    print("=" * 60)


# ============================================================
# 8. メイン実行
# ============================================================

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用デバイス: {device}")

    results = run_all_experiments(configs=CONFIGS, device=device)

    print_summary(results)

    plot_results(results, save_path="figures/sla_llc_logplot.png")