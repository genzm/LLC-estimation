# SGLD による LLC (Local Learning Coefficient) 推定 - Log-Log Plot 版
# 論文: Lau et al. (2023) の Algorithm 1 に準拠
# 理論値: Aoyagi (2024) の Theorem 1
# Figure 3 (Left) の再現: 複数規模のモデルで理論値 vs 推定値をプロット

from typing import Literal, Union, List, Tuple, Optional, Set
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# 1. モデル生成プロセス (Lau et al. 2023, Appendix J.1)
# ============================================================

def generate_random_dln_config(
    L_low: int = 2,
    L_high: int = 5,
    H_low: int = 3,
    H_high: int = 10
) -> List[int]:
    """
    ランダムな DLN 構成 (層の幅) を生成

    Returns:
        H: 各層の幅のリスト [H_0, H_1, ..., H_L]
    """
    L = np.random.randint(L_low, L_high + 1)
    H = [np.random.randint(H_low, H_high + 1) for _ in range(L + 1)]
    return H


def create_random_rank_dln(H: List[int], device='cpu') -> Tuple[nn.Module, int]:
    """
    論文 Appendix J.1 に準拠した真のモデル生成関数。
    各層ごとに確率0.5でランダムなランク制約を適用する。

    Args:
        H: 各層の幅のリスト [H_0, H_1, ..., H_L]
        device: デバイス

    Returns:
        model: 生成された DLN
        effective_rank: ネットワーク全体の積のランク (理論値計算用)
    """
    L = len(H) - 1

    class LayerwiseRankDLN(nn.Module):
        def __init__(self, widths: List[int]):
            super().__init__()
            self.widths = widths
            self.L = len(widths) - 1
            self.weights = nn.ParameterList()

            for s in range(self.L):
                dim_in, dim_out = widths[s], widths[s+1]
                # Xavier 初期化
                std = np.sqrt(2.0 / (dim_in + dim_out))
                W_data = torch.randn(dim_out, dim_in) * std

                # 確率 0.5 でランク制約を適用
                if np.random.rand() < 0.5:
                    max_rank = min(dim_in, dim_out)
                    target_rank = np.random.randint(0, max_rank + 1)

                    if target_rank == 0:
                        W_data = torch.zeros(dim_out, dim_in)
                    elif target_rank < max_rank:
                        U, S, Vh = torch.linalg.svd(W_data, full_matrices=False)
                        S[target_rank:] = 0
                        W_data = U @ torch.diag(S) @ Vh

                W = nn.Parameter(W_data)
                self.weights.append(W)

        def forward(self, x):
            out = x
            for W in self.weights:
                out = out @ W.T
            return out

        def get_product_matrix(self):
            product = self.weights[0].data.clone()
            for s in range(1, self.L):
                product = self.weights[s].data @ product
            return product

    model = LayerwiseRankDLN(H).to(device)

    # 実効ランクの計測
    with torch.no_grad():
        product = model.get_product_matrix()

        # 数値精度を考慮したランク計算
        U, S, Vh = torch.linalg.svd(product, full_matrices=False)
        max_s = S[0].item() if S.numel() > 0 else 0
        if max_s == 0:
            effective_rank = 0
        else:
            tol = max_s * max(product.shape) * torch.finfo(product.dtype).eps
            effective_rank = (S > tol).sum().item()

    return model, effective_rank


# 旧実装 (Global Rank Constraint) - 参考用にコメントアウト
# def create_dln_with_rank(H: List[int], r: int) -> nn.Module:
#     """指定されたランクを持つ DLN を作成 (旧実装)"""
#     ...


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


def compute_llc_theoretical(H: List[int], r: int) -> float:
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


# ============================================================
# 3. SGLD オプティマイザ
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
# 4. 単一実験の関数化
# ============================================================

def run_single_experiment(
    L_range: Tuple[int, int],
    H_range: Tuple[int, int],
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
    単一の DLN 実験を実行 (論文 Appendix J.1 準拠)

    Returns:
        dict: {params, true_llc, est_llc, std_error, H, r} or None (失敗時)
    """
    try:
        # --- モデル生成 (Layer-wise Rank Constraint) ---
        H = generate_random_dln_config(
            L_low=L_range[0], L_high=L_range[1],
            H_low=H_range[0], H_high=H_range[1]
        )

        # Teacher モデル生成 & 実効ランクの計測
        teacher_model, r = create_random_rank_dln(H, device=device)

        # 理論値計算 (計測したランクを使う)
        llc_true = compute_llc_theoretical(H, r)

        # r=0 かつ llc_true=0 の場合はスキップ（相対誤差が定義できない）
        if llc_true == 0:
            return None

        # Student/Initial モデル構築 (Teacher と同じ構造・パラメータで初期化)
        student_model, _ = create_random_rank_dln(H, device=device)
        student_model.load_state_dict(teacher_model.state_dict())
        initial_model, _ = create_random_rank_dln(H, device=device)
        initial_model.load_state_dict(teacher_model.state_dict())
        initial_model.eval()

        # パラメータ数
        param_count = sum(p.numel() for p in student_model.parameters())

        # データ生成
        x_data = torch.randn(num_data, H[0]).to(device)  # N(0, 1)
        # x_data = torch.rand(num_data, H[0]).to(device) * 20 - 10  # U(-10, 10) #論文通りならこっち
        with torch.no_grad():
            y_true = teacher_model(x_data)
            y_data = y_true + torch.randn_like(y_true) * noise_std  # ノイズあり
            # y_data = y_true  # ノイズなし

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

        criterion = nn.MSELoss(reduction='sum')
        nll_scale = 1.0 / (2.0 * noise_std**2)

        energy_diff_trace = []
        student_model.train()

        # SGLD ループ
        for step in range(num_steps):
            indices = torch.randperm(num_data, device=device)[:batch_size]
            x_batch = x_data[indices]
            y_batch = y_data[indices]

            optimizer.zero_grad()

            y_pred = student_model(x_batch)
            batch_loss = criterion(y_pred, y_batch)
            # batch_loss = ((y_pred - y_batch) ** 2).mean(dim=-1).sum()
            energy_current = batch_loss * nll_scale * (num_data / batch_size)

            (beta * batch_loss * nll_scale).backward()
            optimizer.step()

            with torch.no_grad():
                y_pred_star = initial_model(x_batch)
                batch_loss_star = criterion(y_pred_star, y_batch)
                # batch_loss_star = ((y_pred_star - y_batch) ** 2).mean(dim=-1).sum()
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
            "H": H,
            "r": r
        }

    except Exception as e:
        print(f"  実験失敗: {e}")
        return None


# ============================================================
# 5. 実験設定 (Table 1 に準拠)
# ============================================================

# (名前, L範囲, H範囲, ステップサイズ, ステップ数, データ数, 試行回数)
CONFIGS = [
    ("1k",   (2, 5),   (5, 50),      5e-7, 10000, 100000,   20),
    ("10k",  (2, 10),  (5, 100),     5e-7, 10000, 100000,   20),
    # ("100k", (2, 10),  (50, 500),    1e-7, 50000, 1000000,  20),
    # ("1M",   (5, 20),  (100, 1000),  5e-8, 50000, 1000000,  10),
    # ("10M",  (2, 20),  (500, 2000),  2e-8, 50000, 1000000,  5),
    # ("100M", (2, 40),  (500, 3000),  2e-8, 50000, 1000000,  3),
]


def run_all_experiments(configs=CONFIGS, device='cpu'):
    """
    全規模の実験を実行
    """
    results = []

    for name, L_range, H_range, lr, num_steps, num_data, num_trials in configs:
        print(f"\n{'='*50}")
        print(f"規模: {name} (L={L_range}, H={H_range}, n={num_data})")
        print(f"{'='*50}")

        for trial in tqdm(range(num_trials), desc=f"{name}"):
            result = run_single_experiment(
                L_range=L_range,
                H_range=H_range,
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
    """
    Log-Log プロットを作成 (Figure 3 Left 再現)
    """
    if not results:
        print("結果がありません")
        return

    # データ抽出
    true_llcs = np.array([r["true_llc"] for r in results])
    est_llcs = np.array([r["est_llc"] for r in results])
    params = np.array([r["params"] for r in results])

    # プロット
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

    # 対角線
    min_val = min(true_llcs.min(), est_llcs.min()) * 0.8
    max_val = max(true_llcs.max(), est_llcs.max()) * 1.2
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')

    # 軸設定
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    ax.set_xlabel(r'True Learning Coefficient $\lambda$', fontsize=12)
    ax.set_ylabel(r'Estimated LLC $\hat{\lambda}(w^*)$', fontsize=12)
    ax.set_title('Figure 3 (Left) Reproduction: LLC Estimation at Global Minimum', fontsize=12)

    # カラーバー
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Log10 Parameter Count', fontsize=10)

    # 統計情報
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
    """
    結果のサマリーを表示
    """
    if not results:
        return

    print("\n" + "=" * 60)
    print("実験結果サマリー")
    print("=" * 60)

    # 規模ごとの統計
    scale_order = ["1k", "10k", "100k", "1M", "10M", "100M"]
    scales = sorted(set(r["scale"] for r in results),
                    key=lambda x: scale_order.index(x) if x in scale_order else 99)

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

    # 全体統計
    all_errors = [abs(r["est_llc"] - r["true_llc"]) / r["true_llc"] * 100 for r in results]
    print(f"\n全体:")
    print(f"  総試行数: {len(results)}")
    print(f"  平均誤差: {np.mean(all_errors):.1f}%")
    print(f"  中央値誤差: {np.median(all_errors):.1f}%")
    print("=" * 60)


# ============================================================
# 6. メイン実行
# ============================================================

if __name__ == "__main__":
    # デバイス設定
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用デバイス: {device}")

    # 実験実行
    results = run_all_experiments(configs=CONFIGS, device=device)

    # サマリー表示
    print_summary(results)

    # プロット
    plot_results(results, save_path="figures/llc_logplot.png")
