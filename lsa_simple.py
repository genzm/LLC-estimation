# LSA シンプル実験モジュール (α 制御なし、λ_full のみ比較)
# generate_random_simple_config, run_single_experiment,
# run_all_experiments, plot_results, print_summary を提供

from typing import List, Tuple, Optional
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from lsa_common import (
    compute_llc_theoretical_dln,
    generate_icl_batch,
    estimate_llc_with_sgld,
    create_lsa_rank_only,
    CONFIGS,
)


# ============================================================
# 1. ランダム構成生成 (rank(B) のみ)
# ============================================================

def generate_random_simple_config(
    d_low: int = 5,
    d_high: int = 15,
    dl_ratio_range: Tuple[float, float] = (0.5, 1.5),
    max_retries: int = 100,
) -> Optional[Tuple[int, int, int]]:
    """
    rank(B) のみ指定するシンプルな LSA 構成を生成

    決定順序:
        1. d を決める
        2. d_l を決める (d * ratio)
        3. rank を決める

    制約:
        - 1 ≤ rank ≤ min(d_l, d)  (フルランク回避 + d_l 以下)

    Returns:
        (d, d_l, rank) or None
    """
    for _ in range(max_retries):
        d = np.random.randint(d_low, d_high + 1)

        dl_ratio = np.random.uniform(dl_ratio_range[0], dl_ratio_range[1])
        d_l = max(1, int(d * dl_ratio))

        rank_max = min(d_l, d)
        if rank_max < 1:
            continue

        rank = np.random.randint(1, rank_max + 1)
        return d, d_l, rank

    return None


# ============================================================
# 2. 単一実験
# ============================================================

def run_single_experiment(
    d_range: Tuple[int, int],
    dl_ratio_range: Tuple[float, float] = (0.5, 1.5),
    N: int = 100,
    lr: float = 1e-5,
    elasticity: float = 1.0,
    num_steps: int = 5000,
    num_data: int = 5000,
    batch_size: int = 500,
    noise_std: float = 1.0,
    burn_in_ratio: float = 0.9,
    device: str = 'cpu',
) -> Optional[dict]:
    """
    単一の LSA 実験を実行 (rank(B) のみ指定、λ_full のみ計算)

    Returns:
        dict: {d, d_l, rank_B, lambda_full, est_llc, std_error, params, ...} or None
    """
    try:
        config = generate_random_simple_config(
            d_low=d_range[0], d_high=d_range[1],
            dl_ratio_range=dl_ratio_range,
        )
        if config is None:
            return None

        d, d_l, rank = config

        # Teacher モデル生成
        teacher_model, rank_B = create_lsa_rank_only(d, d_l, rank, device=device)

        # 理論値: λ_full ((d+1)×(d+1) 全体を DLN とみなす)
        H_full = [d + 1, d_l, d + 1]
        lambda_full = compute_llc_theoretical_dln(H_full, rank_B)

        if lambda_full == 0:
            return None

        param_count = sum(p.numel() for p in teacher_model.parameters())

        # データ生成
        x_data, _ = generate_icl_batch(num_data, d, N, device=device)

        # SGLD による LLC 推定
        est_llc, std_error = estimate_llc_with_sgld(
            teacher_model, x_data,
            num_steps=num_steps, batch_size=batch_size, lr=lr,
            elasticity=elasticity, noise_std=noise_std, burn_in_ratio=burn_in_ratio,
        )
        if est_llc is None:
            return None

        return {
            "params": param_count,
            "lambda_full": lambda_full,
            "est_llc": est_llc,
            "std_error": std_error,
            "d": d,
            "d_l": d_l,
            "rank_target": rank,
            "rank_B": rank_B,
        }

    except Exception as e:
        print(f"  実験失敗: {e}")
        return None


# ============================================================
# 3. 実験ランナー
# ============================================================

def run_all_experiments(
    configs=CONFIGS,
    device='cpu',
    dl_ratio_range=(0.5, 1.5),
):
    """
    全規模の実験を実行 (シンプル版: rank(B) のみ、λ_full のみ)
    """
    results = []

    print(f"\n{'#'*60}")
    print(f"# シンプル版実験 (rank(B) のみ、λ_full のみ)")
    print(f"# dl_ratio_range: {dl_ratio_range}")
    print(f"{'#'*60}")

    for name, d_range, N, lr, num_steps, num_data, num_trials in configs:
        print(f"\n{'='*50}")
        print(f"規模: {name} (d={d_range}, N={N}, n={num_data})")
        print(f"{'='*50}")

        for _ in tqdm(range(num_trials), desc=f"{name}"):
            result = run_single_experiment(
                d_range=d_range,
                dl_ratio_range=dl_ratio_range,
                N=N,
                lr=lr,
                num_steps=num_steps,
                num_data=num_data,
                device=device,
            )
            if result is not None:
                result["scale"] = name
                results.append(result)

        # 途中経過
        scale_results = [r for r in results if r["scale"] == name]
        if scale_results:
            errors = [abs(r["est_llc"] - r["lambda_full"]) / r["lambda_full"] * 100
                      for r in scale_results]
            print(f"  成功: {len(scale_results)}/{num_trials}, "
                  f"平均誤差(λ_full): {np.mean(errors):.1f}%")

    return results


# ============================================================
# 4. 可視化
# ============================================================

def plot_results(results: List[dict], save_path: str = None, dl_ratio_range: Tuple[float, float] = None):
    """Log-Log プロットを作成 (1列: λ_full vs 推定値のみ)"""
    if not results:
        print("結果がありません")
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    title = 'LSA LLC Estimation (Rank Only, λ_full)'
    if dl_ratio_range:
        title += f'\ndl_ratio_range = {dl_ratio_range}'
    fig.suptitle(title, fontsize=14, y=0.98)

    lambda_full = np.array([r["lambda_full"] for r in results])
    est_llcs = np.array([r["est_llc"] for r in results])
    params = np.array([r["params"] for r in results])
    log_params = np.log10(params)

    sc = ax.scatter(lambda_full, est_llcs, c=log_params, cmap='magma',
                    alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

    plot_min, plot_max = 1e2, 1e3
    ax.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', alpha=0.5, label='y=x')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)

    ax.set_xlabel(r'$\lambda_{full}$ [d+1, d_l, d+1]', fontsize=11)
    ax.set_ylabel(r'Estimated LLC $\hat{\lambda}(w^*)$', fontsize=11)
    ax.set_title('λ_full vs Estimated LLC', fontsize=11)

    errors = np.abs(est_llcs - lambda_full) / lambda_full * 100
    textstr = f'N={len(results)}\nMean: {np.mean(errors):.1f}%\nMedian: {np.median(errors):.1f}%'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Log10 Params', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"保存: {save_path}")

    plt.show()


# ============================================================
# 5. サマリー表示
# ============================================================

def print_summary(results: List[dict]):
    """結果のサマリーを表示 (λ_full との誤差のみ)"""
    if not results:
        return

    print("\n" + "=" * 70)
    print("LSA シンプル実験結果サマリー (λ_full のみ)")
    print("=" * 70)

    errors_full = [abs(r["est_llc"] - r["lambda_full"]) / r["lambda_full"] * 100 for r in results]

    print(f"\n全体: {len(results)} 試行")
    print(f"\n{'理論値':<35} {'平均誤差':<12} {'中央値誤差':<12}")
    print("-" * 60)
    print(f"{'λ_full [d+1, d_l, d+1]':<35} {np.mean(errors_full):>8.1f}%    {np.median(errors_full):>8.1f}%")

    # 次元・ランク統計
    d_list = [r["d"] for r in results]
    d_l_list = [r["d_l"] for r in results]
    rank_B_list = [r["rank_B"] for r in results]

    print(f"\n--- 次元・ランク統計 ---")
    print(f"d:                     平均 {np.mean(d_list):.1f}, 範囲 [{min(d_list)}, {max(d_list)}]")
    print(f"d_l:                   平均 {np.mean(d_l_list):.1f}, 範囲 [{min(d_l_list)}, {max(d_l_list)}]")
    print(f"rank(B) [(d+1)×(d+1)]: 平均 {np.mean(rank_B_list):.1f}, 範囲 [{min(rank_B_list)}, {max(rank_B_list)}]")
    print("=" * 70)
