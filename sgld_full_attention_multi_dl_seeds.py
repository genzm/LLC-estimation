# Full Attention 軌跡 + 複数 d_l × 複数シード比較実験
# 単一 trajectory の実行・可視化、および複数 d_l × seed の比較

import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from lsa_common import (
    generate_icl_batch,
    compute_llc_theoretical_dln,
    estimate_llc_with_sgld,
)
from full_attention import (
    FullAttention,
    train_full_attention_with_checkpoints,
    extract_and_build_lsa,
)


def run_trajectory_with_llc(
    d: int,
    d_l: int,
    N: int = 20,
    num_tasks: int = 10000,
    epochs: int = 200,
    checkpoint_epochs: list = None,
    rtol: float = 1e-3,
    num_data_llc: int = 5000,
    num_steps_llc: int = 4000,
    lr_llc: float = 1e-5,
    noise_std_llc: float = 1.0,
    num_trials: int = 1,
    device: str = 'cpu',
    verbose: bool = True
) -> dict:
    """
    各エポックでのランク、理論値、LLC推定値の推移を追跡

    Args:
        num_trials: 各チェックポイントで LLC 推定を行う回数。
                    複数回行う場合、平均・標準偏差を記録する。

    Returns:
        dict: {
            'd', 'd_l', 'num_trials', 'loss_history',
            'trajectory': [
                {'epoch', 'rank_B', 'lambda_B',
                 'est_llc', 'est_llc_std', 'est_llc_all',
                 'error_B', 'error_B_std'},
                ...
            ]
        }
    """
    input_dim = d + 1

    if checkpoint_epochs is None:
        checkpoint_epochs = [0, 10, 20, 50, 100, epochs]
        checkpoint_epochs = [e for e in checkpoint_epochs if e <= epochs]

    print(f"\n{'='*70}")
    print(f"軌跡実験 (LLC推定付き): d={d}, d_l={d_l}, epochs={epochs}")
    print(f"チェックポイント: {checkpoint_epochs}")
    print(f"num_trials: {num_trials}")
    print(f"{'='*70}")

    # 訓練
    model, loss_history, checkpoints = train_full_attention_with_checkpoints(
        d=d, d_l=d_l, N=N, num_tasks=num_tasks, epochs=epochs,
        checkpoint_epochs=checkpoint_epochs, device=device, verbose=verbose
    )

    # LLC推定用データ（全チェックポイントで共通）
    x_data, _ = generate_icl_batch(num_data_llc, d, N, device=device)

    # 各チェックポイントで解析
    trajectory = []
    temp_model = FullAttention(input_dim, d_l).to(device)

    header = f"{'Epoch':<8} {'rank_B':<8} {'lambda_B':<12} {'est_LLC':<16} {'err_B%':<10}"
    if num_trials > 1:
        header = f"{'Epoch':<8} {'rank_B':<8} {'lambda_B':<12} {'est_LLC (mean±std)':<22} {'err_B% (mean±std)':<20}"
    print(f"\n{header}")
    print("-" * len(header))

    for epoch, state_dict in checkpoints:
        temp_model.load_state_dict(state_dict)

        # LSAモデル構築とランク解析
        lsa_model, analysis = extract_and_build_lsa(
            temp_model, d, d_l, rtol=rtol, device=device, verbose=False
        )
        rank_B = analysis['rank_B']

        # 理論値計算
        lambda_B = compute_llc_theoretical_dln([d+1, d_l, d+1], rank_B)

        # LLC推定 (num_trials 回)
        llc_estimates = []

        if rank_B > 0:
            for trial in range(num_trials):
                est, _ = estimate_llc_with_sgld(
                    lsa_model, x_data,
                    num_steps=num_steps_llc,
                    lr=lr_llc,
                    noise_std=noise_std_llc,
                )
                if est is not None:
                    llc_estimates.append(est)

        # 統計量の計算
        if llc_estimates:
            est_llc = float(np.mean(llc_estimates))
            est_llc_std = float(np.std(llc_estimates)) if len(llc_estimates) > 1 else 0.0

            if lambda_B > 0:
                errors = [abs(e - lambda_B) / lambda_B * 100 for e in llc_estimates]
                error_B = float(np.mean(errors))
                error_B_std = float(np.std(errors)) if len(errors) > 1 else 0.0
            else:
                error_B = None
                error_B_std = None
        else:
            est_llc = None
            est_llc_std = None
            error_B = None
            error_B_std = None

        # 結果表示
        if num_trials > 1:
            if est_llc is not None:
                est_str = f"{est_llc:.2f}±{est_llc_std:.2f}"
            else:
                est_str = "N/A"
            if error_B is not None:
                err_str = f"{error_B:.1f}±{error_B_std:.1f}"
            else:
                err_str = "N/A"
            print(f"{epoch:<8} {rank_B:<8} {lambda_B:<12.2f} {est_str:<22} {err_str:<20}")
        else:
            est_str = f"{est_llc:.2f}" if est_llc is not None else "N/A"
            err_str = f"{error_B:.1f}" if error_B is not None else "N/A"
            print(f"{epoch:<8} {rank_B:<8} {lambda_B:<12.2f} {est_str:<16} {err_str:<10}")

        trajectory.append({
            'epoch': epoch,
            'rank_B': rank_B,
            'lambda_B': lambda_B,
            'singular_values_B': analysis['singular_values_B'].tolist(),
            'est_llc': est_llc,
            'est_llc_std': est_llc_std,
            'est_llc_all': llc_estimates,
            'error_B': error_B,
            'error_B_std': error_B_std,
        })

    return {
        'd': d,
        'd_l': d_l,
        'epochs': epochs,
        'rtol': rtol,
        'num_trials': num_trials,
        'loss_history': loss_history,
        'trajectory': trajectory
    }


def plot_trajectory_with_llc(result: dict, save_path: str = None):
    """軌跡をプロット（ランク、理論値、LLC推定値、エラーバー付き）"""
    trajectory = result['trajectory']
    d, d_l = result['d'], result['d_l']
    num_trials = result.get('num_trials', 1)

    epochs = [t['epoch'] for t in trajectory]
    rank_B = [t['rank_B'] for t in trajectory]
    lambda_B = [t['lambda_B'] for t in trajectory]
    est_llc = [t['est_llc'] for t in trajectory]
    est_llc_std = [t.get('est_llc_std', 0) for t in trajectory]

    # magma 系カラーパレット
    C_DARK = '#3B0F70'
    C_MID = '#8C2981'
    C_PINK = '#DE4968'
    C_ORANGE = '#FB8861'
    C_LIGHT = '#FCFDBF'
    MKR = dict(markersize=7, markeredgecolor='white', markeredgewidth=0.8)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 左上: 有効ランク推移
    ax1 = axes[0, 0]
    ax1.plot(epochs, rank_B, '-o', color=C_PINK, label='rank(B)', **MKR)
    ax1.axhline(y=d_l, color=C_MID, linestyle='--', alpha=0.6,
                label=rf'$d_l={d_l}$')
    ax1.axhline(y=d+1, color=C_ORANGE, linestyle='--', alpha=0.6,
                label=rf'$d+1={d+1}$')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Effective Rank', fontsize=11)
    ax1.set_title('Effective Rank Evolution', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 右上: 学習曲線
    ax2 = axes[0, 1]
    loss_history = result['loss_history']
    loss_epochs = list(range(1, len(loss_history) + 1))
    ax2.plot(loss_epochs, loss_history, '-', color=C_DARK, linewidth=1.2)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('MSE Loss', fontsize=11)
    ax2.set_title('Training Loss (Full Attention)', fontsize=12)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # 左下: LLC推定値 vs 理論値（推移、エラーバー付き）
    ax3 = axes[1, 0]
    valid_idx = [i for i, e in enumerate(est_llc) if e is not None]
    valid_epochs = [epochs[i] for i in valid_idx]
    valid_est = [est_llc[i] for i in valid_idx]
    valid_std = [est_llc_std[i] or 0 for i in valid_idx]
    valid_lambda_B = [lambda_B[i] for i in valid_idx]

    if num_trials > 1:
        ax3.errorbar(valid_epochs, valid_est, yerr=valid_std,
                     fmt='-o', color=C_ORANGE, label=r'Estimated $\hat{\lambda}$ (mean±std)',
                     markeredgecolor='white', markeredgewidth=0.8,
                     markersize=7, linewidth=2, capsize=4)
    else:
        ax3.plot(valid_epochs, valid_est, '-o', color=C_ORANGE,
                 label=r'Estimated $\hat{\lambda}$', linewidth=2, **MKR)
    ax3.plot(valid_epochs, valid_lambda_B, '--^', color=C_MID,
             label=r'$\lambda_B$ (theory)', markersize=6, alpha=0.8,
             markeredgecolor='white', markeredgewidth=0.8)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('LLC', fontsize=11)
    ax3.set_title('LLC: Estimation vs Theory', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # 統計テキストボックス
    if valid_est and valid_lambda_B:
        final_est = valid_est[-1]
        final_theory = valid_lambda_B[-1]
        final_err = abs(final_est - final_theory) / final_theory * 100 if final_theory > 0 else 0
        textstr = f'Final: {final_est:.1f} / {final_theory:.1f}\nError: {final_err:.1f}%'
        ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=9,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor=C_LIGHT, alpha=0.7))

    # 右下: 特異値推移
    ax4 = axes[1, 1]
    sv_data = [t.get('singular_values_B') for t in trajectory]
    if sv_data and sv_data[0] is not None:
        sv_matrix = np.array(sv_data)
        n_sv = min(d + 1, sv_matrix.shape[1])
        sv_colors = plt.cm.magma(np.linspace(0.15, 0.85, n_sv))
        for i in range(n_sv):
            ax4.plot(epochs, sv_matrix[:, i], '-o', color=sv_colors[i],
                     label=rf'$\sigma_{{{i+1}}}$', markersize=4,
                     markeredgecolor='white', markeredgewidth=0.5)
        ax4.set_yscale('log')
        ax4.legend(fontsize=7, ncol=2, loc='upper right')
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Singular Value', fontsize=11)
    ax4.set_title('Singular Values of B', fontsize=12)
    ax4.grid(True, alpha=0.3)

    title = f'd={d}, d_l={d_l}'
    if num_trials > 1:
        title += f', num_trials={num_trials}'
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def run_multi_dl_seeds(
    d: int,
    d_l_values: list,
    num_seeds: int = 3,
    N: int = 20,
    num_tasks: int = 10000,
    epochs: int = 200,
    checkpoint_epochs: list = None,
    rtol: float = 1e-3,
    num_data_llc: int = 5000,
    num_steps_llc: int = 4000,
    lr_llc: float = 1e-5,
    noise_std_llc: float = 1.0,
    num_trials: int = 1,
    device: str = 'cpu',
) -> list:
    """
    複数 d_l × 複数シードで trajectory を実行

    Returns:
        list of dict: 各要素は {
            'd', 'd_l', 'num_seeds',
            'seed_results': [run_trajectory_with_llc の返り値, ...],
        }
    """
    results = []
    for d_l in d_l_values:
        seed_results = []
        for seed in range(num_seeds):
            print(f"\n>>> d_l={d_l}, seed={seed+1}/{num_seeds} "
                  f"(d+1={d+1}, "
                  f"{'rank change expected' if d_l > d+1 else 'no rank change'})")
            r = run_trajectory_with_llc(
                d=d, d_l=d_l, N=N, num_tasks=num_tasks, epochs=epochs,
                checkpoint_epochs=checkpoint_epochs, rtol=rtol,
                num_data_llc=num_data_llc,
                num_steps_llc=num_steps_llc,
                lr_llc=lr_llc,
                noise_std_llc=noise_std_llc,
                num_trials=num_trials,
                device=device, verbose=False
            )
            seed_results.append(r)
        results.append({
            'd': d,
            'd_l': d_l,
            'num_seeds': num_seeds,
            'seed_results': seed_results,
        })
    return results


def plot_multi_dl_seeds(results: list, save_path: str = None):
    """複数 d_l × 複数シードの trajectory を比較プロット（個別表示）"""
    if not results:
        return

    d = results[0]['d']
    num_seeds = results[0]['num_seeds']

    has_llc = any(
        any(t.get('est_llc') is not None
            for sr in r['seed_results'] for t in sr['trajectory'])
        for r in results
    )

    n = len(results)
    colors = plt.cm.magma(np.linspace(0.2, 0.85, n))
    MKR = dict(markersize=3, markeredgecolor='white', markeredgewidth=0.3)

    if has_llc:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) Rank 推移
    ax1 = axes[0, 0]
    offsets = np.linspace(-0.06, 0.06, n)
    for i, r in enumerate(results):
        d_l = r['d_l']
        for s, sr in enumerate(r['seed_results']):
            epochs = [t['epoch'] for t in sr['trajectory']]
            rank_B = [t['rank_B'] + offsets[i] for t in sr['trajectory']]
            label = rf'$d_l={d_l}$' if s == 0 else None
            ax1.plot(epochs, rank_B, '-o', color=colors[i],
                     alpha=0.6, linewidth=1, label=label, **MKR)
    ax1.axhline(y=d+1, color='gray', linestyle='--', alpha=0.6,
                label=rf'$d+1={d+1}$')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Effective Rank of B', fontsize=11)
    ax1.set_title('Rank Evolution', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # (0,1) rank_B 番目の特異値の推移
    ax2 = axes[0, 1]
    for i, r in enumerate(results):
        d_l = r['d_l']
        for s, sr in enumerate(r['seed_results']):
            traj = sr['trajectory']
            epochs = [t['epoch'] for t in traj]
            tail_sv = []
            for t in traj:
                sv = t.get('singular_values_B')
                rank = t['rank_B']
                if sv is not None and rank > 0:
                    tail_sv.append(sv[rank - 1])
                else:
                    tail_sv.append(np.nan)
            label = rf'$d_l={d_l}$' if s == 0 else None
            ax2.plot(epochs, tail_sv, '-s', color=colors[i],
                     alpha=0.6, linewidth=1, label=label, **MKR)
    ax2.set_yscale('log')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel(r'$\sigma_{r}$ (smallest nonzero SV)', fontsize=11)
    ax2.set_title(r'Smallest Nonzero Singular Value ($\sigma_{rank}$)',
                  fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # (1,0) 理論値 λ_B の推移
    ax3 = axes[1, 0]
    all_lambda = [t['lambda_B'] for r in results
                  for sr in r['seed_results'] for t in sr['trajectory']]
    lambda_range = (max(all_lambda) - min(all_lambda)
                    if len(set(all_lambda)) > 1
                    else max(all_lambda) * 0.01)
    offsets_lambda = np.linspace(-lambda_range * 0.012,
                                 lambda_range * 0.012, n)
    for i, r in enumerate(results):
        d_l = r['d_l']
        for s, sr in enumerate(r['seed_results']):
            epochs = [t['epoch'] for t in sr['trajectory']]
            lambda_B = [t['lambda_B'] + offsets_lambda[i]
                        for t in sr['trajectory']]
            label = rf'$d_l={d_l}$' if s == 0 else None
            ax3.plot(epochs, lambda_B, '-^', color=colors[i],
                     alpha=0.6, linewidth=1, label=label, **MKR)
    ax3.axhline(y=(d+1)**2 / 2, color='gray', linestyle='--', alpha=0.6,
                label=rf'$(d+1)^2/2={((d+1)**2)/2:.1f}$')
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel(r'Theoretical $\lambda_B$', fontsize=11)
    ax3.set_title(r'Theoretical LLC $\lambda_B$', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # (1,1) 学習曲線
    ax4 = axes[1, 1]
    for i, r in enumerate(results):
        d_l = r['d_l']
        for s, sr in enumerate(r['seed_results']):
            loss = sr['loss_history']
            loss_epochs = list(range(1, len(loss) + 1))
            label = rf'$d_l={d_l}$' if s == 0 else None
            ax4.plot(loss_epochs, loss, '-', color=colors[i],
                     alpha=0.5, linewidth=0.8, label=label)
    ax4.set_yscale('log')
    from matplotlib.ticker import ScalarFormatter
    ax4.yaxis.set_major_formatter(ScalarFormatter())
    ax4.yaxis.get_major_formatter().set_scientific(False)
    ax4.yaxis.get_major_formatter().set_useOffset(False)
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('MSE Loss', fontsize=11)
    ax4.set_title('Training Loss', fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # LLC 推定がある場合: 追加パネル
    if has_llc:
        # (0,2) LLC推定値の推移
        ax5 = axes[0, 2]
        for i, r in enumerate(results):
            d_l = r['d_l']
            for s, sr in enumerate(r['seed_results']):
                traj = sr['trajectory']
                ep = [t['epoch'] for t in traj if t.get('est_llc') is not None]
                est = [t['est_llc'] for t in traj
                       if t.get('est_llc') is not None]
                std = [t.get('est_llc_std', 0) or 0 for t in traj
                       if t.get('est_llc') is not None]
                if ep:
                    label = rf'$d_l={d_l}$' if s == 0 else None
                    ax5.errorbar(ep, est, yerr=std,
                                 fmt='-o', color=colors[i], alpha=0.5,
                                 capsize=2, linewidth=1, label=label, **MKR)
        ax5.axhline(y=(d+1)**2 / 2, color='gray', linestyle='--', alpha=0.6,
                    label=rf'$(d+1)^2/2={((d+1)**2)/2:.1f}$')
        ax5.set_xlabel('Epoch', fontsize=11)
        ax5.set_ylabel(r'Estimated $\hat{\lambda}$', fontsize=11)
        ax5.set_title('LLC Estimation', fontsize=12)
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)

        # (1,2) 最終 epoch の理論値 vs 推定値（全シード個別表示）
        ax6 = axes[1, 2]
        theory_all = []
        est_all = []
        for i, r in enumerate(results):
            d_l = r['d_l']
            for s, sr in enumerate(r['seed_results']):
                valid = [t for t in reversed(sr['trajectory'])
                         if t.get('est_llc') is not None]
                if valid:
                    tv = valid[0]['lambda_B']
                    ev = valid[0]['est_llc']
                    theory_all.append(tv)
                    est_all.append(ev)
                    label = rf'$d_l={d_l}$' if s == 0 else None
                    ax6.scatter(tv, ev, color=colors[i], s=60, alpha=0.7,
                                edgecolors='white', linewidth=0.5,
                                label=label)
        if theory_all and est_all:
            all_vals = theory_all + est_all
            vmin, vmax = min(all_vals) * 0.85, max(all_vals) * 1.15
            ax6.plot([vmin, vmax], [vmin, vmax], 'k--', alpha=0.5,
                     label='y=x')
            ax6.set_xlim(vmin, vmax)
            ax6.set_ylim(vmin, vmax)
        ax6.set_xlabel(r'Theoretical $\lambda_B$', fontsize=11)
        ax6.set_ylabel(r'Estimated $\hat{\lambda}$', fontsize=11)
        ax6.set_title('Theory vs Estimation (Final, each seed)', fontsize=12)
        ax6.legend(fontsize=8)
        ax6.set_aspect('equal', adjustable='box')
        ax6.grid(True, alpha=0.3)

    title = f'd={d}, multi-$d_l$ comparison ({num_seeds} seeds each)'
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    multi_results = run_multi_dl_seeds(
        d=8,
        d_l_values=[6, 9, 12, 16, 24],
        num_seeds=1,
        N=100,
        epochs=200,
        checkpoint_epochs=list(range(0, 201, 25)),
        num_data_llc=5000,
        num_steps_llc=4000,
        lr_llc=1e-5,
        num_trials=3,
        device=device
    )
    plot_multi_dl_seeds(multi_results,
                        save_path=f"figures/multi_dl_seeds_{timestamp}.png")
