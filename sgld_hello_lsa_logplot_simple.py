# SGLD による LLC 推定 - LSA シンプル版 (α 制御なし、λ_full のみ比較)

import torch
from lsa_simple import run_all_experiments, print_summary, plot_results
from lsa_common import CONFIGS

DL_RATIO_RANGE = (0.5, 1.5)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用デバイス: {device}")
    print(f"dl_ratio_range: {DL_RATIO_RANGE}")

    results = run_all_experiments(
        configs=CONFIGS,
        device=device,
        dl_ratio_range=DL_RATIO_RANGE,
    )

    print_summary(results)

    plot_results(results, save_path="figures/lsa_llc_comparison_simple.png", dl_ratio_range=DL_RATIO_RANGE)
