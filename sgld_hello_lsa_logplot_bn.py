# SGLD による LLC 推定 - LSA Log-Log Plot 版 (ボトルネック制約あり)
# d_l の範囲: r ≤ d_l ≤ 2d - r

import torch
from lsa_alpha import run_all_experiments, print_summary, plot_results
from lsa_common import CONFIGS

DL_RATIO_RANGE = (0.5, 1.5)
ALPHA_LIST = [0, 1, 2]
USE_BOTTLENECK = True

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用デバイス: {device}")
    print(f"dl_ratio_range: {DL_RATIO_RANGE}")
    print(f"alpha_list: {ALPHA_LIST}")

    results = run_all_experiments(
        configs=CONFIGS,
        device=device,
        dl_ratio_range=DL_RATIO_RANGE,
        alpha_list=ALPHA_LIST,
        use_bottleneck=USE_BOTTLENECK
    )

    print_summary(results)

    plot_results(results, save_path="figures/lsa_llc_comparison_dl_bn.png", dl_ratio_range=DL_RATIO_RANGE, alpha_list=ALPHA_LIST)
