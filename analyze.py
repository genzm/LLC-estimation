import matplotlib.pyplot as plt
import numpy as np

# --- Calculation Logic (No changes) ---
def get_lambda_data(d, dl, r):
    if r > d or r < 1 or dl < 1: return 0.0
    if dl >= 2 * d - r:
        return (d ** 2) / 2.0
    elif dl <= r:
        return (d * (dl + r) - dl * r) / 2.0
    else:
        term = dl + r
        base = 4 * d * term - (term ** 2)
        return base / 8.0 if term % 2 == 0 else (base + 1) / 8.0

def get_lambda_w(d, dl, r, alpha):
    if r + alpha > d + 1 or r < 1 or dl < 1: return 0.0
    upper = 2 * (d + 1) - (r + alpha)
    lower = r + alpha
    term = dl + r + alpha
    if dl >= upper:
        return ((d + 1) ** 2) / 2.0
    elif dl <= lower:
        return ((d + 1) * term - dl * (r + alpha)) / 2.0
    else:
        base = 4 * (d + 1) * term - (term ** 2)
        return base / 8.0 if term % 2 == 0 else (base + 1) / 8.0


def get_LSA_query(d, d_l, r1, r2):
    """
    LSA query part の理論値。

    λ_LSA_query = 2 * λ[d, d_l, 1, r1] + λ[1, d_l, 1, r2]

    Args:
        d: 入力次元
        d_l: 隠れ層次元
        r1: [d, d_l, 1] 用の rank (0, 1, ..., min(d, 1))
        r2: [1, d_l, 1] 用の rank (0 or 1)

    Returns:
        λ_LSA_query
    """
    # [d, d_l, 1] の2層 λ の2倍
    lambda_d_dl_1 = get_general_lambda(d, d_l, 1, r1)

    # [1, d_l, 1] の2層 λ
    lambda_1_dl_1 = get_general_lambda(1, d_l, 1, r2)

    return 2 * lambda_d_dl_1 + lambda_1_dl_1
# --------------------------------------

def plot_analysis_with_boundaries():
    # === Parameters ===
    d_fixed = 15
    r_samples = [1, 5, 10, 14]
    dl_max = 60
    alphas = [0, 1, 2]
    # ==================

    dl_range = list(range(1, dl_max + 1))

    num_plots = len(r_samples)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots), sharex=True)
    if num_plots == 1: axes = [axes]

    print(f"Generating plots for d={d_fixed} with boundary markers...")

    for i, r in enumerate(r_samples):
        ax = axes[i]

        # --- 境界線の計算 (Phase Boundaries for lambda_data) ---
        bound_linear = r                # Boundary 1: d_l = r
        bound_saturation = 2 * d_fixed - r # Boundary 2: d_l = 2d - r

        # 1. LSA Query Lines (commented out: r1, r2 fixed values unrelated to panel's r)
        # r2_fixed = 1
        # r1_list = [1]
        # lsa_colors = ['black', 'gray', 'brown']
        # lsa_styles = ['--', ':', '-.']
        # for k, r1 in enumerate(r1_list):
        #     la_values = [get_LSA_query(d_fixed, dl, r1, r2_fixed) for dl in dl_range]
        #     ax.plot(dl_range, la_values, color=lsa_colors[k], linestyle=lsa_styles[k],
        #             linewidth=1.5, alpha=0.7, label=f'LSA Query (r1={r1})')

        # 2. Phase Boundaries (Vertical Lines & Shading)
        # 色分け: 線形領域境界(Magenta), 飽和領域境界(Cyan)
        ax.axvline(x=bound_linear, color='magenta', linestyle='-.', linewidth=1.5, alpha=0.8, label=f'Bound 1 ($d_l=r={r}$)')
        ax.axvline(x=bound_saturation, color='cyan', linestyle='-.', linewidth=1.5, alpha=0.8, label=f'Bound 2 ($d_l=2d-r={bound_saturation}$)')
        
        # 領域の強調（オプション: 遷移領域を薄く塗る）
        # 線形領域と飽和領域の間にある "Transition Zone" を可視化
        if bound_linear < bound_saturation:
            ax.axvspan(bound_linear, bound_saturation, color='yellow', alpha=0.05, label='Transition Zone')

        # 2.5. SLA Theory line: λ_SLA = λ_matrix + (d + 0.5)
        ax.axhline(y=d_fixed + 0.5, color='orange', linestyle='-', linewidth=2,
                   alpha=0.8, label=f'SLA Theory ($d+0.5={d_fixed + 0.5}$)')

        # 2.6. LSA valid range: d_l >= r + alpha (lower bound per alpha)
        colors = ['blue', 'green', 'red']
        for j, alpha in enumerate(alphas):
            dl_lower = r + alpha
            ax.axvline(x=dl_lower, color=colors[j], linestyle=':', linewidth=1, alpha=0.5,
                       label=f'$d_l \\geq r+\\alpha={dl_lower}$ (α={alpha})')

        # 3. Diff Curves Plotting
        for j, alpha in enumerate(alphas):
            diff_values = []
            for dl in dl_range:
                ld = get_lambda_data(d_fixed, dl, r)
                lw = get_lambda_w(d_fixed, dl, r, alpha)
                diff_values.append(lw - ld)
            
            ax.plot(dl_range, diff_values, label=f'Diff (α={alpha})', color=colors[j], marker='o', markersize=4, linewidth=1.5, alpha=0.8)

        # Decorations
        ax.set_title(f"Condition: d={d_fixed}, r={r}")
        ax.set_ylabel("Value (Diff)")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # 凡例の位置調整
        ax.legend(loc='lower right', fontsize='small', framealpha=0.9)

    axes[-1].set_xlabel("$d_l$ (Sweep Parameter)")
    plt.tight_layout()
    # 画像を保存する場合
    plt.savefig("figures/lambda_analysis_plot.png", dpi=150)
    plt.show()


def get_general_lambda(h_in, h_hid, h_out, r):
    """
    Calculates the learning coefficient lambda for a 2-layer linear network
    (Input -> Hidden -> Output) based on Theorem 2 of Neural Networks 172 (2024).
    
    Refined to perfectly match the strict derivation for symmetric cases.
    """
    # Pre-check: Prior Assumption
    if r > min(h_in, h_out):
        return 0.0
    if r < 0 or h_hid < 1:
        return 0.0

    # --- Case A: Saturated Region (Hidden layer is very large) ---
    # Condition: H(1) + H(3) <= H(2) + r
    # Note: Using <= to match the boundary continuity
    if h_out + h_in <= h_hid + r:
        return (h_out * h_in) / 2.0

    # --- Case B: Input Dominant (Input is very large relative to others) ---
    # Condition: H(1) + H(2) <= H(3) + r
    elif h_out + h_hid <= h_in + r:
        return (h_hid * h_out - h_hid * r + h_in * r) / 2.0

    # --- Case C: Output Dominant (Output is very large relative to others) ---
    # Condition: H(2) + H(3) <= H(1) + r
    elif h_hid + h_in <= h_out + r:
        return (h_hid * h_in - h_hid * r + h_out * r) / 2.0

    # --- Case D: Intermediate Region (Balanced dimensions) ---
    else:
        # The formula derived from the interaction of all three layers.
        # This term ALREADY includes the base rank complexity.
        # Formula: [ -(H2+r)^2 - (H1-H3)^2 + 2(H2+r)(H1+H3) ] / 8
        
        # Calculation using sum of squares form for clarity:
        # Term 1: -(H_hid + r)^2
        # Term 2: -(H_out - H_in)^2
        # Term 3: + 2 * (H_hid + r) * (H_out + H_in)
        
        term1 = -(h_hid + r)**2
        term2 = -(h_out - h_in)**2
        term3 = 2 * (h_hid + r) * (h_out + h_in)
        
        numerator = term1 + term2 + term3
        lambda_val = numerator / 8.0

        # Parity Check (Odd/Even adjustment)
        # Condition: Check if sum of all dims + r is odd
        total_sum = h_out + h_hid + h_in + r
        if total_sum % 2 != 0:
            lambda_val += 0.125 # Add 1/8

        return lambda_val
    
# --- 検証用: あなたの元のコードと同じ条件(h_in = h_out = d)で一致するか確認 ---
def test_consistency():
    d = 50
    r = 10
    print(f"Testing Consistency (d={d}, r={r})...")
    print(f"{'Hidden(dl)':<12} | {'Original':<10} | {'General':<10} | {'Match'}")
    print("-" * 45)
    
    for dl in range(1, 150): # 隠れ層を変化させる
        # あなたの元のロジック
        val_original = 0.0
        # 再現 (元のコードロジック)
        if dl >= 2 * d - r:
            val_original = (d ** 2) / 2.0
        elif dl <= r:
            val_original = (d * (dl + r) - dl * r) / 2.0
        else:
            term = dl + r
            base = 4 * d * term - (term ** 2)
            val_original = base / 8.0 if term % 2 == 0 else (base + 1) / 8.0
            
        # 一般化ロジック
        val_general = get_general_lambda(d, dl, d, r)
        
        is_match = abs(val_original - val_general) < 1e-9
        if dl % 20 == 0 or not is_match: # 間引き出力
            print(f"{dl:<12} | {val_original:<10.2f} | {val_general:<10.2f} | {is_match}")

# test_consistency()

if __name__ == "__main__":
    plot_analysis_with_boundaries()