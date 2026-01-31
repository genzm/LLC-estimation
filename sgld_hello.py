# SGLD による LLC (Local Learning Coefficient) 推定
# 論文: Lau et al. (2023) の Algorithm 1 に準拠
# 理論値: Aoyagi (2024) の Theorem 1

from typing import Literal, Union, List, Tuple, Optional, Set
import torch
import torch.nn as nn
import numpy as np


# ============================================================
# 1. モデル生成プロセス (Lau et al. 2023, Appendix J)
# ============================================================

def generate_random_dln_config(
    L_low: int = 2,
    L_high: int = 5,
    H_low: int = 3,
    H_high: int = 10
) -> Tuple[List[int], int]:
    """
    ランダムな DLN 構成を生成

    Args:
        L_low, L_high: 層数の範囲 (L ~ U(L_low, L_high))
        H_low, H_high: 幅の範囲 (H^{(s)} ~ U(H_low, H_high))

    Returns:
        H: 各層の幅のリスト [H^{(1)}, H^{(2)}, ..., H^{(L+1)}]
        r: 真のランク
    """
    # 層数 L をサンプリング
    L = np.random.randint(L_low, L_high + 1)

    # 各層の幅をサンプリング (入力層 s=1 から出力層 s=L+1 まで)
    H = [np.random.randint(H_low, H_high + 1) for _ in range(L + 1)]

    # ランク r のサンプリング
    min_H = min(H)
    if np.random.rand() < 0.5:
        # 確率 0.5 で低ランク
        r = np.random.randint(0, min_H + 1)
    else:
        # 確率 0.5 でフルランク
        r = min_H

    return H, r


def create_dln_with_rank(H: List[int], r: int) -> nn.Module:
    """
    指定されたランクを持つ DLN を作成

    Args:
        H: 各層の幅 [H^{(1)}, ..., H^{(L+1)}]
        r: 真のランク

    Returns:
        DLN モデル（行列積のランクが r になるように修正済み）
    """
    L = len(H) - 1  # 層数

    class RankConstrainedDLN(nn.Module):
        def __init__(self, widths: List[int], rank: int):
            super().__init__()
            self.widths = widths
            self.rank = rank
            self.L = len(widths) - 1

            # 各層の重み行列を作成（Xavier初期化で安定化）, 元の論文ではrandn初期化
            self.weights = nn.ParameterList()
            for s in range(self.L):
                # Xavier初期化: std = sqrt(2 / (fan_in + fan_out))
                std = np.sqrt(2.0 / (widths[s] + widths[s+1]))
                W = nn.Parameter(torch.randn(widths[s+1], widths[s]) * std)
                self.weights.append(W)

            # ランク制約を適用
            self._apply_rank_constraint()

        def _apply_rank_constraint(self):
            """行列積のランクを r に強制"""
            if self.rank == 0:
                # ランク 0 の場合、すべての重みを 0 に
                for W in self.weights:
                    W.data.zero_()
                return

            # 行列積を計算
            product = self.weights[0].data.clone()
            for s in range(1, self.L):
                product = self.weights[s].data @ product

            # SVD でランク r に切り詰め
            U, S, Vh = torch.linalg.svd(product, full_matrices=False)
            S_truncated = S.clone()
            S_truncated[self.rank:] = 0

            # ランク r の行列を再構成
            product_rank_r = U @ torch.diag(S_truncated) @ Vh

            # 最後の層の重みを更新してランク制約を満たす
            # W_L ... W_1 = product_rank_r となるように W_L を調整
            if self.L == 1:
                self.weights[0].data = product_rank_r
            else:
                # W_{L-1} ... W_1 の疑似逆行列を計算
                prefix_product = self.weights[0].data.clone()
                for s in range(1, self.L - 1):
                    prefix_product = self.weights[s].data @ prefix_product

                # W_L = product_rank_r @ pinv(prefix_product)
                pinv_prefix = torch.linalg.pinv(prefix_product)
                self.weights[-1].data = product_rank_r @ pinv_prefix

        def forward(self, x):
            out = x
            for W in self.weights:
                out = out @ W.T
            return out

        def get_product_matrix(self):
            """行列積 W_L ... W_1 を計算"""
            product = self.weights[0].data.clone()
            for s in range(1, self.L):
                product = self.weights[s].data @ product
            return product

    return RankConstrainedDLN(H, r)


# ============================================================
# 2. 理論値計算 (Aoyagi 2024, Theorem 1)
# ============================================================

def compute_deficiency(H: List[int], r: int) -> List[int]:
    """
    Step 1: 余剰次元（Deficiency）を計算

    M^{(s)} = H^{(s)} - r

    Args:
        H: 各層の幅 [H^{(1)}, ..., H^{(L+1)}]
        r: ランク

    Returns:
        M: 余剰次元のリスト [M^{(1)}, ..., M^{(L+1)}]
    """
    return [h - r for h in H]


def find_bottleneck_set(M: List[int]) -> Set[int]:
    """
    Step 2: ボトルネック集合 M を探索（ソートベース最適化版）

    3つの条件を満たす集合を探す:
    1. 分離条件: max_{s∈M} M^{(s)} < min_{k∉M} M^{(k)}
    2. 総和条件（下界）: Σ_{s∈M} M^{(s)} >= l * max_{s∈M} M^{(s)}
    3. 総和条件（上界）: Σ_{s∈M} M^{(s)} < l * min_{k∉M} M^{(k)}

    計算量: O(L log L) (ソートベース)

    Args:
        M: 余剰次元のリスト

    Returns:
        bottleneck_set: ボトルネック集合のインデックス
    """
    n = len(M)

    # インデックスと共にソート (値が小さい順)
    sorted_indices = list(np.argsort(M))
    sorted_M = [M[i] for i in sorted_indices]

    # サイズ k = l+1 のプレフィックス集合のみを探索
    for k in range(1, n + 1):
        l = k - 1

        # 候補集合（ソート済みの先頭 k 個）
        subset_indices = set(sorted_indices[:k])

        # 値の取得
        max_M_in = sorted_M[k - 1]  # ソート済みなので最後の要素が最大
        sum_M_in = sum(sorted_M[:k])

        # 補集合の最小値 (存在する場合)
        if k < n:
            min_M_out = sorted_M[k]  # ソート済みなので次の要素が最小
        else:
            min_M_out = float('inf')  # 補集合なし

        # 条件判定
        # 1. 分離条件 (ソートしているので自動的に満たしやすい)
        cond1 = max_M_in < min_M_out

        # 2. 総和条件（下界）
        if l == 0:
            cond2 = True  # l=0 の場合は常に満たす
        else:
            cond2 = sum_M_in >= l * max_M_in

        # 3. 総和条件（上界）: 補集合がある場合のみ
        if k < n:
            if l == 0:
                cond3 = True  # l=0 の場合は常に満たす
            else:
                cond3 = sum_M_in < l * min_M_out
        else:
            cond3 = True  # 補集合なし

        if cond1 and cond2 and cond3:
            return subset_indices

    # 理論上ここには来ないはずだが、念のため全体を返す
    return set(range(n))


def compute_llc_theoretical(H: List[int], r: int) -> float:
    """
    Aoyagi (2024) Theorem 1 による LLC の理論値計算

    λ = Base Term + Adjustment 1 - Adjustment 2 + Interaction Term

    Args:
        H: 各層の幅 [H^{(1)}, ..., H^{(L+1)}]
        r: ランク

    Returns:
        lambda: 学習係数の理論値
    """
    # Step 1: 余剰次元
    M = compute_deficiency(H, r)

    # Step 2: ボトルネック集合
    bottleneck = find_bottleneck_set(M)
    l = len(bottleneck) - 1  # |M| - 1

    # Step 3: 学習係数 λ の計算
    H_1 = H[0]       # 入力層の幅
    H_Lp1 = H[-1]    # 出力層の幅

    # Base Term: (-r² + r(H^{(1)} + H^{(L+1)})) / 2
    base_term = (-r**2 + r * (H_1 + H_Lp1)) / 2

    # ★修正: l=0 (ボトルネックが単一層) の場合は基本項のみ
    # 調整項や相互作用項は存在しない
    if l == 0:
        return base_term

    # 以下、l > 0 の場合のみ計算
    S_M = sum(M[i] for i in bottleneck)
    M_ceil = int(np.ceil(S_M / l))  # 平均天井値
    a = S_M - (M_ceil - 1) * l       # 調整項

    # Adjustment 1: a(l-a) / (4l)
    adj1 = a * (l - a) / (4 * l)

    # Adjustment 2: l(l-1)/4 * (S_M / l)²
    adj2 = l * (l - 1) / 4 * (S_M / l)**2

    # Interaction Term: (1/2) * Σ_{i<j, i,j∈M} M^{(i)} M^{(j)}
    interaction = 0
    bottleneck_list = sorted(bottleneck)
    for i in range(len(bottleneck_list)):
        for j in range(i + 1, len(bottleneck_list)):
            interaction += M[bottleneck_list[i]] * M[bottleneck_list[j]]
    interaction /= 2

    # 最終結果
    llc = base_term + adj1 - adj2 + interaction

    return llc


def print_dln_info(H: List[int], r: int):
    """DLN の情報を表示"""
    M = compute_deficiency(H, r)
    bottleneck = find_bottleneck_set(M)
    llc = compute_llc_theoretical(H, r)

    print("=" * 50)
    print("DLN Configuration")
    print("=" * 50)
    print(f"層数 L = {len(H) - 1}")
    print(f"各層の幅 H = {H}")
    print(f"ランク r = {r}")
    print(f"余剰次元 M = {M}")
    print(f"ボトルネック集合 = {sorted(bottleneck)}")
    print(f"理論的 LLC = {llc:.4f}")
    print("=" * 50)


class SGLD(torch.optim.Optimizer):
    r"""
    Implements Stochastic Gradient Langevin Dynamics (SGLD) optimizer.

    This optimizer blends Stochastic Gradient Descent (SGD) with Langevin Dynamics,
    introducing Gaussian noise to the gradient updates. It can also include an
    elasticity term that , acting like
    a special form of weight decay.

    It follows Lau et al.'s (2023) implementation, which is a modification of
    Welling and Teh (2011) that omits the learning rate schedule and introduces
    an elasticity term that pulls the weights towards their initial values.

    The equation for the update is as follows:

    $$
    \begin{gathered}
    \Delta w_t=\frac{\epsilon}{2}\left(\frac{\beta n}{m} \sum_{i=1}^m \nabla \log p\left(y_{l_i} \mid x_{l_i}, w_t\right)+\gamma\left(w^_0-w_t\right) - \lambda w_t\right) \\
    +N(0, \epsilon\sigma^2)
    \end{gathered}
    $$

    where $w_t$ is the weight at time $t$, $\epsilon$ is the learning rate,
    $(\beta n)$ is the inverse temperature (we're in the tempered Bayes paradigm),
    $n$ is the number of training samples, $m$ is the batch size, $\gamma$ is
    the elasticity strength, $\lambda$ is the weight decay strength, $n$ is the
    number of samples, and $\sigma$ is the noise term.

    :param params: Iterable of parameters to optimize or dicts defining parameter groups
    :param lr: Learning rate (required)
    :param noise_level: Amount of Gaussian noise introduced into gradient updates (default: 1). This is multiplied by the learning rate.
    :param weight_decay: L2 regularization term, applied as weight decay (default: 0)
    :param elasticity: Strength of the force pulling weights back to their initial values (default: 0)
    :param temperature: Temperature. (default: 1)
    :param num_samples: Number of samples to average over (default: 1)

    Example:
        >>> optimizer = SGLD(model.parameters(), lr=0.1, temperature=torch.log(n)/n)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    Note:
        - The `elasticity` term is unique to this implementation and serves to guide the
        weights towards their original values. This is useful for estimating quantities over the local
        posterior.
    """

    def __init__(self, params, lr=1e-3, noise_level=1., elasticity=0., temperature: Union[Literal['adaptive'], float]=1., bounding_box_size=None, num_samples=1, batch_size=None):
        defaults = dict(lr=lr, noise_level=noise_level, elasticity=elasticity, temperature=temperature, bounding_box_size=None, num_samples=num_samples, batch_size=batch_size)
        super(SGLD, self).__init__(params, defaults)

        # Save the initial parameters if the elasticity term is set
        for group in self.param_groups:
            if group['elasticity'] != 0:
                for p in group['params']:
                    param_state = self.state[p]
                    param_state['initial_param'] = torch.clone(p.data).detach()
            if group['temperature'] == "adaptive":  # TODO: Better name
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

                    p.add_(dw, alpha = -0.5 * group['lr'])

                    # Add Gaussian noise
                    noise = torch.normal(mean=0., std=group['noise_level'], size=dw.size(), device=dw.device)
                    p.add_(noise, alpha=group['lr'] ** 0.5)

                    # Clamp to bounding box size
                    if group['bounding_box_size'] is not None:
                        torch.clamp_(p, min=-group['bounding_box_size'], max=group['bounding_box_size'])


# ============================================================
# 3. メイン: モデル生成 → SGLD 推定 → 理論値比較
# ============================================================

# 乱数シード（再現性のため）
# np.random.seed(42)
# torch.manual_seed(42)

# --- モデル構成の設定 ---
# 方法1: 手動で指定（検証用）- より単純なケース
# H = [3, 3, 3]  # 2層DLN: 入力3, 隠れ3, 出力1
# r = 1          # ランク
# 理論 LLC = (-r² + r(H_1 + H_{L+1})) / 2 = (-1 + 1*(3+1)) / 2 = 1.5

# 方法2: ランダム生成 (Lau et al. 2023, Appendix J)
H, r = generate_random_dln_config(L_low=2, L_high=4, H_low=5, H_high=10)

# モデル情報を表示
print_dln_info(H, r)

# 理論的 LLC を計算
llc_theoretical = compute_llc_theoretical(H, r)

# --- DLN モデルを作成 ---
teacher_model = create_dln_with_rank(H, r)

# ランク確認
with torch.no_grad():
    product = teacher_model.get_product_matrix()
    actual_rank = torch.linalg.matrix_rank(product).item()
    print(f"実際の行列積ランク: {actual_rank} (目標: {r})")

# --- データ生成 ---
noise_std = 1.0         # 観測ノイズの標準偏差 σ
num_data = 5000         # データ数 n
batch_size = 100
input_dim = H[0]        # 入力次元

# 入力: N(0, 1) 標準正規分布
# x_data = torch.rand(num_data, input_dim) * 20 - 10  # Lau23論文中の元の設定: U[-10, 10]
x_data = torch.randn(num_data, input_dim)

with torch.no_grad():
    y_true = teacher_model(x_data)
    y_data = y_true + torch.randn_like(y_true) * noise_std

print(f"データ形状: x={x_data.shape}, y={y_data.shape}")

# --- Student モデル（Teacher と同じ重みで初期化）---
student_model = create_dln_with_rank(H, r)
student_model.load_state_dict(teacher_model.state_dict())

# --- Initial モデル（基準点 w* として固定）---
initial_model = create_dln_with_rank(H, r)
initial_model.load_state_dict(student_model.state_dict())
initial_model.eval()

# ハイパーパラメータ
lr = 0.00001       # 学習率 (epsilon) - 安定性のため小さく
elasticity = 100.0  # 局所化の強さ (gamma)
num_steps = 10000   # ステップ数
beta = 1.0 / np.log(num_data)  # 逆温度 = 1 / log(n)

# ★ SGLD オプティマイザの設定
# 比較テスト: 元の設計（temperature=1.0, backward で β を掛ける）
optimizer = SGLD(
    student_model.parameters(),
    lr=lr,
    num_samples=num_data,
    batch_size=batch_size,
    temperature=1.0,  # 元の設計: temperature=1.0
    elasticity=elasticity  # ★必須: 初期位置(w*)に引き戻す力
)

# 損失関数 (MSE sum reduction)
criterion = nn.MSELoss(reduction='sum')

# NLL スケーリング係数: 1 / (2σ²)
# ガウスノイズ N(0, σ²) を仮定した負の対数尤度は SSE / (2σ²)
nll_scale = 1.0 / (2.0 * noise_std**2)

# 【論文準拠】エネルギー差分を記録するリスト (Algorithm 1, Line 9)
# 同じバッチで比較することで、バッチのサンプリングノイズが相殺される (Control Variates)
energy_diff_trace = []

# --- SGLD ループ (Algorithm 1) ---
student_model.train()

for step in range(num_steps):
    # 1. ミニバッチのサンプリング (Algorithm 1, Line 3)
    indices = torch.randperm(num_data)[:batch_size]
    x_batch = x_data[indices]
    y_batch = y_data[indices]

    optimizer.zero_grad()

    # 2. 現在のパラメータ w_t でのエネルギー推定 (Algorithm 1, Line 4)
    # U(w) = n * L(w)
    # ミニバッチ損失を (n/m) 倍して全データに換算
    y_pred = student_model(x_batch)
    batch_loss = criterion(y_pred, y_batch)  # SSE (sum)
    # NLL = SSE / (2σ²), scaled to full data: * (n/m)
    energy_current = batch_loss * nll_scale * (num_data / batch_size)

    # 3. 勾配計算と更新 (SGLD step)
    # 元の設計: backward で β を掛ける
    (beta * batch_loss * nll_scale).backward()
    optimizer.step()

    # 4. 【重要】基準点 w* でのエネルギー推定 (Algorithm 1, Line 6)
    # 同じミニバッチを使って計算する (Control Variates による分散低減)
    with torch.no_grad():
        y_pred_star = initial_model(x_batch)
        batch_loss_star = criterion(y_pred_star, y_batch)
        energy_star = batch_loss_star * nll_scale * (num_data / batch_size)

    # 5. エネルギー差分の記録 (Algorithm 1, Line 9 の Σ の中身)
    # ΔU = U(w_t) - U(w*)
    diff = energy_current.item() - energy_star.item()
    energy_diff_trace.append(diff)

# --- LLCの計算 (Algorithm 1, Line 9 / Eq. 12) ---
# Burn-in (初期の安定しない期間) を捨てる (Appendix H 推奨)
burn_in = int(num_steps * 0.5)
valid_diffs = energy_diff_trace[burn_in:]

# デバッグ: NaN チェック
nan_count = sum(1 for d in energy_diff_trace if np.isnan(d))
inf_count = sum(1 for d in energy_diff_trace if np.isinf(d))
print(f"\nデバッグ: NaN数={nan_count}, Inf数={inf_count}, 全サンプル数={len(energy_diff_trace)}")
if len(energy_diff_trace) > 0:
    print(f"最初の10個: {energy_diff_trace[:10]}")
    print(f"最後の10個: {energy_diff_trace[-10:]}")
    print(f"burn_in後の平均: {np.mean(valid_diffs):.4f}")
    print(f"beta = {beta:.6f}, temperature = {np.log(num_data):.4f}")

# 【論文 Algorithm 1 準拠】エネルギー差分の平均を計算
mean_energy_diff = np.mean(valid_diffs)

# LLC の推定式 (Eq. 12):
# λ̂ = β * E[U(w) - U(w*)]
# ここで U(w) = n * L(w) は既に (n/m) 倍でスケーリング済み
# よって: LLC = beta * mean_energy_diff
estimated_llc = beta * mean_energy_diff

# 標準誤差の計算
std_error = np.std(valid_diffs) / np.sqrt(len(valid_diffs))
std_error_scaled = beta * std_error

# --- 結果の表示 ---
print("\n" + "=" * 50)
print("結果比較")
print("=" * 50)
print(f"理論的 LLC (Aoyagi):  {llc_theoretical:.4f}")
print(f"推定 LLC (SGLD):      {estimated_llc:.4f} ± {std_error_scaled:.4f}")
print(f"誤差:                 {abs(estimated_llc - llc_theoretical):.4f} ({abs(estimated_llc - llc_theoretical) / llc_theoretical * 100:.1f}%)")
print("=" * 50)
print(f"パラメータ数: {sum(p.numel() for p in student_model.parameters())}")
print(f"noise_std: {noise_std}, nll_scale: {nll_scale:.4f}")