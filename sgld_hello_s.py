# SGLD による LLC (Local Learning Coefficient) 推定
# 論文: Lau et al. (2023) の Algorithm 1 に準拠

from typing import Literal, Union
import torch
import torch.nn as nn
import numpy as np


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


class SelfLinearAttention(nn.Module):
    """
    Self-Linear Attention (SLA) モデル for In-Context Learning

    h_SLA(X, w) = X W_Q W_K^T X^T

    ここで:
    - X: (N+1, d+1) プロンプト行列 (N個の例 + 1個のクエリ)
    - W_Q: (d+1, d_l) Query 行列
    - W_K: (d+1, d_l) Key 行列
    - 出力: X W_Q W_K^T X^T の (N+1, N+1) 要素 (クエリに対する予測)
    """
    def __init__(self, input_dim, latent_dim, init_scale=0.01):
        """
        Args:
            input_dim: 入力次元 (d+1)
            latent_dim: 潜在次元 (d_l)
            init_scale: 初期化のスケール
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        # W_Q: (d+1, d_l)
        self.W_Q = nn.Parameter(torch.randn(input_dim, latent_dim) * init_scale)
        # W_K: (d+1, d_l)
        self.W_K = nn.Parameter(torch.randn(input_dim, latent_dim) * init_scale)

    def get_effective_matrix(self):
        """
        W_Q[:d, :] @ W_K[:d, :].T を計算 (d × d 部分の積)
        """
        d = self.input_dim - 1
        return self.W_Q[:d, :] @ self.W_K[:d, :].T

    def get_effective_rank(self):
        """
        W_Q[:d, :] @ W_K[:d, :].T の実効ランクを計算
        """
        M = self.get_effective_matrix()
        return torch.linalg.matrix_rank(M).item()

    def forward(self, X):
        """
        Args:
            X: (batch_size, N+1, d+1) バッチ化されたプロンプト行列
        Returns:
            (batch_size, 1) 各プロンプトに対するクエリの予測値
        """
        # 最後の行（クエリ行）のみを投影（計算効率化）
        # X[:, -1, :] の形状: (batch_size, d+1)
        X_query = X[:, -1, :]  # (batch_size, d+1)

        # クエリ行のみを投影
        XWQ_query = X_query @ self.W_Q  # (batch_size, d_l)
        XWK_query = X_query @ self.W_K  # (batch_size, d_l)

        # --- 旧実装（全行を投影 → 非効率）---
        # XWQ = X @ self.W_Q  # (batch_size, N+1, d_l) ← N+1行すべて計算
        # XWK = X @ self.W_K
        # pred = (XWQ[:, -1, :] * XWK[:, -1, :]).sum(dim=-1)

        # --- 新実装（クエリ行のみ投影）---
        # attn[-1, -1] = <X[-1] @ W_Q, X[-1] @ W_K>
        pred = (XWQ_query * XWK_query).sum(dim=-1)  # (batch,)

        return pred.unsqueeze(-1)


# ============================================================
# モデル生成プロセス (sgld_hello.py と同様)
# ============================================================

from typing import Tuple, List, Set


# ============================================================
# 理論値計算 (Aoyagi 2024, Theorem 1) - sgld_hello.py から移植
# ============================================================

def compute_deficiency(H: List[int], r: int) -> List[int]:
    """
    余剰次元（Deficiency）を計算: M^{(s)} = H^{(s)} - r
    """
    return [h - r for h in H]


def find_bottleneck_set(M: List[int]) -> Set[int]:
    """
    ボトルネック集合を探索（ソートベース最適化版）
    """
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
    """
    Aoyagi (2024) Theorem 1 による DLN の LLC 理論値計算
    """
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
    SLA モデルの LLC 理論値計算

    λ_SLA = λ(W_Q[:d,:] @ W_K[:d,:].T) + (d + 0.5)

    W_Q[:d,:] @ W_K[:d,:].T は 2層 DLN と同等: H = [d, d_l, d], rank = r

    Args:
        d: 入力次元
        d_l: 潜在次元
        r: ランク

    Returns:
        λ_SLA: SLA の学習係数の理論値
    """
    # ランクの整合性チェック
    max_possible_rank = min(d, d_l)
    if r > max_possible_rank:
        raise ValueError(f"Rank r={r} is impossible for dimensions d={d}, d_l={d_l}. Max rank = {max_possible_rank}")

    # W_Q[:d,:] @ W_K[:d,:].T 部分の λ を 2層 DLN として計算
    H = [d, d_l, d]  # 入力 d, 隠れ d_l, 出力 d
    lambda_matrix = compute_llc_theoretical_dln(H, r)

    # SLA 全体の λ = λ_matrix + (d + 0.5)
    lambda_sla = lambda_matrix + (d + 0.5)

    return lambda_sla

def generate_random_sla_config(
    d_low: int = 5,
    d_high: int = 15,
    dl_multiplier_low: float = 1.5,
    dl_multiplier_high: float = 3.0
) -> Tuple[int, int, int]:
    """
    ランダムな SLA 構成を生成

    Args:
        d_low, d_high: 入力次元 d の範囲
        dl_multiplier_low, dl_multiplier_high: d_l = d * multiplier の範囲

    Returns:
        d: 入力次元
        d_l: 潜在次元 (d より十分大きい)
        r: 真のランク (W_Q[:d,:] @ W_K[:d,:].T の d×d 部分のランク)
    """
    # d をサンプリング
    d = np.random.randint(d_low, d_high + 1)

    # d_l を d より十分大きくサンプリング: d_l = d * multiplier
    multiplier = np.random.uniform(dl_multiplier_low, dl_multiplier_high)
    d_l = int(d * multiplier)

    # ランク r のサンプリング
    # d_l >= d なので、max_rank = d
    max_rank = d
    if np.random.rand() < 0.5:
        # 確率 0.5 で低ランク
        r = np.random.randint(1, max_rank + 1)
    else:
        # 確率 0.5 でフルランク
        r = max_rank

    return d, d_l, r


def create_sla_with_rank(d: int, d_l: int, r: int, init_scale: float = 0.1) -> SelfLinearAttention:
    """
    指定されたランクを持つ SLA モデルを作成

    W_Q[:d, :] @ W_K[:d, :].T が d × d でランク r を持つように調整

    Args:
        d: 入力次元
        d_l: 潜在次元
        r: 真のランク

    Returns:
        SLA モデル（d×d 部分の積がランク r になるように修正済み）
    """
    # ランクの整合性チェック
    max_possible_rank = min(d, d_l)
    if r > max_possible_rank:
        raise ValueError(f"Rank r={r} is impossible for dimensions d={d}, d_l={d_l}. Max rank = {max_possible_rank}")

    input_dim = d + 1
    model = SelfLinearAttention(input_dim, d_l, init_scale)

    with torch.no_grad():
        if r == 0:
            # ★修正: 行列全体をゼロにする（クエリ・交差部分も含む）
            # これにより完全な特異点（Singular Point）を作成
            model.W_Q.zero_()
            model.W_K.zero_()
        else:
            # 現在の d×d 部分の積を計算
            W_Q_d = model.W_Q[:d, :].clone()  # (d, d_l)
            W_K_d = model.W_K[:d, :].clone()  # (d, d_l)
            M = W_Q_d @ W_K_d.T  # (d, d)

            # SVD でランク r に切り詰め
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
            S_truncated = S.clone()
            S_truncated[r:] = 0

            # ランク r の行列を再構成
            M_rank_r = U @ torch.diag(S_truncated) @ Vh

            # W_K[:d, :] を調整して M_rank_r = W_Q[:d, :] @ W_K[:d, :].T となるようにする
            # W_K[:d, :].T = pinv(W_Q[:d, :]) @ M_rank_r
            # W_K[:d, :] = (pinv(W_Q[:d, :]) @ M_rank_r).T = M_rank_r.T @ pinv(W_Q[:d, :]).T
            W_Q_pinv = torch.linalg.pinv(W_Q_d)  # (d_l, d)
            new_W_K_d = (W_Q_pinv @ M_rank_r).T  # (d, d_l)
            model.W_K[:d, :] = new_W_K_d

    return model


def print_sla_info(d: int, d_l: int, r: int, model: SelfLinearAttention = None):
    """SLA の情報を表示"""
    # 理論値計算
    llc_theoretical = compute_llc_theoretical_sla(d, d_l, r)
    H = [d, d_l, d]
    lambda_matrix = compute_llc_theoretical_dln(H, r)

    print("=" * 50)
    print("SLA Configuration")
    print("=" * 50)
    print(f"入力次元 d = {d}")
    print(f"潜在次元 d_l = {d_l}")
    print(f"目標ランク r = {r}")
    if model is not None:
        actual_rank = model.get_effective_rank()
        print(f"実際のランク (W_Q[:d,:] @ W_K[:d,:].T) = {actual_rank}")
    print(f"パラメータ数 = 2 * (d+1) * d_l = {2 * (d + 1) * d_l}")
    print("-" * 50)
    print(f"λ(W_Q[:d,:]@W_K[:d,:].T) = {lambda_matrix:.4f}  (2層DLN: H=[{d},{d_l},{d}], r={r})")
    print(f"理論的 LLC = λ_matrix + (d + 0.5) = {lambda_matrix:.4f} + {d + 0.5:.1f} = {llc_theoretical:.4f}")
    print("=" * 50)

    return llc_theoretical


def generate_icl_prompt(d, N, Lambda=None):
    """
    ICL プロンプトを生成する

    P_τ = X^T = [
        [x_{τ,1}, x_{τ,2}, ..., x_{τ,N}, x_{query}],  (d行)
        [<m_τ, x_{τ,1}>, <m_τ, x_{τ,2}>, ..., <m_τ, x_{τ,N}>, 0]  (1行)
    ] ∈ R^{(d+1)×(N+1)}

    Args:
        d: 入力次元
        N: 例の数
        Lambda: 共分散行列 (d, d), None の場合は単位行列

    Returns:
        X: (N+1, d+1) プロンプト行列
        y_query: クエリの真のラベル <m, x_query>
    """
    if Lambda is None:
        Lambda = torch.eye(d)

    # m_τ ~ N(0, I_d)
    m = torch.randn(d)

    # x_{τ,i}, x_query ~ N(0, Λ)
    # Λ = L L^T (Cholesky分解) として、x = L @ z, z ~ N(0, I)
    L = torch.linalg.cholesky(Lambda)
    z = torch.randn(N + 1, d)
    x_all = z @ L.T  # (N+1, d)

    x_examples = x_all[:N]  # (N, d)
    x_query = x_all[N]      # (d,)

    # ラベル: y_i = <m, x_i>
    y_examples = x_examples @ m  # (N,)
    y_query = x_query @ m        # scalar

    # プロンプト行列 P_τ^T = X ∈ R^{(N+1)×(d+1)}
    # 各行: [x_i, y_i] for examples, [x_query, 0] for query
    X = torch.zeros(N + 1, d + 1)
    X[:N, :d] = x_examples
    X[:N, d] = y_examples
    X[N, :d] = x_query
    X[N, d] = 0  # クエリのラベル位置は 0

    return X, y_query


def generate_icl_batch(num_tasks, d, N, Lambda=None):
    """
    バッチ化された ICL プロンプトを生成（ベクトル化版）

    Args:
        num_tasks: タスク数 (バッチサイズ)
        d: 入力次元
        N: 例の数
        Lambda: 共分散行列

    Returns:
        X_batch: (num_tasks, N+1, d+1) バッチ化プロンプト
        y_batch: (num_tasks, 1) クエリの真のラベル
    """
    if Lambda is None:
        Lambda = torch.eye(d)

    # 全タスクの m ベクトルを一度に生成
    m = torch.randn(num_tasks, d)  # (num_tasks, d)

    # 全タスクの x ベクトルを一度に生成
    L = torch.linalg.cholesky(Lambda)
    z = torch.randn(num_tasks, N + 1, d)  # (num_tasks, N+1, d)
    x_all = z @ L.T  # (num_tasks, N+1, d)

    x_examples = x_all[:, :N, :]  # (num_tasks, N, d)
    x_query = x_all[:, N, :]      # (num_tasks, d)

    # ラベル: y_i = <m, x_i> をベクトル化計算
    y_examples = torch.einsum('td,tnd->tn', m, x_examples)  # (num_tasks, N)
    y_query = torch.einsum('td,td->t', m, x_query)  # (num_tasks,)

    # プロンプト行列を構築
    X_batch = torch.zeros(num_tasks, N + 1, d + 1)
    X_batch[:, :N, :d] = x_examples
    X_batch[:, :N, d] = y_examples
    X_batch[:, N, :d] = x_query
    X_batch[:, N, d] = 0

    return X_batch, y_query.unsqueeze(-1)


# ============================================================
# メイン: モデル生成 → SGLD 推定
# ============================================================

# --- モデル構成の設定 ---
# 方法1: 手動で指定
# d = 10          # 入力次元
# latent_dim = 5  # d_l (潜在次元)
# r = 3           # 真のランク

# 方法2: ランダム生成
d, latent_dim, r = generate_random_sla_config(d_low=5, d_high=15)

# その他の設定
N = 1000                 # ICL の例の数
input_dim = d + 1       # プロンプトの列数 (d+1)
noise_std = 1.0         # 観測ノイズの標準偏差 σ
num_data = 5000         # タスク数 (データ数 n)
batch_size = 100

# 共分散行列 Λ (None = 単位行列)
Lambda = None

# 1. 真のモデル (Teacher) の作成 - ランク r の制約付き
teacher_model = create_sla_with_rank(d, latent_dim, r)

# モデル情報を表示 & 理論値を取得
llc_theoretical = print_sla_info(d, latent_dim, r, teacher_model)

# 2. データの生成 (ICL プロンプト)
# 各タスク τ に対して:
#   P_τ = X^T = [[x_1, ..., x_N, x_query], [<m,x_1>, ..., <m,x_N>, 0]]
print("Generating ICL prompts...")
x_data, _ = generate_icl_batch(num_data, d, N, Lambda)
# x_data: (num_data, N+1, d+1)

# ★重要: Teacher モデルの出力を正解とする（sgld_hello.py と同様）
with torch.no_grad():
    y_true = teacher_model(x_data)

# ノイズを加える
y_data = y_true + torch.randn_like(y_true) * noise_std
print(f"x_data shape: {x_data.shape}, y_data shape: {y_data.shape}")

# 3. 推定用モデル (Student) の準備
# Teacherと同じ構造を作り、重みをTeacherと全く同じ値 (w*) にコピーする
student_model = SelfLinearAttention(input_dim, latent_dim)
student_model.load_state_dict(teacher_model.state_dict())

# 4. 【論文準拠】基準点 w* (初期パラメータ) を保存するモデル
# SGLDで動く前の状態を固定しておく (Algorithm 1, Line 6 の w* に相当)
initial_model = SelfLinearAttention(input_dim, latent_dim)
initial_model.load_state_dict(student_model.state_dict())
initial_model.eval()  # 評価モード (重みを固定)

# ハイパーパラメータ
lr = 0.00001     # 学習率 (epsilon) - 安定性のため小さく
elasticity = 10.0  # 局所化の強さ (gamma)
num_steps = 10000
beta = 1.0 / np.log(num_data)  # 逆温度 = 1 / log(n)

# ★ SGLD オプティマイザの設定
optimizer = SGLD(
    student_model.parameters(),
    lr=lr,
    num_samples=num_data,
    batch_size=batch_size,
    temperature=1.0,  # sgld_hello.py と同じ設定
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
    # U(w) = n * NLL(w) = n * SSE / (2σ²)
    # ミニバッチから全データを推定: (n/m) 倍
    y_pred = student_model(x_batch)
    batch_loss = criterion(y_pred, y_batch)
    energy_current = batch_loss * nll_scale * (num_data / batch_size)

    # 3. 勾配計算と更新 (SGLD step)
    # SGLDクラス内部で n/m 倍しているので、beta * NLL を渡す
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

# デバッグ: エネルギー差分の確認
print(f"\nデバッグ:")
print(f"最初の10個: {energy_diff_trace[:10]}")
print(f"最後の10個: {energy_diff_trace[-10:]}")
print(f"burn_in後の平均: {np.mean(valid_diffs):.4f}")

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
print(f"理論的 LLC:           {llc_theoretical:.4f}")
print(f"推定 LLC (SGLD):      {estimated_llc:.4f} ± {std_error_scaled:.4f}")
if llc_theoretical != 0:
    print(f"誤差:                 {abs(estimated_llc - llc_theoretical):.4f} ({abs(estimated_llc - llc_theoretical) / llc_theoretical * 100:.1f}%)")
print("=" * 50)
print(f"パラメータ数: {sum(p.numel() for p in student_model.parameters())}")
print(f"noise_std: {noise_std}, nll_scale: {nll_scale:.4f}")