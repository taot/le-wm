import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    import sys
    import inspect

    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    # The notebook is in playground/; module.py is in the repo root.
    sys.path.insert(0, str(mo.notebook_dir().parent))
    from module import SIGReg

    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
    return SIGReg, inspect, np, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 理解 SIGReg

    这个 notebook 一步一步解释 `module.py` 里的 `SIGReg`
    （Sketch Isotropic Gaussian Regularizer）。

    **一句话总结：** SIGReg 在很多随机方向上检查 embedding 的投影
    "看起来像不像标准正态 N(0,1)"。不像的话，loss 就变大。

    ## 0. SIGReg 在训练中的位置

    在 `train.py` 里，loss 是这样计算的：

    ```python
    output["pred_loss"]   = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = sigreg(emb.transpose(0, 1))      # emb: (B, T, D) -> (T, B, D)
    output["loss"]        = output["pred_loss"] + lambd * output["sigreg_loss"]   # lambd = 0.09
    ```

    **为什么需要它？** 只用 `pred_loss` 的话，有一个"作弊"的解：
    encoder 对所有输入都输出同一个向量。这时预测永远正确，`pred_loss = 0`，
    但是 embedding 没有任何信息。这叫做 **表示坍缩（representation collapse）**。

    SIGReg 要求 embedding 的分布是 **各向同性高斯分布** N(0, I)。
    坍缩的分布离 N(0, I) 很远，所以 SIGReg 会惩罚它。

    要理解 SIGReg，你需要 3 个概念。下面按顺序讲：

    1. **特征函数**：用一条曲线描述一个 1 维分布。
    2. **Epps–Pulley 统计量**：用一个数字衡量"这组 1 维样本离 N(0,1) 有多远"。
    3. **随机投影**：把 D 维问题变成很多个 1 维问题。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 特征函数：你只需要一个事实

    随机变量 $X$ 的特征函数是

    $$
    \varphi_X(t) = \mathbb{E}\left[e^{itX}\right]
    = \underbrace{\mathbb{E}[\cos(tX)]}_{\text{实部}} + i\,\underbrace{\mathbb{E}[\sin(tX)]}_{\text{虚部}}
    $$

    $\varphi_X$ 读作"$X$ 的特征函数"。

    （第二个等号用的是欧拉公式 $e^{i\theta} = \cos\theta + i\sin\theta$。）

    **关键事实：** 对标准正态 $X \sim \mathcal{N}(0,1)$，

    $$
    \mathbb{E}[\cos(tX)] = e^{-t^2/2}, \qquad \mathbb{E}[\sin(tX)] = 0 .
    $$

    虚部是 0，因为正态分布关于 0 对称，而 $\sin$ 是奇函数。

    /// details | 推导：为什么 $\mathbb{E}[\cos(tX)] = e^{-t^2/2}$？（点击展开）

    $X \sim \mathcal{N}(0,1)$ 的密度是 $p(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$。要证明的是

    $$
    f(t) := \mathbb{E}[\cos(tX)] = \int_{-\infty}^{\infty} \cos(tx)\, p(x)\, dx = e^{-t^2/2} .
    $$

    #### 方法 1：微分方程

    **第 1 步：p(x) 的一个性质。** 对 $p$ 求导：

    $$
    p'(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2} \cdot (-x) = -x\, p(x) .
    $$

    **第 2 步：对 t 求导。** 把求导放到积分里面（$p$ 衰减得很快，所以可以这样做），
    再用第 1 步把 $-x\,p(x)$ 换成 $p'(x)$：

    $$
    f'(t) = \int_{-\infty}^{\infty} -x \sin(tx)\, p(x)\, dx = \int_{-\infty}^{\infty} \sin(tx)\, p'(x)\, dx .
    $$

    **第 3 步：分部积分。** 取 $u = \sin(tx)$，$v = p(x)$：

    $$
    f'(t) = \Big[\sin(tx)\, p(x)\Big]_{-\infty}^{\infty} - \int_{-\infty}^{\infty} t \cos(tx)\, p(x)\, dx .
    $$

    第一项是 0，因为 $x \to \pm\infty$ 时 $p(x) \to 0$。第二项里的积分正好是 $f(t)$。所以得到这个推导中最关键的方程：

    $$
    f'(t) = -t\, f(t) .
    $$

    **第 4 步：初始值。** $f(0) = \int p(x)\, dx = 1$（任何密度的积分都是 1）。

    **第 5 步：解方程。** 定义 $g(t) = f(t)\, e^{t^2/2}$，然后求导：

    $$
    g'(t) = \big(f'(t) + t\, f(t)\big)\, e^{t^2/2} = 0 .
    $$

    所以 $g$ 是常数：$g(t) = g(0) = f(0) = 1$。因此 $f(t) = e^{-t^2/2}$。$\blacksquare$

    #### 方法 2：泰勒展开（用来验证）

    把 $\cos$ 展开成级数，再对每一项取期望。需要用到 N(0,1) 的偶数阶矩
    $\mathbb{E}[X^{2k}] = \frac{(2k)!}{2^k\, k!}$（例如 $\mathbb{E}[X^2]=1$，$\mathbb{E}[X^4]=3$）：

    $$
    \mathbb{E}[\cos(tX)]
    = \sum_{k=0}^{\infty} \frac{(-1)^k t^{2k}}{(2k)!}\, \mathbb{E}[X^{2k}]
    = \sum_{k=0}^{\infty} \frac{1}{k!}\left(-\frac{t^2}{2}\right)^k
    = e^{-t^2/2} .
    $$

    最后一步用的是 $e^y = \sum_k y^k / k!$，这里 $y = -t^2/2$。

    #### 为什么 $\mathbb{E}[\sin(tX)] = 0$？

    $\sin(tx)$ 是奇函数，$p(x)$ 是偶函数，所以 $\sin(tx)\,p(x)$ 是奇函数。
    奇函数在 $(-\infty, \infty)$ 上的积分是 0。

    #### 怎么能想到这些方法？

    这两个方法都不是"灵感"，而是遇到这类问题时的常用套路：

    - **方法 2 的思路：把不会算的东西，变成会算的东西。**
      $\mathbb{E}[\cos(tX)]$ 不会算，但是 $\mathbb{E}[X^k]$（矩）会算。
      所以把 $\cos$ 展开成 $x$ 的幂，再一项一项算。最后看到 $\frac{1}{k!}(\cdots)^k$ 的形式，就想到 $e^y$ 的级数。
    - **方法 1 的思路：找到这个对象最特别的性质，然后用它。**
      高斯密度最特别的性质是 $p'(x) = -x\,p(x)$。所以看到"$x$ 乘以高斯"，
      就要想到"这是一个导数"，然后用分部积分把 $x$ 去掉。
    - **积分里有参数 t 时，对 t 求导**，常常能得到一个简单的微分方程。这是计算含参积分的标准技巧。
    - **先猜答案，再证明。** 下面的图里，$N(0,1)$ 样本的 cos 平均值曲线看起来就像 $e^{-t^2/2}$。
      知道了答案，再去找证明，会容易很多。

    ///

    **唯一性定理：** 两个分布的特征函数相同，那么这两个分布就相同。
    所以，特征函数是分布的"指纹"。

    **用样本估计：** 有样本 $x_1, \dots, x_N$ 时，把期望换成平均值：

    $$
    \hat\varphi(t) = \frac{1}{N}\sum_{j=1}^N \cos(t x_j) + i \cdot \frac{1}{N}\sum_{j=1}^N \sin(t x_j)
    $$

    这叫 **经验特征函数**。检查"样本是不是 N(0,1)"，
    就是检查：cos 平均值曲线是否接近 $e^{-t^2/2}$，sin 平均值曲线是否接近 0。

    下面你可以选择不同的分布，自己看。
    所有基础分布的均值都是 0，方差都是 1。你可以用滑块改变尺度 σ 和平移 μ：
    $x = \sigma \cdot x_{\text{base}} + \mu$。设 σ = 0 就是 **完全坍缩**（所有点都相同）。
    """)
    return


@app.cell
def _(np):
    DISTS: dict[str, str] = {
        "正态 N(0,1)": "normal",
        "均匀 (方差 1)": "uniform",
        "拉普拉斯 (方差 1)": "laplace",
        "Student-t, df=5 (方差 1)": "student_t",
        "双峰 (方差 1)": "bimodal",
    }

    def sample_1d(name: str, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n samples with mean 0 and variance 1 from the named distribution."""
        if name == "normal":
            return rng.standard_normal(n)
        if name == "uniform":
            return rng.uniform(-np.sqrt(3), np.sqrt(3), n)
        if name == "laplace":
            return rng.laplace(0.0, 1 / np.sqrt(2), n)
        if name == "student_t":
            return rng.standard_t(5, n) * np.sqrt(3 / 5)
        if name == "bimodal":
            signs = rng.choice([-1.0, 1.0], n)
            return 0.9 * signs + np.sqrt(0.19) * rng.standard_normal(n)
        raise ValueError(name)

    def ecf(x: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Empirical characteristic function: (mean cos(t x), mean sin(t x)) for each t."""
        xt = np.outer(x, t)
        return np.cos(xt).mean(0), np.sin(xt).mean(0)

    def sigreg_weights(knots: int = 17, t_max: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
        """Same knots and weights as SIGReg.__init__ in module.py."""
        t = np.linspace(0, t_max, knots)
        dt = t_max / (knots - 1)
        w = np.full(knots, 2 * dt)
        w[[0, -1]] = dt
        return t, w * np.exp(-(t**2) / 2)

    def epps_pulley_1d(x: np.ndarray, knots: int = 17) -> float:
        """1-D Epps-Pulley statistic, exactly as SIGReg computes it for one projection."""
        t, w = sigreg_weights(knots)
        c, s = ecf(x, t)
        err = (c - np.exp(-(t**2) / 2)) ** 2 + s**2
        return float(len(x) * (err @ w))

    return DISTS, ecf, epps_pulley_1d, sample_1d, sigreg_weights


@app.cell
def _(DISTS: dict[str, str], mo):
    dist_dd = mo.ui.dropdown(options=DISTS, value="正态 N(0,1)", label="分布")
    sigma_sl = mo.ui.slider(0.0, 3.0, step=0.05, value=1.0, label="尺度 σ", show_value=True)
    mu_sl = mo.ui.slider(-2.0, 2.0, step=0.05, value=0.0, label="平移 μ", show_value=True)
    n_sl = mo.ui.slider(16, 4096, step=16, value=512, label="样本数 N", show_value=True)
    mo.hstack([dist_dd, sigma_sl, mu_sl, n_sl], wrap=True)
    return dist_dd, mu_sl, n_sl, sigma_sl


@app.cell
def _(dist_dd, ecf, mu_sl, n_sl, np, plt, sample_1d, sigma_sl):
    _rng = np.random.default_rng(0)
    x_1d = sigma_sl.value * sample_1d(dist_dd.value, n_sl.value, _rng) + mu_sl.value

    _t = np.linspace(0, 6, 300)
    _c, _s = ecf(x_1d, _t)

    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.6))
    _ax = _axes[0]
    _bins = np.linspace(-5, 5, 61)
    _ax.hist(np.clip(x_1d, -5, 5), bins=_bins, density=True, alpha=0.6, label="samples")
    _g = np.linspace(-5, 5, 300)
    _ax.plot(_g, np.exp(-_g**2 / 2) / np.sqrt(2 * np.pi), "k--", label="N(0,1) pdf")
    _ax.set_title("Samples")
    _ax.legend(fontsize=8)

    _ax = _axes[1]
    _ax.plot(_t, _c, label="mean cos(t x)  (real part)")
    _ax.plot(_t, np.exp(-_t**2 / 2), "k--", label="exp(-t^2/2)  (target)")
    _ax.axvspan(0, 3, color="C2", alpha=0.08, label="SIGReg range t in [0,3]")
    _ax.set_ylim(-1.1, 1.1)
    _ax.set_xlabel("t")
    _ax.set_title("Real part of ECF")
    _ax.legend(fontsize=8)

    _ax = _axes[2]
    _ax.plot(_t, _s, color="C1", label="mean sin(t x)  (imag part)")
    _ax.axhline(0, color="k", ls="--", label="0  (target)")
    _ax.axvspan(0, 3, color="C2", alpha=0.08)
    _ax.set_ylim(-1.1, 1.1)
    _ax.set_xlabel("t")
    _ax.set_title("Imaginary part of ECF")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return (x_1d,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **试一试：**

    - **正态，σ=1，μ=0**：两条曲线几乎和目标重合。N 越大，越接近。
    - **σ = 0（完全坍缩）**：所有 $x_j = 0$，所以 $\cos(t\cdot 0) = 1$。实部是一条水平线 1，离目标很远。
    - **σ = 2（太宽）**：实部下降得太快。σ = 0.5（太窄）：实部下降得太慢。
      特征函数的"宽度"和分布的宽度相反（傅里叶变换的性质）。
    - **μ ≠ 0（均值不是 0）**：虚部不再是 0。理论上 $\varphi(t) = e^{i\mu t}e^{-t^2/2}$，
      所以虚部是 $\sin(\mu t)e^{-t^2/2}$。
    - **均匀、双峰**：均值和方差都正确！但是曲线形状还是不对。
      这说明特征函数比较的是 **整个分布**，不只是前两阶矩（均值、方差）。
    - **拉普拉斯、Student-t**：差距较小，主要在 t 较大的地方。这些分布比较难检测。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Epps–Pulley 统计量：把一条曲线变成一个数字

    我们需要一个数字作为 loss。自然的想法：把 $\hat\varphi(t)$ 和 $\varphi(t)$ 之间的差距
    在所有 $t$ 上积分。Epps–Pulley 统计量就是这样定义的：

    $$
    T = N \int_{-\infty}^{\infty} \big|\hat\varphi(t) - \varphi(t)\big|^2 \, w(t)\, dt,
    \qquad w(t) = e^{-t^2/2}
    $$

    其中复数的模平方是

    $$
    \big|\hat\varphi(t) - \varphi(t)\big|^2
    = \Big(\underbrace{\tfrac1N\textstyle\sum_j \cos(t x_j)}_{\text{cos 平均}} - e^{-t^2/2}\Big)^2 +
    \Big(\underbrace{\tfrac1N\textstyle\sum_j \sin(t x_j)}_{\text{sin 平均}}\Big)^2 .
    $$

    这正是代码里的这一行：

    ```python
    err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
    ```

    **为什么有权重 $w(t)$？** 当 $t$ 很大时，$\cos(t x)$ 振荡得很快，估计噪声很大，
    而且对应的是分布的很细的细节。所以给大的 $t$ 很小的权重。

    **为什么代码只用 $t \in [0, 3]$？**

    1. **对称性**：$\hat\varphi(-t) = \overline{\hat\varphi(t)}$（共轭），$\varphi(-t) = \varphi(t)$（实数）。
       所以 $|\hat\varphi(t)-\varphi(t)|^2$ 关于 $t$ 对称。$\int_{-\infty}^{\infty} = 2\int_0^{\infty}$。
    2. **截断**：$w(3) = e^{-4.5} \approx 0.011$。$t > 3$ 的部分贡献很小，可以忽略。

    **积分怎么算？** 用 **梯形法则**，17 个等距点（knots）。
    梯形法则的权重是：端点 $dt/2$，中间点 $dt$。再乘以对称性的因子 2，得到：
    端点 $dt$，中间点 $2\,dt$。这正是 `__init__` 里的代码：

    ```python
    weights = torch.full((knots,), 2 * dt)   # 中间点: 2·dt
    weights[[0, -1]] = dt                    # 端点:   dt
    self.weights = weights * window          # 再乘以 w(t) = exp(-t²/2)
    ```

    注意：`self.phi` 和 `window` 都是 $e^{-t^2/2}$，但是作用不同。
    `phi` 是 **目标**（N(0,1) 的特征函数），`window` 是 **权重** $w(t)$。它们相同只是巧合。

    下图用上面选择的样本，显示误差曲线和加权后的曲线。**阴影面积（乘以 N）就是 $T$。**
    """)
    return


@app.cell
def _(epps_pulley_1d, mo, np, plt, sigreg_weights, x_1d):
    _t = np.linspace(0, 3, 300)
    _xt = np.outer(x_1d, _t)
    _err = (np.cos(_xt).mean(0) - np.exp(-_t**2 / 2)) ** 2 + np.sin(_xt).mean(0) ** 2
    _weighted = _err * np.exp(-_t**2 / 2)
    _tk, _wk = sigreg_weights(17)
    _xk = np.outer(x_1d, _tk)
    _errk = (np.cos(_xk).mean(0) - np.exp(-_tk**2 / 2)) ** 2 + np.sin(_xk).mean(0) ** 2

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.4))
    _axes[0].plot(_t, _err, label="|ECF - target|^2")
    _axes[0].plot(_tk, _errk, "o", ms=4, label="17 knots")
    _axes[0].set_xlabel("t")
    _axes[0].set_title("Squared error")
    _axes[0].legend(fontsize=8)
    _axes[1].plot(_t, _weighted, color="C3", label="error * exp(-t^2/2)")
    _axes[1].fill_between(_t, _weighted, color="C3", alpha=0.25)
    _axes[1].set_xlabel("t")
    _axes[1].set_title("Weighted error (area x 2N = T)")
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()

    mo.vstack([
        _fig,
        mo.md(f"当前样本（N = {len(x_1d)}）的统计量：**T = {epps_pulley_1d(x_1d):.4f}**"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.1 为什么乘以 N？

    **一句话：** 乘以 $N$，是为了让"完全正确"的样本总是得到差不多相同的分数（约 1.05），
    不管样本有多少。这样就可以用一个固定的标准判断"像不像 N(0,1)"。

    先看一个更简单的例子（抛硬币），再回到 SIGReg。

    #### 例子：一枚硬币是不是公平的？

    抛 $N$ 次，正面的比例是 $\hat p$。公平硬币的正面概率是 0.5，所以我们看误差 $(\hat p - 0.5)^2$。

    **情况 A：硬币是公平的。** $\hat p$ 不会正好是 0.5，因为有随机误差。$N$ 越大，误差越小。
    下面会证明：

    $$
    \mathbb{E}\big[(\hat p - 0.5)^2\big] = \frac{0.25}{N} .
    $$

    | $N$ | 误差 $(\hat p - 0.5)^2$ 大约是 | $N \times$ 误差 |
    |---|---|---|
    | 100 | 0.0025 | **0.25** |
    | 10000 | 0.000025 | **0.25** |

    **情况 B：硬币不公平，正面概率是 0.6。** $N$ 很大时 $\hat p \to 0.6$，
    误差 $\to (0.6 - 0.5)^2 = 0.01$，不再减小。

    | $N$ | 误差 $(\hat p - 0.5)^2$ 大约是 | $N \times$ 误差 |
    |---|---|---|
    | 100 | 0.01 | **1** |
    | 10000 | 0.01 | **100** |

    **看两张表的最后一列：**

    - 只看误差：它的大小依赖于 $N$。0.0025 是大还是小？不知道 $N$ 就没法判断。
    - 看 **$N \times$ 误差**：公平硬币 **总是约 0.25**，和 $N$ 无关；不公平的硬币 **随 $N$ 一起变大**。

    所以乘以 $N$ 以后，我们有了一个固定的标准：约 0.25 就是"正常"，远大于 0.25 就是"不公平"。

    #### 0.25/N 是怎么算出来的？

    第 $j$ 次抛硬币，正面记 $Y_j = 1$，反面记 $Y_j = 0$。正面的比例就是平均值：

    $$
    \hat p = \frac{1}{N}\sum_{j=1}^{N} Y_j .
    $$

    **第 1 步：这个"误差"就是方差。** 每个 $\mathbb{E}[Y_j] = 1 \cdot 0.5 + 0 \cdot 0.5 = 0.5$，
    所以 $\mathbb{E}[\hat p] = 0.5$。方差的定义是 $\mathrm{Var}(Z) = \mathbb{E}\big[(Z - \mathbb{E}[Z])^2\big]$，所以

    $$
    \mathbb{E}\big[(\hat p - 0.5)^2\big] = \mathrm{Var}(\hat p) .
    $$

    **第 2 步：一次抛硬币的方差是 0.25。** 用 $\mathrm{Var}(Y) = \mathbb{E}[Y^2] - (\mathbb{E}[Y])^2$。
    $Y$ 只能是 0 或 1，所以 $Y^2 = Y$，$\mathbb{E}[Y^2] = 0.5$。因此

    $$
    \mathrm{Var}(Y_j) = 0.5 - 0.5^2 = 0.25 .
    $$

    **第 3 步：N 次的平均值，方差变成 1/N。** 用方差的两个性质：

    1. 常数提出来要平方：$\mathrm{Var}(cZ) = c^2\, \mathrm{Var}(Z)$。
    2. 独立随机变量的和，方差可以相加。每次抛硬币互相独立，所以可以用。

    $$
    \mathrm{Var}(\hat p)
    = \mathrm{Var}\Big(\frac{1}{N}\sum_j Y_j\Big)
    \overset{\text{性质 1}}{=} \frac{1}{N^2}\, \mathrm{Var}\Big(\sum_j Y_j\Big)
    \overset{\text{性质 2}}{=} \frac{1}{N^2} \cdot N \cdot 0.25
    = \frac{0.25}{N} .
    $$

    分母里的 $N^2$ 来自"平均值的 $1/N$ 要平方"，分子里的 $N$ 来自"$N$ 个方差相加"。
    两者相除，剩下 $1/N$。

    这是一个普遍的规律：**$N$ 个独立样本的平均值，方差是单个样本方差的 $1/N$。**

    #### 回到 SIGReg：完全一样的道理

    | 硬币 | SIGReg |
    |---|---|
    | 一次抛硬币 $Y_j$ | 一个样本的 $e^{itx_j}$ |
    | 正面比例 $\hat p$ | 经验特征函数 $\hat\varphi(t)$ |
    | 公平硬币的 0.5 | N(0,1) 的 $e^{-t^2/2}$ |
    | 误差 $(\hat p - 0.5)^2$ | 加权积分 $I = \int \lvert\hat\varphi - \varphi\rvert^2 w\, dt$ |
    | 单次的方差 0.25 | 单个样本的方差 $1 - e^{-t^2}$ |
    | 抛硬币次数 $N$ | batch size $B$（不是 knots 的个数 17） |

    **同样的 3 步：** 如果样本真的来自 N(0,1)：

    1. $\hat\varphi(t)$ 是平均值，它的期望是 $\varphi(t)$。所以误差 $\mathbb{E}\lvert\hat\varphi(t) - \varphi(t)\rvert^2$ 就是方差。
    2. 单个 $e^{itX}$ 的方差是 $\mathbb{E}\lvert e^{itX}\rvert^2 - \lvert\varphi(t)\rvert^2 = 1 - e^{-t^2}$
       （因为 $\lvert e^{i\theta}\rvert = 1$）。
    3. 平均 $N$ 个样本，方差除以 $N$：

    $$
    \mathbb{E}\,\big|\hat\varphi(t) - \varphi(t)\big|^2 = \frac{1 - e^{-t^2}}{N} .
    $$

    再乘以权重、在 $t$ 上积分：

    $$
    \mathbb{E}[I] = \frac{1}{N}\int_{-3}^{3} (1 - e^{-t^2})\, e^{-t^2/2}\, dt \approx \frac{1.05}{N} .
    $$

    所以 1.05 在 SIGReg 里的作用，就和 0.25 在硬币例子里的作用一样。

    **一般情况下**，积分 $I$ 由两部分组成：

    $$
    I \;\approx\; \underbrace{D}_{\text{真实的差距}} \;+\; \underbrace{\frac{1.05}{N}}_{\text{抽样噪声}}
    \qquad\Longrightarrow\qquad
    T = N \cdot I \;\approx\; N \cdot D + 1.05 .
    $$

    其中 $D$ 是样本的真实分布和 N(0,1) 的差距。

    - **样本是 N(0,1)**（$D = 0$）：$T \approx 1.05$，**不管 $N$ 是多少**。
      这是 loss 的"底"，SIGReg loss 不会降到 0。
    - **样本不是 N(0,1)**（$D > 0$）：$T \approx N \cdot D + 1.05$，**随 $N$ 线性变大**。

    #### 对训练有什么影响？

    - **在训练中，$B$ 是固定的**（`batch_size: 128`）。所以乘以 $B$ 只是把 loss 乘以一个常数。
      它不改变"什么样的 embedding 最好"，只改变 SIGReg loss 的大小，等价于调整 $\lambda$。
    - 乘以 $N$ 的真正好处是 **loss 的数值有了意义**：看到 SIGReg loss ≈ 1，
      就知道 embedding 已经和高斯分布分不出来了；看到 50，就知道还差得很远。这对任何 batch size 都成立。
    - **一个实际后果：** 如果把 batch size 从 128 改成 512，在 embedding 不是高斯的时候，
      SIGReg loss 会变大约 4 倍，相当于 $\lambda$ 变大了 4 倍。所以改 batch size 时，可能要重新调 $\lambda$。

    下面的实验验证这一点。每个点是 20 次重复的平均值。
    """)
    return


@app.cell
def _(
    DISTS: dict[str, str],
    epps_pulley_1d,
    mo,
    np,
    plt,
    sample_1d,
    sigreg_weights,
):
    _tk, _wk = sigreg_weights(17)
    null_mean = float(((1 - np.exp(-_tk**2)) @ _wk))

    _Ns = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    _rng = np.random.default_rng(1)
    _rows = []
    _fig, _ax = plt.subplots(figsize=(7, 4))
    for _label, _key in DISTS.items():
        _means = [
            np.mean([epps_pulley_1d(sample_1d(_key, _n, _rng)) for _ in range(20)])
            for _n in _Ns
        ]
        _ax.plot(_Ns, _means, "o-", label=_key)
        _rows.append({"分布": _label, **{f"N={_n}": round(_m, 2) for _n, _m in zip(_Ns, _means) if _n in (64, 512, 4096)}})
    _ax.axhline(null_mean, color="k", ls="--", lw=1, label=f"theory for N(0,1): {null_mean:.3f}")
    _ax.set_xscale("log", base=2)
    _ax.set_yscale("log")
    _ax.set_xlabel("N (batch size)")
    _ax.set_ylabel("mean T")
    _ax.set_title("Epps-Pulley T vs sample size")
    _ax.legend(fontsize=8)
    _fig.tight_layout()

    mo.vstack([
        _fig,
        mo.md(f"理论值（用 17 个 knots 的梯形法则计算）：$\\mathbb{{E}}[T] = {null_mean:.4f}$"),
        mo.ui.table(_rows, selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **观察：**

    - 正态分布的线是平的，在理论值附近。
    - 其他分布的线是斜率为 1 的直线（log-log 图），也就是 $T \propto N$。
    - 双峰、均匀分布最容易检测。拉普拉斯、Student-t 要更大的 $N$ 才能和正态区分开。

    ## 3. 从 1 维到高维：随机投影

    embedding 是 $D$ 维的（例如 $D = 192$）。直接在 $D$ 维上比较特征函数很难：
    $t$ 变成了 $D$ 维向量，积分要在 $D$ 维空间里做。

    **Cramér–Wold 定理** 给出了一个简单的办法：

    > 如果对 **每个** 单位方向 $a$，投影 $a^\top z$ 都服从 N(0,1)，
    > 那么 $z$ 服从 N(0, I)。

    所以 SIGReg 的做法是：

    1. 随机取很多（`num_proj = 1024`）单位方向 $a_1, \dots, a_M$。
    2. 对每个方向，把 $B$ 个 embedding 投影成 $B$ 个数字。
    3. 对每个方向计算 1 维的 Epps–Pulley 统计量 $T(a_k)$。
    4. 取平均值：$\text{SIGReg} = \frac{1}{M}\sum_k T(a_k)$。

    **每一步训练都重新取随机方向。** 所以经过很多步，所有方向都会被检查到。
    这就是名字里 "Sketch" 的意思：每次只看一个随机的"草图"。

    下面用 2 维的例子看看。选择一个点云，然后旋转投影方向 θ。
    右图是 $T(\theta)$ 在所有方向上的曲线。SIGReg 估计的就是这条曲线的平均值。
    """)
    return


@app.cell
def _(mo):
    CLOUDS: dict[str, str] = {
        "各向同性高斯 N(0, I)": "isotropic",
        "拉长的高斯（方差 2.0 / 0.3，旋转 30°）": "elongated",
        "坍缩到一条直线（30° 方向）": "line",
        "四个簇": "clusters",
        "均匀正方形（每个坐标方差 1）": "square",
    }
    cloud_dd = mo.ui.dropdown(options=CLOUDS, value="坍缩到一条直线（30° 方向）", label="点云")
    theta_sl = mo.ui.slider(0, 179, step=1, value=0, label="投影方向 θ (度)", show_value=True)
    mo.hstack([cloud_dd, theta_sl], wrap=True)
    return cloud_dd, theta_sl


@app.cell
def _(np):
    def rot(deg: float) -> np.ndarray:
        r = np.deg2rad(deg)
        return np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])

    def cloud_2d(name: str, n: int, rng: np.random.Generator) -> np.ndarray:
        """Return an (n, 2) point cloud."""
        if name == "isotropic":
            return rng.standard_normal((n, 2))
        if name == "elongated":
            return (rng.standard_normal((n, 2)) * np.sqrt([2.0, 0.3])) @ rot(30).T
        if name == "line":
            u = rng.standard_normal(n)
            return np.outer(u, [np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))])
        if name == "clusters":
            signs = rng.choice([-1.0, 1.0], (n, 2))
            return 0.9 * signs + np.sqrt(0.19) * rng.standard_normal((n, 2))
        if name == "square":
            return rng.uniform(-np.sqrt(3), np.sqrt(3), (n, 2))
        raise ValueError(name)

    return (cloud_2d,)


@app.cell
def _(cloud_2d, cloud_dd, epps_pulley_1d, np, plt, theta_sl):
    _rng = np.random.default_rng(2)
    _z = cloud_2d(cloud_dd.value, 512, _rng)
    _th = np.deg2rad(theta_sl.value)
    _a = np.array([np.cos(_th), np.sin(_th)])
    _p = _z @ _a

    _angles = np.arange(180)
    _Ts = np.array([
        epps_pulley_1d(_z @ np.array([np.cos(np.deg2rad(_d)), np.sin(np.deg2rad(_d))]))
        for _d in _angles
    ])

    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4))
    _ax = _axes[0]
    _ax.scatter(_z[:, 0], _z[:, 1], s=4, alpha=0.4)
    _ax.plot([-4 * _a[0], 4 * _a[0]], [-4 * _a[1], 4 * _a[1]], "r-", lw=1)
    _ax.annotate("", xy=2.5 * _a, xytext=(0, 0), arrowprops=dict(color="r", width=2, headwidth=8))
    _ax.set_xlim(-4, 4)
    _ax.set_ylim(-4, 4)
    _ax.set_aspect("equal")
    _ax.set_title(f"Point cloud and direction a (theta={theta_sl.value} deg)")

    _ax = _axes[1]
    _ax.hist(np.clip(_p, -5, 5), bins=np.linspace(-5, 5, 61), density=True, alpha=0.6)
    _g = np.linspace(-5, 5, 300)
    _ax.plot(_g, np.exp(-_g**2 / 2) / np.sqrt(2 * np.pi), "k--", label="N(0,1)")
    _ax.set_title(f"Projection a^T z:  T = {epps_pulley_1d(_p):.2f}")
    _ax.legend(fontsize=8)

    _ax = _axes[2]
    _ax.plot(_angles, _Ts)
    _ax.axvline(theta_sl.value, color="r", lw=1)
    _ax.axhline(_Ts.mean(), color="k", ls="--", lw=1, label=f"mean over directions = {_Ts.mean():.2f}")
    _ax.set_yscale("log")
    _ax.set_xlabel("theta (deg)")
    _ax.set_title("T for every direction")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **试一试：**

    - **坍缩到一条直线**：把 θ 调到 30°。投影正好是 N(0,1)，$T$ 很小！
      但是把 θ 调到 120°（垂直方向），所有点都投影到 0，$T$ 非常大。
      这说明：**只检查一个方向是不够的**。只要有一个方向不对，平均值就会变大。
    - **四个簇**：沿坐标轴（0°、90°）投影是双峰，$T$ 大。
      沿对角线（45°）投影有 3 个峰，也不像正态。
    - **均匀正方形**：每个方向的投影都不是正态，但是在对角线方向（两个均匀分布的和）更接近正态。
    - **各向同性高斯**：所有方向的 $T$ 都在 1 附近，曲线是平的。

    在高维中，随机方向几乎不会正好对准"坏"的方向。所以 SIGReg 需要很多方向（1024 个），
    并且每一步都换新的方向。第 6 节会讨论这个问题。

    ## 4. 逐行读代码

    现在你已经知道所有概念。下面是 `module.py` 里的完整代码：
    """)
    return


@app.cell
def _(SIGReg, inspect, mo):
    mo.md(f"""
    ```python\n{inspect.getsource(SIGReg)}\n```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.1 `__init__`：knots 和权重

    下表是 `knots=17` 时的值。最后一行检查：所有权重的和应该约等于
    $\int_{-3}^{3} e^{-t^2/2}\,dt \approx \sqrt{2\pi} \approx 2.5066$（因为这就是对常数 1 积分）。
    """)
    return


@app.cell
def _(SIGReg, mo, np):
    _s = SIGReg(knots=17, num_proj=8)
    _t = _s.t.numpy()
    _dt = 3 / 16
    _trap = np.full(17, 2 * _dt)
    _trap[[0, -1]] = _dt
    _rows = [
        {
            "i": _i,
            "t": round(float(_t[_i]), 4),
            "梯形权重 (含因子2)": round(float(_trap[_i]), 4),
            "window = exp(-t²/2)": round(float(_s.phi[_i]), 4),
            "weights (最终)": round(float(_s.weights[_i]), 5),
        }
        for _i in range(17)
    ]
    mo.vstack([
        mo.ui.table(_rows, selection=None, pagination=False),
        mo.md(f"`weights.sum()` = **{float(_s.weights.sum()):.4f}**，"
              f"$\\sqrt{{2\\pi}}$ = {np.sqrt(2 * np.pi):.4f}"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.2 `forward`：每一步的 shape

    输入 `proj` 的 shape 是 `(T, B, D)`：T 个时间步，B 个样本，D 维。
    `train.py` 传入的是 `emb.transpose(0, 1)`，所以 **每个时间步单独检验**，
    在 batch 维上求经验特征函数。

    下面用一个小例子运行每一行，并打印 shape：T=2, B=256, D=8, num_proj=16, knots=17。
    为了得到和 `SIGReg` 相同的随机方向 `A`，我们在两边用同一个随机种子。
    """)
    return


@app.cell
def _(SIGReg, mo, torch):
    _T, _B, _D, _M = 2, 256, 8, 16
    _gen = torch.Generator().manual_seed(0)
    # Time step 0: Gaussian. Time step 1: Gaussian scaled by 2 (too wide).
    proj_demo = torch.randn(_T, _B, _D, generator=_gen)
    proj_demo[1] *= 2.0
    sig_demo = SIGReg(knots=17, num_proj=_M)

    # Line by line, same as SIGReg.forward
    torch.manual_seed(123)
    A_demo = torch.randn(proj_demo.size(-1), sig_demo.num_proj)
    A_demo = A_demo.div_(A_demo.norm(p=2, dim=0))
    _p = proj_demo @ A_demo
    _x_t = _p.unsqueeze(-1) * sig_demo.t
    _cos_mean = _x_t.cos().mean(-3)
    _sin_mean = _x_t.sin().mean(-3)
    _err = (_cos_mean - sig_demo.phi).square() + _sin_mean.square()
    _stat = (_err @ sig_demo.weights) * proj_demo.size(-2)
    manual_out = _stat.mean()

    # Official forward with the same seed
    torch.manual_seed(123)
    official_out = sig_demo(proj_demo)

    _rows = [
        ("proj", tuple(proj_demo.shape), "(T, B, D) 输入"),
        ("A", tuple(A_demo.shape), "(D, M) M 个随机单位方向（每列范数为 1）"),
        ("proj @ A", tuple(_p.shape), "(T, B, M) 每个样本在每个方向上的投影"),
        ("x_t = (proj @ A).unsqueeze(-1) * t", tuple(_x_t.shape), "(T, B, M, K) 每个投影乘以每个 knot t"),
        ("x_t.cos().mean(-3)", tuple(_cos_mean.shape), "(T, M, K) 对 batch 求平均 → ECF 实部"),
        ("x_t.sin().mean(-3)", tuple(_sin_mean.shape), "(T, M, K) 对 batch 求平均 → ECF 虚部"),
        ("err", tuple(_err.shape), "(T, M, K) |ECF − exp(−t²/2)|²"),
        ("(err @ weights) * B", tuple(_stat.shape), "(T, M) 每个时间步、每个方向的 Epps–Pulley T"),
        ("statistic.mean()", tuple(manual_out.shape), "() 对时间步和方向求平均 → 标量 loss"),
    ]
    _table = mo.ui.table(
        [{"表达式": _e, "shape": str(_s), "含义": _m} for _e, _s, _m in _rows],
        selection=None,
        pagination=False,
    )
    _per_t = _stat.mean(-1)
    mo.vstack([
        _table,
        mo.md(
            f"每个时间步的平均 T：t=0（高斯）= **{_per_t[0]:.3f}**，"
            f"t=1（高斯×2）= **{_per_t[1]:.3f}**\n\n"
            f"逐行计算结果 = `{manual_out.item():.6f}`，`SIGReg.forward` 结果 = `{official_out.item():.6f}`"
        ),
    ])
    return A_demo, official_out, proj_demo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.3 用 numpy 和 1 维函数复算

    最后一次检查：用第 2 节的 `epps_pulley_1d`（纯 numpy，一次只处理一个方向），
    对每个时间步、每个方向计算 $T$，再取平均值。结果应该和 `SIGReg` 相同。
    这证明 **SIGReg = 在随机方向上的 1 维 Epps–Pulley 统计量的平均值**，没有别的东西。
    """)
    return


@app.cell
def _(A_demo, epps_pulley_1d, mo, np, official_out, proj_demo):
    _P = proj_demo.numpy().astype(np.float64)
    _A = A_demo.numpy().astype(np.float64)
    _vals = [
        epps_pulley_1d(_P[_ti] @ _A[:, _k])
        for _ti in range(_P.shape[0])
        for _k in range(_A.shape[1])
    ]
    numpy_out = float(np.mean(_vals))
    assert np.isclose(numpy_out, official_out.item(), rtol=1e-4), (numpy_out, official_out.item())
    mo.md(f"numpy 复算 = `{numpy_out:.6f}`，SIGReg = `{official_out.item():.6f}` ✅ 一致")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. 作为 loss：梯度会做什么？

    SIGReg 是可导的（只有 matmul、cos、sin、mean），所以可以直接作为 loss 反向传播。

    下面的实验：**不用 encoder**，直接把 512 个 2 维点当作参数，只用 SIGReg 作为 loss，
    用 Adam 优化。每一步都有新的随机方向（num_proj=64）。
    看看点云会怎么变化。
    """)
    return


@app.cell
def _(mo):
    INITS: dict[str, str] = {
        "几乎坍缩到一点 (中心 (2,-1), std 0.05)": "point",
        "坍缩到一条直线": "line",
        "拉长的高斯": "elongated",
        "四个簇": "clusters",
    }
    init_dd = mo.ui.dropdown(options=INITS, value="几乎坍缩到一点 (中心 (2,-1), std 0.05)", label="初始点云")
    steps_sl = mo.ui.slider(50, 1500, step=50, value=600, label="步数", show_value=True)
    lr_sl = mo.ui.dropdown(options={"0.003": 0.003, "0.01": 0.01, "0.03": 0.03, "0.1": 0.1}, value="0.03", label="学习率")
    mo.hstack([init_dd, steps_sl, lr_sl], wrap=True)
    return init_dd, lr_sl, steps_sl


@app.cell
def _(SIGReg, cloud_2d, init_dd, lr_sl, np, plt, steps_sl, torch):
    def make_init(name: str, n: int) -> torch.Tensor:
        rng = np.random.default_rng(3)
        if name == "point":
            z = np.array([2.0, -1.0]) + 0.05 * rng.standard_normal((n, 2))
        elif name == "line":
            u = rng.standard_normal(n)
            z = np.outer(u, [np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))])
            z += 0.02 * rng.standard_normal((n, 2))
        else:
            z = cloud_2d(name, n, rng)
        return torch.tensor(z, dtype=torch.float32)

    torch.manual_seed(0)
    _z0 = make_init(init_dd.value, 512)
    _z = _z0.clone().requires_grad_(True)
    _opt = torch.optim.Adam([_z], lr=lr_sl.value)
    _reg = SIGReg(knots=17, num_proj=64)
    _n_steps = steps_sl.value
    _snap_at = [0, _n_steps // 10, _n_steps // 3, _n_steps]
    _snaps = {0: _z0.numpy().copy()}
    _losses = []
    for _i in range(1, _n_steps + 1):
        _loss = _reg(_z.unsqueeze(0))  # (T=1, B=512, D=2)
        _opt.zero_grad()
        _loss.backward()
        _opt.step()
        _losses.append(_loss.item())
        if _i in _snap_at:
            _snaps[_i] = _z.detach().numpy().copy()

    _fig, _axes = plt.subplots(1, 5, figsize=(17, 3.6))
    for _ax, _s in zip(_axes[:4], _snap_at):
        _pts = _snaps[_s]
        _ax.scatter(_pts[:, 0], _pts[:, 1], s=3, alpha=0.4)
        _ax.set_xlim(-4, 4)
        _ax.set_ylim(-4, 4)
        _ax.set_aspect("equal")
        _ax.set_title(f"step {_s}")
    _axes[4].plot(_losses, lw=0.8)
    _axes[4].axhline(1.05, color="k", ls="--", lw=1, label="random N(0,1) sample ~1.05")
    _axes[4].set_yscale("log")
    _axes[4].set_xlabel("step")
    _axes[4].set_title("SIGReg loss")
    _axes[4].legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **观察：**

    - 不管从哪里开始，点云都会被推开、移动到原点、变成圆形的高斯云。
    - loss 先快速下降，然后在很小的值附近波动。波动来自每一步不同的随机方向。
    - **注意：** loss 最后会比 1.05（随机 N(0,1) 样本的水平，见 2.1 节）还低。
      原因：这里的点是可以自由移动的参数，它们可以排列得比真正的随机样本"更像正态"（没有抽样噪声）。
      实际训练中，每个 batch 都是新的图像，encoder 不能为每个 batch 单独排列点，
      所以真实训练中的 SIGReg loss 一般在 1 附近或更高。
    - 学习率 0.01、600 步时还没有收敛。试试增加步数，看看要多久。
    - SIGReg **没有** 显式的"均值 = 0"或"方差 = 1"的项。这些都是"像 N(0,1)"的结果。

    ### 5.1 一个细节：完全坍缩时梯度没有用

    如果所有点 **完全相同**，那么每个点得到的梯度也完全相同。
    所有点会一起移动，不会分开。下面先验证一步的梯度：
    """)
    return


@app.cell
def _(SIGReg, mo, torch):
    torch.manual_seed(0)
    _z = torch.full((256, 2), 0.5, requires_grad=True)
    _loss = SIGReg(num_proj=64)(_z.unsqueeze(0))
    _loss.backward()
    _spread = (_z.grad - _z.grad.mean(0)).abs().max().item()
    mo.md(
        f"loss = {_loss.item():.1f}。不同点之间梯度的最大差异 = **{_spread:.2e}**（约等于 0，只是浮点误差）。"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **多做一些 step 会怎样？** 下面从同一个点 (0.5, 0.5) 开始，只用 SIGReg 训练 1000 步，比较三种起点：

    - **噪声 = 0**：256 个点完全相同。
    - **噪声 = 1e-6**：每个点加一点点噪声，小到肉眼看不出来。
    - **噪声 = 1e-3**：噪声稍大一点。

    "不同的点数"是指坐标 **完全相同** 的点算作一个，一共有几个不同的位置。
    """)
    return


@app.cell
def _(SIGReg, mo, np, plt, torch):
    def run_from_point(noise: float, steps: int = 1000) -> tuple[list[float], list[int], np.ndarray]:
        """Train 256 points that start at (0.5, 0.5) with SIGReg only."""
        torch.manual_seed(0)
        z = torch.full((256, 2), 0.5) + noise * torch.randn(256, 2)
        z.requires_grad_(True)
        opt = torch.optim.Adam([z], lr=0.03)
        reg = SIGReg(num_proj=64)
        losses: list[float] = []
        n_distinct: list[int] = []
        for _ in range(steps):
            loss = reg(z.unsqueeze(0))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            n_distinct.append(len(torch.unique(z.detach(), dim=0)))
        return losses, n_distinct, z.detach().numpy()

    _runs = {_noise: run_from_point(_noise) for _noise in [0.0, 1e-6, 1e-3]}

    _fig, _axes = plt.subplots(1, 4, figsize=(17, 3.6))
    for _noise, (_losses, _nd, _zf) in _runs.items():
        _axes[0].plot(_losses, lw=0.8, label=f"noise={_noise:g}")
        _axes[1].plot(_nd, lw=1, label=f"noise={_noise:g}")
    _axes[0].set_yscale("log")
    _axes[0].set_xlabel("step")
    _axes[0].set_title("SIGReg loss")
    _axes[0].legend(fontsize=8)
    _axes[1].set_yscale("log")
    _axes[1].set_xlabel("step")
    _axes[1].set_title("Number of distinct points")
    _axes[1].legend(fontsize=8)
    for _ax, _noise in zip(_axes[2:], [0.0, 1e-6]):
        _zf = _runs[_noise][2]
        _ax.scatter(_zf[:, 0], _zf[:, 1], s=6, alpha=0.4)
        _ax.set_xlim(-4, 4)
        _ax.set_ylim(-4, 4)
        _ax.set_aspect("equal")
        _ax.set_title(f"noise={_noise:g}: after 1000 steps")
    _fig.tight_layout()

    _rows = []
    for _noise, (_losses, _nd, _zf) in _runs.items():
        _pts, _counts = np.unique(_zf, axis=0, return_counts=True)
        _rows.append({
            "噪声": f"{_noise:g}",
            "最后 50 步的平均 loss": round(float(np.mean(_losses[-50:])), 2),
            "不同的点数": _nd[-1],
            "最大的几组（每组的点数）": str(sorted(_counts.tolist(), reverse=True)[:5]),
        })
    mo.vstack([_fig, mo.ui.table(_rows, selection=None, pagination=False)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **观察：**

    - **噪声 = 1e-6 和 1e-3**：一开始所有点几乎重合，loss 很大。但是点之间的微小差别会被梯度放大，
      几百步以后，点云变成了 N(0, I)，loss 降到接近 0。噪声越大，分开得越快。
    - **噪声 = 0**：多做 step 也没有用。256 个点一开始完全相同，梯度也完全相同，所以它们一起移动。
      过了几十步，**浮点舍入误差** 让点分成了 **2 组**（在写这个 notebook 时的测试中，是 252 个点和 4 个点。
      这和 CPU 怎样分批计算有关，在别的机器上可能不同）。
      但是每一组里面的点仍然 **完全相同**，梯度也完全相同，所以组内的点永远分不开。
      最后只有 2 个不同的位置，loss 停在一个很大的值，不再下降。

    **结论：** SIGReg 本身不能「从一个点里」把点分开。只要有一点点差别，它就能把点推开；
    但是完全相同的点，多少步都没有用。

    实际训练中这不是问题：不同的图像输入会给出不同的 embedding，梯度通过 encoder 传回，
    SIGReg 会阻止 encoder 走向坍缩。第 5 节的实验也加了一点噪声（std 0.05），所以可以分开。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. 超参数的直觉

    ### 6.1 `num_proj`：方向越多，估计越稳定

    SIGReg 是"在所有方向上的平均 $T$"的 **随机估计**。方向越少，估计的方差越大。

    下面在 $D = 64$、$B = 256$ 的数据上，对每个 `num_proj` 重复计算 20 次（每次不同的随机方向），
    画平均值和标准差。三种数据：

    - **各向同性高斯**：理想情况。
    - **第 0 维 × 3**：只有一个方向的方差是 9，其他 63 个方向正确。
    - **坍缩到 4 维子空间**：embedding 只用了 64 维中的 4 维。

    **重要观察：** "第 0 维 × 3" 的 SIGReg 值只比高斯大一点点。原因：
    在 64 维中，随机单位方向 $a$ 在第 0 维上的分量 $a_0^2 \approx 1/64$，
    所以投影的方差只是 $1 + 8a_0^2 \approx 1.13$。**高维中只有少数方向有问题时，随机投影很难发现。**
    但是因为每一步都换方向，而且 $T$ 乘以了 $B$，经过很多步之后梯度仍然会修正它。
    """)
    return


@app.cell
def _(SIGReg, mo, np, plt, torch):
    _D, _B, _reps = 64, 256, 20
    _gen = torch.Generator().manual_seed(5)
    _iso = torch.randn(1, _B, _D, generator=_gen)
    _dim0 = _iso.clone()
    _dim0[..., 0] *= 3.0
    _Q, _ = torch.linalg.qr(torch.randn(_D, 4, generator=_gen))
    _low = torch.randn(1, _B, 4, generator=_gen) @ _Q.T
    _datasets = {"isotropic N(0,I)": _iso, "dim 0 x3": _dim0, "rank-4 subspace": _low}

    _Ms = [8, 32, 128, 512, 1024]
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 3.8))
    _rows = []
    torch.manual_seed(0)
    with torch.no_grad():
        for _name, _x in _datasets.items():
            _mean, _std = [], []
            for _m in _Ms:
                _reg = SIGReg(num_proj=_m)
                _v = np.array([_reg(_x).item() for _ in range(_reps)])
                _mean.append(_v.mean())
                _std.append(_v.std())
            _mean, _std = np.array(_mean), np.array(_std)
            _axes[0].errorbar(_Ms, _mean, yerr=_std, fmt="o-", capsize=3, label=_name)
            _axes[1].plot(_Ms, _std / _mean, "o-", label=_name)
            _rows.append({"数据": _name, **{f"M={_m}": f"{_a:.2f} ± {_b:.2f}" for _m, _a, _b in zip(_Ms, _mean, _std)}})
    _axes[0].set_xscale("log", base=2)
    _axes[0].set_yscale("log")
    _axes[0].set_xlabel("num_proj")
    _axes[0].set_title("SIGReg value (mean +- std over 20 draws)")
    _axes[0].legend(fontsize=8)
    _axes[1].set_xscale("log", base=2)
    _axes[1].set_xlabel("num_proj")
    _axes[1].set_title("Relative std (std / mean)")
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.ui.table(_rows, selection=None)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.2 `knots`：影响很小

    `knots` 只决定数值积分的精度。被积函数很光滑，所以 17 个点已经足够。
    下面用同一组随机方向，比较不同 `knots` 的结果：
    """)
    return


@app.cell
def _(SIGReg, mo, torch):
    _gen = torch.Generator().manual_seed(7)
    _x = torch.randn(1, 256, 64, generator=_gen)
    _x[..., :8] *= 1.5
    _rows = []
    with torch.no_grad():
        for _k in [5, 9, 17, 33, 65, 129]:
            torch.manual_seed(0)  # same random directions for every knots value
            _rows.append({"knots": _k, "SIGReg": round(SIGReg(knots=_k, num_proj=1024)(_x).item(), 5)})
    mo.ui.table(_rows, selection=None, pagination=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. 总结

    | 概念 | 一句话 | 代码 |
    |---|---|---|
    | 特征函数 | 分布的"指纹"：$\mathbb{E}[\cos tX] + i\,\mathbb{E}[\sin tX]$；N(0,1) 时是 $e^{-t^2/2}$ | `self.phi` |
    | 经验特征函数 | 用 batch 平均代替期望 | `x_t.cos().mean(-3)`, `x_t.sin().mean(-3)` |
    | Epps–Pulley | $N\int |\hat\varphi-\varphi|^2 e^{-t^2/2}dt$，梯形法则，$t\in[0,3]$ 利用对称性 | `err @ self.weights * B` |
    | 随机投影 | Cramér–Wold：每个 1 维投影都是 N(0,1) ⇒ 整体是 N(0, I) | `A = randn(D, M)`, `proj @ A` |
    | 平均 | 对方向和时间步取平均 | `statistic.mean()` |

    **常见问题：**

    - **SIGReg 的最小值是多少？** 不是 0。真正的 N(0, I) 样本给出约 1.05（有限样本噪声）。
    - **为什么不用 VICReg 那样的方差/协方差项？** VICReg 只约束前两阶矩（均值、协方差）。
      第 1 节的"均匀"和"双峰"例子说明：前两阶矩正确，分布仍然可以不是高斯。
      SIGReg 比较的是 **整个分布**。
    - **为什么是高斯？** LeJEPA 论文的论证是：在下游线性探测任务上，各向同性高斯是使最坏情况风险最小的 embedding 分布。
      从实用角度看：它是"信息分布得最均匀、没有坍缩"的分布。
    - **λ = 0.09 的含义？** SIGReg 值随 batch size $B$ 线性增长（非高斯时）。
      所以改变 batch size 等于改变 SIGReg 的有效权重。改 batch size 时要注意。
    - **"single-GPU!" 是什么意思？** 经验特征函数是在本 GPU 的 batch 上求平均的。
      多 GPU 时，每个 GPU 只看到自己的那部分 batch，除非先做 all-reduce。
    """)
    return


if __name__ == "__main__":
    app.run()
