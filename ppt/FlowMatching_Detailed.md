# 第四代：Flow Matching (Conditional Flow Matching)

> **论文：** Flow Matching for Generative Modeling (Lipman et al., 2022)
>
> **地位：** 当前最主流的生成模型训练范式，被 Stable Diffusion 3、MovieGen 等顶级产品采用

---

## 一、核心思想

### 1.1 从"预测噪声"到"回归向量场"

```
前三代:                              Flow Matching:
┌──────────────────────┐            ┌──────────────────────┐
│ 学什么?              │            │ 学什么?              │
│ → 预测噪声 ε         │   范式转换   │ → 回归向量场 u        │
│                      │    ──→      │                      │
│ 损失函数:             │            │ 损失函数:             │
│ L = ‖ε - εθ(x,t)‖²  │            │ L = ‖u - u^{target}‖²│
│                      │            │                      │
│ 需要设计:             │            │ 需要设计:             │
│ β_t 或 g(t)          │            │ 无！                 │
│ (噪声调度)           │            │ (自动确定)           │
└──────────────────────┘            └──────────────────────┘
 复杂的超参数选择                    简洁的回归问题
```

### 1.2 核心直觉

**Flow Matching 的训练过程就像教一个学生走迷宫：**

```
目标数据 z（如一张猫图）     噪声 x₀（起点）
    ● ─────────────────────────── ●
    
    ↑ 我们告诉学生："从 x 出发，朝 z 的方向走"
    
    学生学到的就是向量场 u_t(x)
    
    训练时：随机选一个中间点 x = t·z + (1-t)·x₀
            让学生预测方向 u_t(x)
            真实方向是 u^{target} = z - x₀
            损失 = ‖预测方向 - 真实方向‖²
```

---

## 二、数学框架

### 2.1 条件概率路径

**定义：** 一个随时间 $t \in [0,1]$ 变化的概率分布族 $p_t(x|z)$，满足：

| 时间 | 分布 | 含义 |
|------|------|------|
| $t=0$ | $p_0(x|z) = p_{\text{simple}}(x)$ | 从简单分布出发（如高斯） |
| $t=1$ | $p_1(x|z) \approx \delta_z(x)$ | 收敛到目标数据点 $z$ |
| 中间 | 平滑插值 | 连续过渡 |

#### 高斯条件概率路径（最常用）

$$p_t(x|z) = \mathcal{N}(x; \mu_t(z), \Sigma_t)$$

最简单的线性选择：

$$\mu_t(z) = tz + (1-t)\mu_0, \quad \Sigma_t = \sigma^2 I$$

其中 $\mu_0$ 是简单分布的均值（通常为 0），$\sigma$ 是固定常数。

**参数详解：**

| 参数 | 类型 | 说明 |
|------|------|------|
| $z$ | 数据向量 | 目标数据点（条件变量） |
| $x$ | 向量 | 当前位置 |
| $t \in [0,1]$ | 标量 | 时间参数 |
| $\mu_0$ | 向量 | 简单分布的均值（通常 = 0） |
| $\sigma$ | 标量 | 固定的噪声标准差 |

### 2.2 条件向量场（解析可得！）

**这是 Flow Matching 最优雅的地方——向量场可以直接算出来，不需要学习！**

对于线性概率路径 $p_t(x|z) = \mathcal{N}(tz + (1-t)\mu_0, \sigma^2 I)$：

**推导：**

考虑 ODE：
$$\frac{\mathrm{d}}{\mathrm{d}t}X_t = u_t(X_t|z), \quad X_0 \sim p_{\text{simple}}$$

如果 $X_t$ 的分布恰好是 $p_t(\cdot|z)$，则：

$$X_t = t \cdot z + (1-t) \cdot X_0$$

对时间求导：

$$\frac{\mathrm{d}X_t}{\mathrm{d}t} = z - X_0$$

但我们需要用 $(X_t, t)$ 表示。由 $X_t = tz + (1-t)X_0$ 得到 $X_0 = \frac{X_t - tz}{1-t}$，代入：

$$\boxed{u_t^{\text{target}}(x|z) = \frac{z - x}{1-t}}$$

**验证：**

| 性质 | 验证 |
|------|------|
| $t=0$ 时 | $u_0(x\|z) = z - x$（把粒子推向 $z$） |
| $t=1$ 时 | 极限存在（实际使用时 $t < 1$） |
| 方向正确 | 当 $x$ 在 $z$ 左边时推力向右，反之亦然 |

### 2.3 边际向量场与边际分布

**边际分布：**
$$p_t(x) = \int p_t(x|z)p_{\text{data}}(z)\,\mathrm{d}z$$

**边际向量场：**
$$u_t^{\text{target}}(x) = \int u_t^{\text{target}}(x|z)\,\frac{p_t(x|z)p_{\text{data}}(z)}{p_t(x)}\,\mathrm{d}z$$

> ⚠️ **边际向量场无法直接计算**（积分涉及未知的 $p_t(x)$）。但这正是 Flow Matching 的精妙之处——我们不需要它！

---

## 三、训练方法

### 3.1 条件 Flow Matching 损失

**核心定理：最小化条件损失等价于最小化边际损失**

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t,z,x}\left[\|u_t^\theta(x) - u_t^{\text{target}}(x|z)\|^2\right]$$

其中采样过程为：
1. 采样 $z \sim p_{\text{data}}$
2. 采样 $t \sim \text{Uniform}[0,1]$
3. 采样 $x_0 \sim p_{\text{simple}}$
4. 计算 $x = \mu_t(z) + \Sigma_t^{1/2} \epsilon$（或直接 $x = tz + (1-t)x_0$）

### 3.2 直线调度（Straight Line Schedule）的简化形式

令 $\alpha_t = t$, $\beta_t = 1-t$, $\mu_0 = 0$, $\sigma = 1$：

$$\begin{aligned}
x &= tz + (1-t)x_0 = tz + (1-t)\epsilon \quad (\text{设 } x_0 = \epsilon \sim \mathcal{N}(0,I)) \\
u_t^{\text{target}}(x|z) &= \frac{z - x}{1-t} = \frac{z - (tz + (1-t)\epsilon)}{1-t} = z - \epsilon
\end{aligned}$$

**最终损失函数（极其简洁）：**

$$\boxed{\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t,z,\epsilon}\left[\left\|u_t^\theta(tz + (1-t)\epsilon) - (z - \epsilon)\right\|^2\right]}$$

| 公式部分 | 含义 |
|----------|------|
| $tz + (1-t)\epsilon$ | **输入**：噪声和数据的线性插值 |
| $u_t^\theta(\cdot)$ | **神经网络输出**：预测的向量场 |
| $z - \epsilon$ | **目标**：真实方向（数据减去噪声） |

### 3.3 完整训练算法

**算法：Conditional Flow Matching 训练**

| 步骤 | 操作 |
|------|------|
| **输入** | 数据集 $\{z_i\}$，网络 $u_t^\theta$，迭代次数 $M$ |
| **循环** | 对每个 epoch / mini-batch： |
| ① | 采样 $z \sim p_{\text{data}}$（从数据集中取一个样本） |
| ② | 采样 $t \sim \text{Uniform}[0,1]$（均匀采一个时间） |
| ③ | 采样 $\epsilon \sim \mathcal{N}(0, I_d)$（标准高斯噪声） |
| ④ | 构造混合样本 $x = tz + (1-t)\epsilon$ |
| ⑤ | 计算损失 $\mathcal{L} = \|u_t^\theta(x, t) - (z - \epsilon)\|^2$ |
| ⑥ | 反向传播更新参数 $\theta$ |

> 🎯 **这就是全部了！没有 $\beta_t$、没有 $\bar{\alpha}_t$、没有复杂的辅助变量。就是一个普通的回归问题。**

---

## 四、与前几代的详细对比

### 4.1 公式对比总表

| 项目 | DDPM | DDIM | Score SDE | **Flow Matching** |
|------|------|------|-----------|-------------------|
| **年份** | 2020 | 2021 | 2021 | **2022+** |
| **数学框架** | 离散马尔可夫链 | 离散非马尔可夫 | 连续 SDE/ODE | **连续 ODE** |
| **前向过程** | $q(x_t\|x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)$ | 同左 | $\mathrm{d}x = f\,\mathrm{d}t + g\,\mathrm{d}W_t$ | $p_t(x\|z) = \mathcal{N}(tz+(1-t)x_0, \sigma^2 I)$ |
| **学习目标** | 噪声 $\epsilon_\theta$ | 噪声 $\epsilon_\theta$ | Score $s_\theta$ | **向量场 $u_t^\theta$** |
| **损失函数** | $\|\epsilon - \epsilon_\theta\|^2$ | 同左 | $\lambda\|s_\theta - \nabla\log p\|^2$ | **$\|u_t^\theta - u^{\text{target}}\|^2$** |
| **噪声调度** | ⚠️ 需设计 $\beta_t$ | 复用 DDPM | ⚠️ 需设计 $g(t)$ | **❌ 无需设计！** |
| **辅助变量** | $\alpha_t, \bar{\alpha}_t, \tilde{\beta}_t...$ | 同左 + $\eta$ | $f, g, \lambda$ | **无！** |
| **采样步数** | 1000 | 50 | 可变 | **5-20** |
| **采样方式** | 随机 SDE | 确定 ODE ($\eta=0$) | SDE 或 ODE | **纯 ODE** |
| **理论复杂度** | 高 | 中 | 高 | **低** |

### 4.2 损失函数演变

```
DDPM:       L = E[ ‖ε - εθ(√ᾱ·x₀ + √(1-ᾱ)·ε, t)‖² ]
               ↓ 需要 ᾱ_t = ∏(1-β_s)，需要设计 β₁,...,β_T
               ↓ 预测噪声

DDIM:       L = E[ ‖ε - εθ(...)‖² ]        ← 同 DDPM
               ↓ 但采样时去掉随机项

Score SDE:   L = E[ λ(t)·‖sθ(x,t) - ∇log p₀ₜ(x|x₀)‖² ]
               ↓ 需要设计 g(t), λ(t)
               ↓ 预测 score

Flow Match:  L = E[ ‖uθ(tz + (1-t)ε) - (z - ε)‖² ]
               ↓ 无需任何超参数！
               ↓ 直接回归向量场
```

### 4.3 变量数量对比

| 方法 | 必须设计的超参数 | 辅助变量数 | 网络输出含义 |
|------|------------------|------------|-------------|
| DDPM | $\beta_1, \ldots, \beta_T$ | ~5 个 ($\alpha, \bar{\alpha}, \tilde{\beta}...$) | 预测噪声 |
| DDIM | 复用 DDPM 的 | ~6 个 (+$\eta$, $\hat{x}_0$) | 预测噪声 |
| Score SDE | $g(t), \lambda(t)$ | ~4 个 ($f, g, s$...) | 预测 score |
| **Flow Matching** | **无** | **0** | **回归向量场** |

---

## 五、为什么 Flow Matching 更好？

### 5.1 简洁性优势

**DDPM 的完整公式链：**
$$\begin{cases}
\alpha_t = 1 - \beta_t \\
\bar{\alpha}_t = \prod_{s=1}^t \alpha_s \\
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon \\
\mu_\theta = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta) \\
\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t \\
x_{t-1} = \mu_\theta + \sqrt{\tilde{\beta}_t}\,z
\end{cases}
$$

**Flow Matching 的完整公式链：**
$$\begin{cases}
x = tz + (1-t)\epsilon \\
\text{target} = z - \epsilon \\
\text{loss} = \|u_t^\theta(x) - \text{target}\|^2 \\
x_{t+h} = x_t + h \cdot u_t^\theta(x_t)
\end{cases}
$$

### 5.2 无超参数的优势

| 问题 | DDPM/Score SDE | Flow Matching |
|------|---------------|---------------|
| "如何选择 $\beta_t$？" | 需要实验调参 | 不存在此问题 |
| "线性还是余弦调度？" | 需要尝试 | 不需要 |
| "调度对结果影响大吗？" | 影响很大 | 无影响 |
| "不同任务需要重新设计吗？" | 通常需要 | **不需要** |

### 5.3 灵活性优势

**Flow Matching 可以轻松处理任意源分布和目标分布之间的转换：**

```python
path = LinearConditionalProbabilityPath(
    p_simple = CirclesSampleable(),    # 任意源分布
    p_data = CheckerboardSampleable()  # 任意目标分布
)

# 训练完全相同！无需修改任何代码
trainer.train(model, path)
```

这在 DDPM 中很难做到（需要为每对分布重新设计噪声调度）。

---

## 六、Flow Matching 的完整变量表

| 变量 | 类型 | 定义域 | 说明 |
|------|------|--------|------|
| $z$ | 数据向量 | $\mathbb{R}^d$ | 目标数据（条件变量），来自 $p_{\text{data}}$ |
| $x$ | 向量 | $\mathbb{R}^d$ | 当前位置（插值后的混合样本） |
| $t$ | 时间 | $[0, 1]$ | 插值系数（0=纯噪声, 1=纯数据） |
| $x_0$ | 噪声向量 | $\mathbb{R}^d$ | 来自简单分布 $p_{\text{simple}}$（通常 $\mathcal{N}(0,I)$） |
| $\epsilon$ | 噪声向量 | $\mathbb{R}^d$ | 标准高斯噪声 $\mathcal{N}(0, I)$ |
| $p_{\text{simple}}$ | 分布 | — | 简单分布（通常是标准高斯） |
| $p_{\text{data}}$ | 分布 | — | 数据分布 |
| $p_t(x\|z)$ | 条件密度 | — | 给定 $z$ 时 $x$ 在时刻 $t$ 的分布 |
| $u_t^{\text{target}}(x\|z)$ | 向量场 | $\mathbb{R}^d \times [0,1] \times \mathbb{R}^d \to \mathbb{R}^d$ | 真实的条件向量场（解析可得） |
| $u_t^\theta(x)$ | 神经网络 | $\mathbb{R}^d \times [0,1] \to \mathbb{R}^d$ | 学习到的向量场 |
| $\theta$ | 参数集 | — | 神经网络的全部可学习参数 |
| $\alpha_t$ | 插值系数 | $[0,1]$ | 通常 $\alpha_t = t$ |
| $\beta_t$ | 插值系数 | $[0,1]$ | 通常 $\beta_t = 1-t$ |

---

## 七、Flow Matching 的优缺点总结

### ✅ 优点

| 优点 | 说明 |
|------|------|
| **极简公式** | 无需 $\alpha$, $\beta$, $\bar{\alpha}$ 等辅助变量 |
| **无超参数** | 不需要设计噪声调度 |
| **通用性强** | 支持任意源/目标分布对 |
| **速度快** | 5-20 步即可高质量生成 |
| **确定性** | 相同输入 → 相同输出 |
| **易于实现** | 几十行代码即可完成训练 |
| **理论优美** | 条件损失 = 边际损失的隐式优化 |

### ❌ 缺点

| 缺点 | 说明 |
|------|------|
| **相对较新** | 生态系统不如 DDPM 成熟 |
| **多样性有限** | 确定性输出可能缺乏多样性（可通过加噪声缓解） |
| **直线假设** | 默认线性插值可能不是最优路径（Rectified Flow 改进此问题） |

---

## 八、四代演进全景总结

### 8.1 发展脉络图

```
2020                          2021                           2021                         2022+
  DDPM                          DDIM                        Score SDE                     Flow Matching
  ┌─────────┐                 ┌─────────┐                  ┌──────────┐                ┌──────────────┐
  │离散马尔可夫│                │非马尔可夫│                   │连续SDE/ODE│                │连续ODE       │
  │         │                 │         │                  │          │                │              │
  │L=‖ε-εθ‖²│                 │L=‖ε-εθ‖²│                  │L=λ‖s-∇logp‖²│               │L=‖u-u*‖²    │
  │         │                 │         │                  │          │                │              │
  │学:噪声   │                 │学:噪声   │                  │学:score   │                │学:向量场     │
  │         │                 │         │                  │          │                │              │
  │1000步   │  去掉随机       │50步     │  统一+连续化       │可变步数   │  彻底简化        │5-20步        │
  │         │  +加速          │         │                  │          │  +去超参数        │              │
  │设计β_t  │                 │复用β_t  │                  │设计g(t)   │                │无！          │
  └─────────┘                 └─────────┘                  └──────────┘                └──────────────┘
                                                                                              
  开创扩散模型时代               加速采样                    统一理论框架                   主流范式
```

### 8.2 核心改进总结

| 迭代 | 核心改进 | 本质变化 | 公式行数 |
|------|----------|----------|----------|
| **DDPM → DDIM** | 去掉随机项 $\sigma_t z$ | SDE → ODE | 减少 ~30% |
| **DDPM → Score** | 统一到 SDE 框架 | 离散 → 连续 | 增加（更抽象） |
| **Score → Flow** | 直接回归向量场 | 复杂 → **简洁** | **减少 ~70%** |
| **DDPM → Flow** | 全部改进 | 最简最优 | **减少 ~80%** |

### 8.3 一句话总结四代演进

> **DDPM 证明了扩散模型可行 → DDIM 发现可以去掉随机性加速 → Score SDE 统一了所有方法的理论基础 → Flow Matching 将一切简化为最纯粹的回归问题。**
>
> **每一次迭代的本质都是：让公式更短、超参数更少、速度更快。**
